"""
Portable realtime and offline inference entry point.

Usage:
    python -m imu_inference
    python -m imu_inference --no-gui
    python -m imu_inference --csv path/to/dialogue.csv
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from imu_inference.predict import (
    DecodeConfig,
    IDLE_LABEL,
    TOTAL_FEATURES,
    InferenceEngine,
    collapse_reference_labels,
    load_csv_features,
)
from imu_inference.receiver import BNO_COUNT, FEATURES_PER_IMU, DualHandReceiver

MODEL_DIR = _SRC.parent / "model"

RIGHT_HOST = "192.168.31.124"
RIGHT_PORT = 8911
LEFT_HOST = "192.168.31.135"
LEFT_PORT = 8913


def gyro_norm_signals(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != TOTAL_FEATURES:
        return np.empty((0, 0), dtype=np.float32)
    signals = np.empty((BNO_COUNT, len(sequence)), dtype=np.float32)
    for imu_idx in range(BNO_COUNT):
        base = imu_idx * FEATURES_PER_IMU + 7
        signals[imu_idx] = np.linalg.norm(sequence[:, base : base + 3], axis=1)
    return signals


class LiveController:
    def __init__(self, engine: InferenceEngine, args: argparse.Namespace):
        self.engine = engine
        self.args = args
        self.receiver = DualHandReceiver(
            args.right_host,
            args.right_port,
            args.left_host,
            args.left_port,
            capacity=max(args.tail_frames * 2, 2048),
        )

        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.history: deque[np.ndarray] = deque(maxlen=args.tail_frames)

        self.sentence: list[str] = []
        self.tokens: list[dict] = []
        self.last_prediction: dict | None = None
        self.last_commit_end = -1
        self.last_commit_label = ""

        self.total_frames = 0
        self.total_inferences = 0
        self.fps = 0.0
        self.decode_config = DecodeConfig(
            min_confidence=args.min_confidence,
            smooth_radius=args.smooth_radius,
            min_token_frames=args.min_token_frames,
            merge_gap=args.merge_gap,
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.receiver.start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.receiver.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def reset(self) -> None:
        with self.lock:
            self.history.clear()
            self.sentence.clear()
            self.tokens.clear()
            self.last_prediction = None
            self.last_commit_end = -1
            self.last_commit_label = ""
            self.total_frames = 0
            self.total_inferences = 0
            self.fps = 0.0
        self.receiver.clear()

    def recent_signal(self, n_frames: int = 800) -> np.ndarray:
        with self.lock:
            if len(self.history) < 2:
                return np.empty((0, TOTAL_FEATURES), dtype=np.float32)
            rows = list(self.history)[-n_frames:]
        return np.asarray(rows, dtype=np.float32)

    def sentence_text(self) -> str:
        with self.lock:
            return " ".join(self.sentence)

    def status(self) -> dict:
        with self.lock:
            frames = len(self.history)
            total = self.total_frames
            inferences = self.total_inferences
            words = len(self.sentence)
            pred = None if self.last_prediction is None else dict(self.last_prediction)
            tokens = [dict(t) for t in self.tokens]
        return {
            "connected": self.receiver.connected,
            "receiver": self.receiver.status,
            "frames": frames,
            "total_frames": total,
            "inferences": inferences,
            "words": words,
            "prediction": pred,
            "tokens": tokens,
            "sentence": self.sentence_text(),
            "fps": self.fps,
        }

    def _loop(self) -> None:
        last_received = 0
        last_infer_total = 0
        fps_timer = time.monotonic()
        fps_count = 0

        while self.running:
            rows, last_received = self.receiver.drain_since(last_received)
            if rows:
                with self.lock:
                    self.history.extend(rows)
                    self.total_frames += len(rows)
                    total_frames = self.total_frames
                    history_len = len(self.history)
                fps_count += len(rows)
            else:
                with self.lock:
                    total_frames = self.total_frames
                    history_len = len(self.history)

            now = time.monotonic()
            if now - fps_timer >= 1.0:
                self.fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            if history_len < self.args.window_len or total_frames - last_infer_total < self.args.step:
                time.sleep(0.01)
                continue

            with self.lock:
                seq = np.asarray(self.history, dtype=np.float32)
                seq_global_start = self.total_frames - len(seq)

            try:
                tokens = self.engine.predict_continuous(
                    seq,
                    window_len=self.args.window_len,
                    step=self.args.step,
                    decode_config=self.decode_config,
                )
            except Exception as exc:
                with self.lock:
                    self.last_prediction = {"label": "推理错误", "confidence": 0.0, "error": str(exc)}
                time.sleep(0.2)
                continue

            committed = []
            visible_prediction = None
            latest_safe_end = len(seq) - self.args.commit_lag_frames
            for token in tokens:
                if token["end"] > latest_safe_end:
                    visible_prediction = token
                    continue

                global_start = seq_global_start + token["start"]
                global_end = seq_global_start + token["end"]
                if global_end <= self.last_commit_end:
                    continue
                if (
                    token["label"] == self.last_commit_label
                    and global_start - self.last_commit_end < self.args.repeat_gap_frames
                ):
                    continue

                item = dict(token)
                item["global_start"] = int(global_start)
                item["global_end"] = int(global_end)
                committed.append(item)
                self.last_commit_end = int(global_end)
                self.last_commit_label = token["label"]

            with self.lock:
                self.total_inferences += 1
                if visible_prediction is not None:
                    self.last_prediction = {
                        "label": visible_prediction["label"],
                        "confidence": visible_prediction["confidence"],
                    }
                elif committed:
                    self.last_prediction = {
                        "label": committed[-1]["label"],
                        "confidence": committed[-1]["confidence"],
                    }
                for item in committed:
                    self.sentence.append(item["label"])
                    self.tokens.append(item)

            last_infer_total = total_frames
            time.sleep(0.005)


def run_csv(engine: InferenceEngine, args: argparse.Namespace) -> None:
    features, labels = load_csv_features(args.csv)
    config = DecodeConfig(
        min_confidence=args.min_confidence,
        smooth_radius=args.smooth_radius,
        min_token_frames=args.min_token_frames,
        merge_gap=args.merge_gap,
    )
    tokens = engine.predict_continuous(features, args.window_len, args.step, config)
    hyp = [token["label"] for token in tokens]
    ref = collapse_reference_labels(labels) if labels else []

    print(f"模型: {engine.describe()}")
    print(f"CSV: {args.csv}")
    print(f"帧数: {len(features)}, 特征: {features.shape[1]}")
    if ref:
        print(f"参考: {' '.join(ref)}")
    print(f"识别: {' '.join(hyp) if hyp else '(无)'}")
    print("Token:")
    for token in tokens:
        print(
            f"  {token['label']:6s} "
            f"{token['start']:4d}-{token['end']:4d} "
            f"{token['confidence']:.1%}"
        )


def run_terminal(engine: InferenceEngine, args: argparse.Namespace) -> None:
    controller = LiveController(engine, args)
    controller.start()
    print(f"模型: {engine.describe()}")
    print(f"右手: {args.right_host}:{args.right_port}")
    print(f"左手: {args.left_host}:{args.left_port}")
    print("Ctrl+C 退出")
    last_count = 0
    try:
        while True:
            status = controller.status()
            if not status["connected"]:
                print(f"\r等待连接: {status['receiver']}", end="", flush=True)
            elif len(status["tokens"]) != last_count:
                last_count = len(status["tokens"])
                pred = status["tokens"][-1]
                print(
                    f"\n{pred['label']} {pred['confidence']:.1%} "
                    f"| 句子: {status['sentence']}"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        print(f"\n最终结果: {controller.sentence_text()}")


def run_gui(engine: InferenceEngine, args: argparse.Namespace) -> None:
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    pg.setConfigOptions(antialias=True)
    controller = LiveController(engine, args)

    imu_names = [
        "右手背",
        "右尾指",
        "右无名指",
        "右中指",
        "右食指",
        "右拇指",
        "左拇指",
        "左食指",
        "左中指",
        "左无名指",
        "左尾指",
        "左手背",
    ]
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#15803d",
        "#7c3aed",
        "#ca8a04",
        "#0f766e",
    ]

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("手语实时识别")
            self.setMinimumSize(1220, 760)
            self.setStyleSheet(
                """
                QMainWindow { background: #f7f8fa; }
                QLabel { color: #111827; }
                QPushButton {
                    background: #2563eb; color: white; border: 0;
                    padding: 8px 18px; border-radius: 4px; font-size: 14px;
                }
                QPushButton:disabled { background: #9ca3af; }
                QGroupBox {
                    border: 1px solid #d1d5db; border-radius: 6px;
                    margin-top: 8px; padding-top: 14px; font-weight: 600;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
                QLineEdit {
                    background: white; border: 1px solid #d1d5db; border-radius: 4px;
                    padding: 10px; font-size: 24px;
                }
                QListWidget { background: white; border: 1px solid #d1d5db; border-radius: 4px; }
                """
            )

            root_widget = QtWidgets.QWidget()
            self.setCentralWidget(root_widget)
            root = QtWidgets.QVBoxLayout(root_widget)
            root.setContentsMargins(14, 12, 14, 12)
            root.setSpacing(10)

            toolbar = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("开始")
            self.stop_btn = QtWidgets.QPushButton("停止")
            self.reset_btn = QtWidgets.QPushButton("重置")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("QPushButton { background: #dc2626; }")
            self.reset_btn.setStyleSheet("QPushButton { background: #4b5563; }")
            self.status_label = QtWidgets.QLabel("就绪")
            self.status_label.setStyleSheet("color: #6b7280; font-size: 13px;")
            toolbar.addWidget(self.start_btn)
            toolbar.addWidget(self.stop_btn)
            toolbar.addWidget(self.reset_btn)
            toolbar.addStretch()
            toolbar.addWidget(self.status_label)
            root.addLayout(toolbar)

            self.plot = pg.PlotWidget(title="IMU 陀螺仪范数")
            self.plot.setMinimumHeight(260)
            self.plot.showGrid(x=True, y=True, alpha=0.25)
            self.plot.setLabel("bottom", "帧")
            self.plot.setLabel("left", "gyro norm")
            self.plot.addLegend(offset=(8, 8))
            self.curves = []
            for i, name in enumerate(imu_names):
                curve = self.plot.plot([], [], pen=pg.mkPen(colors[i], width=1.4), name=name)
                self.curves.append(curve)
            root.addWidget(self.plot, stretch=3)

            bottom = QtWidgets.QHBoxLayout()
            root.addLayout(bottom, stretch=3)

            left = QtWidgets.QVBoxLayout()
            bottom.addLayout(left, stretch=3)
            current_box = QtWidgets.QGroupBox("当前识别")
            current_layout = QtWidgets.QVBoxLayout(current_box)
            self.current_label = QtWidgets.QLabel("-")
            self.current_label.setAlignment(QtCore.Qt.AlignCenter)
            self.current_label.setStyleSheet("font-size: 64px; font-weight: 700; color: #1d4ed8;")
            self.conf_label = QtWidgets.QLabel("等待输入")
            self.conf_label.setAlignment(QtCore.Qt.AlignCenter)
            self.conf_label.setStyleSheet("font-size: 15px; color: #6b7280;")
            current_layout.addWidget(self.current_label)
            current_layout.addWidget(self.conf_label)
            left.addWidget(current_box, stretch=2)

            sentence_box = QtWidgets.QGroupBox("识别句子")
            sentence_layout = QtWidgets.QVBoxLayout(sentence_box)
            self.sentence = QtWidgets.QLineEdit()
            self.sentence.setReadOnly(True)
            sentence_layout.addWidget(self.sentence)
            left.addWidget(sentence_box, stretch=1)

            right = QtWidgets.QVBoxLayout()
            bottom.addLayout(right, stretch=2)
            info_box = QtWidgets.QGroupBox("连接信息")
            info_layout = QtWidgets.QVBoxLayout(info_box)
            self.info_labels = {
                "right": QtWidgets.QLabel(f"右手: {args.right_host}:{args.right_port}"),
                "left": QtWidgets.QLabel(f"左手: {args.left_host}:{args.left_port}"),
                "frames": QtWidgets.QLabel("帧数: 0"),
                "infer": QtWidgets.QLabel("推理: 0"),
                "fps": QtWidgets.QLabel("帧率: 0 fps"),
                "model": QtWidgets.QLabel(engine.describe()),
            }
            for label in self.info_labels.values():
                label.setStyleSheet("font-size: 12px;")
                info_layout.addWidget(label)
            right.addWidget(info_box, stretch=2)

            history_box = QtWidgets.QGroupBox("识别历史")
            history_layout = QtWidgets.QVBoxLayout(history_box)
            self.history = QtWidgets.QListWidget()
            history_layout.addWidget(self.history)
            right.addWidget(history_box, stretch=3)

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(50)
            self.timer.timeout.connect(self.tick)
            self.start_btn.clicked.connect(self.start)
            self.stop_btn.clicked.connect(self.stop)
            self.reset_btn.clicked.connect(self.reset)

        def start(self) -> None:
            controller.start()
            self.timer.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("运行中")
            self.status_label.setStyleSheet("color: #16a34a; font-size: 13px;")

        def stop(self) -> None:
            controller.stop()
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("已停止")
            self.status_label.setStyleSheet("color: #dc2626; font-size: 13px;")

        def reset(self) -> None:
            controller.reset()
            self.history.clear()
            self.sentence.clear()
            self.current_label.setText("-")
            self.conf_label.setText("等待输入")
            for curve in self.curves:
                curve.setData([], [])

        def tick(self) -> None:
            status = controller.status()
            connected_color = "#16a34a" if status["connected"] else "#dc2626"
            self.info_labels["right"].setStyleSheet(f"font-size: 12px; color: {connected_color};")
            self.info_labels["left"].setStyleSheet(f"font-size: 12px; color: {connected_color};")
            self.info_labels["frames"].setText(f"帧数: {status['total_frames']}")
            self.info_labels["infer"].setText(f"推理: {status['inferences']}")
            self.info_labels["fps"].setText(f"帧率: {status['fps']:.0f} fps")
            self.sentence.setText(status["sentence"])

            pred = status["prediction"]
            if pred and pred.get("label") and pred["label"] != IDLE_LABEL:
                conf = float(pred.get("confidence", 0.0))
                color = "#1d4ed8" if conf >= 0.85 else "#d97706"
                self.current_label.setText(pred["label"])
                self.current_label.setStyleSheet(f"font-size: 64px; font-weight: 700; color: {color};")
                self.conf_label.setText(f"置信度 {conf:.1%}")
            else:
                self.current_label.setText("-")
                self.current_label.setStyleSheet("font-size: 64px; font-weight: 700; color: #9ca3af;")
                self.conf_label.setText("等待输入")

            if len(status["tokens"]) != self.history.count():
                self.history.clear()
                for token in status["tokens"]:
                    self.history.addItem(f"{token['label']}  {token['confidence']:.1%}")
                self.history.scrollToBottom()

            seq = controller.recent_signal(800)
            signals = gyro_norm_signals(seq)
            for i, curve in enumerate(self.curves):
                if i < len(signals):
                    curve.setData(np.arange(signals.shape[1]), signals[i])

        def closeEvent(self, event) -> None:
            controller.stop()
            event.accept()

    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable IMU sign-language inference")
    parser.add_argument("--csv", default=None, help="processed CSV file for offline inference")
    parser.add_argument("--no-gui", action="store_true", help="run terminal realtime mode")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="directory with model.pt and normalization.json")
    parser.add_argument("--device", default=None, help="cpu / cuda; default auto")

    parser.add_argument("--right-host", default=RIGHT_HOST)
    parser.add_argument("--right-port", type=int, default=RIGHT_PORT)
    parser.add_argument("--left-host", default=LEFT_HOST)
    parser.add_argument("--left-port", type=int, default=LEFT_PORT)

    parser.add_argument("--window-len", type=int, default=128)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--tail-frames", type=int, default=640)
    parser.add_argument("--commit-lag-frames", type=int, default=12)
    parser.add_argument("--repeat-gap-frames", type=int, default=24)

    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--smooth-radius", type=int, default=2)
    parser.add_argument("--min-token-frames", type=int, default=16)
    parser.add_argument("--merge-gap", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decode_config = DecodeConfig(
        min_confidence=args.min_confidence,
        smooth_radius=args.smooth_radius,
        min_token_frames=args.min_token_frames,
        merge_gap=args.merge_gap,
    )
    engine = InferenceEngine(args.model_dir, args.device, decode_config)

    if args.csv:
        run_csv(engine, args)
    elif args.no_gui:
        run_terminal(engine, args)
    else:
        run_gui(engine, args)


if __name__ == "__main__":
    main()
