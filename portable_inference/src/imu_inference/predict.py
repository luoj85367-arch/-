"""
Self-contained inference utilities for the portable IMU sign-language package.

The public surface is intentionally small:
- InferenceEngine: load model files and run continuous sequence inference.
- load_csv_features: read processed CSV files for offline tests.
- decode_predictions: convert frame probabilities into gesture tokens.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from imu_inference.model import IMUFrameNetV2

TOTAL_IMU_COUNT = 12
FEATURES_PER_IMU = 10
TOTAL_FEATURES = TOTAL_IMU_COUNT * FEATURES_PER_IMU
IDLE_LABEL = "静止"


@dataclass(frozen=True)
class DecodeConfig:
    min_confidence: float = 0.70
    smooth_radius: int = 2
    min_token_frames: int = 16
    merge_gap: int = 12


def feature_columns(headers: list[str]) -> list[int]:
    """Return CSV column indexes for imu0..imu11 qw/qx/qy/qz/accel/gyro features."""
    names = []
    for imu_idx in range(TOTAL_IMU_COUNT):
        prefix = f"imu{imu_idx}_"
        names.extend(
            [
                prefix + "qw",
                prefix + "qx",
                prefix + "qy",
                prefix + "qz",
                prefix + "ax",
                prefix + "ay",
                prefix + "az",
                prefix + "gx",
                prefix + "gy",
                prefix + "gz",
            ]
        )

    index = {name: i for i, name in enumerate(headers)}
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(f"CSV 缺少 IMU 特征列: {missing[:5]}...")
    return [index[name] for name in names]


def load_csv_features(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load processed CSV data. Returns (features, labels). Labels may be empty."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        feat_idx = feature_columns(headers)
        label_idx = headers.index("label") if "label" in headers else None

        rows: list[list[float]] = []
        labels: list[str] = []
        for row in reader:
            if not row:
                continue
            rows.append([float(row[i]) if row[i] else 0.0 for i in feat_idx])
            if label_idx is not None:
                labels.append(row[label_idx].strip())

    features = np.asarray(rows, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"CSV 特征维度错误: {features.shape}, 期望 (*, {TOTAL_FEATURES})")
    return features, labels


def collapse_reference_labels(labels: list[str]) -> list[str]:
    """Collapse frame labels into token labels for readable CSV test output."""
    tokens: list[str] = []
    prev = IDLE_LABEL
    for label in labels:
        label = IDLE_LABEL if label in ("", "0", IDLE_LABEL) else label
        if label != prev and label != IDLE_LABEL:
            tokens.append(label)
        prev = label
    return tokens


def compute_relative_quaternions(features: np.ndarray) -> np.ndarray:
    """
    Convert absolute quaternions to quaternions relative to the first frame.

    Acceleration and gyro channels are preserved. Quaternion channels are
    normalized before multiplication to tolerate mild sensor drift.
    """
    if len(features) == 0:
        return features.astype(np.float32, copy=True)

    result = features.astype(np.float32, copy=True)
    for imu_idx in range(TOTAL_IMU_COUNT):
        base = imu_idx * FEATURES_PER_IMU
        q = result[:, base : base + 4]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.where(q_norm < 1e-8, 1.0, q_norm)
        q = q / q_norm

        q0 = q[0].copy()
        q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]], dtype=np.float32)
        w0, x0, y0, z0 = q0_inv
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        result[:, base + 0] = w0 * w - x0 * x - y0 * y - z0 * z
        result[:, base + 1] = w0 * x + x0 * w + y0 * z - z0 * y
        result[:, base + 2] = w0 * y - x0 * z + y0 * w + z0 * x
        result[:, base + 3] = w0 * z + x0 * y - y0 * x + z0 * w
    return result


def smooth_probabilities(probs: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(probs) == 0:
        return probs
    smoothed = np.empty_like(probs)
    for t in range(len(probs)):
        start = max(0, t - radius)
        end = min(len(probs), t + radius + 1)
        smoothed[t] = probs[start:end].mean(axis=0)
    return smoothed


def decode_predictions(
    probs: np.ndarray,
    labels: list[str],
    config: DecodeConfig | None = None,
) -> list[dict]:
    """Decode frame-level class probabilities into stable non-idle gesture tokens."""
    config = config or DecodeConfig()
    if len(probs) == 0:
        return []

    smoothed = smooth_probabilities(probs, config.smooth_radius)
    frame_ids = np.argmax(smoothed, axis=1)
    frame_confs = np.max(smoothed, axis=1)
    frame_labels = [
        labels[int(frame_ids[i])] if frame_confs[i] >= config.min_confidence else IDLE_LABEL
        for i in range(len(smoothed))
    ]

    segments: list[dict] = []
    cur_label = frame_labels[0]
    cur_start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != cur_label:
            segments.append({"label": cur_label, "start": cur_start, "end": i})
            cur_label = frame_labels[i]
            cur_start = i
    segments.append({"label": cur_label, "start": cur_start, "end": len(frame_labels)})

    non_idle = [s for s in segments if s["label"] != IDLE_LABEL]
    merged: list[dict] = []
    for segment in non_idle:
        if (
            merged
            and segment["label"] == merged[-1]["label"]
            and segment["start"] - merged[-1]["end"] <= config.merge_gap
        ):
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(segment.copy())

    tokens: list[dict] = []
    for segment in merged:
        duration = segment["end"] - segment["start"]
        if duration < config.min_token_frames:
            continue
        conf = float(frame_confs[segment["start"] : segment["end"]].mean())
        tokens.append(
            {
                "label": segment["label"],
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "confidence": round(conf, 4),
            }
        )
    return tokens


class InferenceEngine:
    """Model loading, preprocessing, sliding-window inference and decoding."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        decode_config: DecodeConfig | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.decode_config = decode_config or DecodeConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint_path = self.model_dir / "model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = self.model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {self.model_dir}/model.pt")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.label_map = ckpt["label_map"]
        self.id_to_label = {int(k): v for k, v in ckpt["id_to_label"].items()}
        self.labels = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        self.model = IMUFrameNetV2(num_classes=len(self.labels)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.epoch = ckpt.get("epoch")
        self.val_acc = ckpt.get("val_acc")

        with (self.model_dir / "normalization.json").open("r", encoding="utf-8") as f:
            norm = json.load(f)
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def describe(self) -> str:
        score = f", val_acc={self.val_acc:.4f}" if isinstance(self.val_acc, float) else ""
        epoch = f", epoch={self.epoch}" if self.epoch is not None else ""
        return f"{len(self.labels)} 类, device={self.device}{epoch}{score}"

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != TOTAL_FEATURES:
            raise ValueError(f"特征维度错误: {features.shape}, 期望 (*, {TOTAL_FEATURES})")
        features = compute_relative_quaternions(features)
        return (features - self.mean) / self.std

    @torch.no_grad()
    def predict_probabilities(
        self,
        features: np.ndarray,
        window_len: int = 128,
        step: int = 8,
    ) -> np.ndarray:
        processed = self.preprocess(features)
        total = len(processed)
        if total == 0:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        probs_sum = np.zeros((total, len(self.labels)), dtype=np.float32)
        counts = np.zeros(total, dtype=np.float32)

        starts = list(range(0, max(total - window_len + 1, 1), step))
        if not starts or starts[-1] + window_len < total:
            starts.append(max(0, total - window_len))

        for start in starts:
            end = min(start + window_len, total)
            window = processed[start:end]
            if len(window) < window_len:
                pad = np.zeros((window_len - len(window), TOTAL_FEATURES), dtype=np.float32)
                window = np.concatenate([window, pad], axis=0)

            x = torch.from_numpy(window).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0][: end - start]
            probs_sum[start:end] += probs
            counts[start:end] += 1

        mask = counts > 0
        probs_sum[mask] /= counts[mask, None]
        return probs_sum

    def predict_continuous(
        self,
        features: np.ndarray,
        window_len: int = 128,
        step: int = 8,
        decode_config: DecodeConfig | None = None,
    ) -> list[dict]:
        probs = self.predict_probabilities(features, window_len=window_len, step=step)
        return decode_predictions(probs, self.labels, decode_config or self.decode_config)
