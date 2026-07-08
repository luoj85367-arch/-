"""CSV 回放查看器 —— 双设备 12 路 BNO055 四元数静态浏览

从已有 CSV 文件读取数据，在原双面板布局中静态浏览：
  左侧面板：设备2（右手 IMU0-5）
  右侧面板：设备1（左手 IMU6-11）

顶部：
  [选择 CSV 文件]  显示文件名、总帧数
  滚动条：拖动查看任意时段（窗口宽度 = SCROLL_WINDOW 帧）

空格键不再使用；支持鼠标拖动 Y 轴缩放。
"""

import sys
import csv
import os
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ===================== 显示参数 =====================
BNO_COUNT       = 6
TOTAL_IMU       = BNO_COUNT * 2
SCROLL_WINDOW   = 500       # 每次显示的帧数（窗口宽度）
PLOT_REFRESH_HZ = 60

QUAT_LABELS = ["w", "x", "y", "z"]
QUAT_COLORS = [
    (0,   0,   0),
    (220, 50,  50),
    (34,  160, 60),
    (50,  120, 220),
]
DEVICE_LABELS = ["右手 (设备2)", "左手 (设备1)"]

# IMU 编号 → 部位名称（按图表对照表）
# 右手 dev0: IMU0~5  (端口8911)
# 左手 dev1: IMU6~11 (端口8913)
IMU_PART_NAMES = {
    0:  "手背",
    1:  "尾指",
    2:  "无名指",
    3:  "中指",
    4:  "食指",
    5:  "拇指",
    6:  "拇指",
    7:  "食指",
    8:  "中指",
    9:  "无名指",
    10: "尾指",
    11: "手背",
}

# CSV 列名前缀映射：imu0~5 属于 dev0（右手），imu6~11 属于 dev1（左手）
# 每列格式: imu{N}_qw / qx / qy / qz


def load_csv(filepath: str):
    """
    读取 CSV 文件，返回 quats 数组：shape = (N_frames, 12, 4)
    列顺序与原采集格式匹配：
      seq_0, ts_0, imu0_qw..., imu5_qgz, seq_1, ts_1, imu6_qw..., imu11_qgz
    """
    rows = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    if not rows:
        return None, header, 0

    # 建立列名 → 索引映射
    col_idx = {name: i for i, name in enumerate(header)}

    n_frames = len(rows)
    quats = np.zeros((n_frames, TOTAL_IMU, 4), dtype=np.float32)

    for frame_i, row in enumerate(rows):
        for imu_j in range(TOTAL_IMU):
            for k, comp in enumerate(["qw", "qx", "qy", "qz"]):
                col_name = f"imu{imu_j}_{comp}"
                if col_name in col_idx:
                    try:
                        quats[frame_i, imu_j, k] = float(row[col_idx[col_name]])
                    except (ValueError, IndexError):
                        quats[frame_i, imu_j, k] = 0.0

    return quats, header, n_frames


# ===================== 主窗口 =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU CSV 数据查看器 —— 双设备 12 路")
        self.setMinimumSize(1400, 900)
        self.resize(1800, 1000)

        self._quats      = None   # (N, 12, 4)
        self._n_frames   = 0
        self._current_pos = 0     # 滚动条当前起始帧

        # ---- 中央 Widget ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        # ---- 顶部工具栏 ----
        toolbar = QtWidgets.QHBoxLayout()

        self.btn_open = QtWidgets.QPushButton("📂  选择 CSV 文件")
        self.btn_open.setFixedHeight(32)
        self.btn_open.setStyleSheet(
            "QPushButton { font-size: 10pt; padding: 4px 16px; "
            "background: #1a6bb5; color: white; border-radius: 4px; } "
            "QPushButton:hover { background: #145a9a; } "
            "QPushButton:pressed { background: #0f4070; }"
        )
        toolbar.addWidget(self.btn_open)
        toolbar.addSpacing(12)

        self.lbl_file = QtWidgets.QLabel("未加载文件")
        self.lbl_file.setStyleSheet("font-size: 9pt; color: #555;")
        toolbar.addWidget(self.lbl_file)
        toolbar.addSpacing(20)

        # 图例
        for j, (label, color) in enumerate(zip(QUAT_LABELS, QUAT_COLORS)):
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(13, 13)
            r, g, b = color
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #999; border-radius: 2px;"
            )
            toolbar.addWidget(swatch)
            text = QtWidgets.QLabel(label)
            text.setStyleSheet("font-size: 9pt; color: #333;")
            toolbar.addWidget(text)
            if j < 3:
                toolbar.addSpacing(8)

        toolbar.addSpacing(12)
        toolbar.addStretch()
        self.lbl_status = QtWidgets.QLabel("请先打开 CSV 文件")
        self.lbl_status.setStyleSheet("font-size: 9pt; color: #888;")
        toolbar.addWidget(self.lbl_status)
        main_layout.addLayout(toolbar)

        # ---- 滚动条（帧偏移）----
        scroll_bar_row = QtWidgets.QHBoxLayout()
        lbl_scroll = QtWidgets.QLabel("时间轴：")
        lbl_scroll.setStyleSheet("font-size: 9pt; color: #444;")
        scroll_bar_row.addWidget(lbl_scroll)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setPageStep(SCROLL_WINDOW)
        self.scrollbar.setEnabled(False)
        scroll_bar_row.addWidget(self.scrollbar, stretch=1)

        self.lbl_frame_pos = QtWidgets.QLabel("帧: -")
        self.lbl_frame_pos.setStyleSheet("font-size: 9pt; color: #555; min-width: 120px;")
        scroll_bar_row.addWidget(self.lbl_frame_pos)
        main_layout.addLayout(scroll_bar_row)

        # ---- pyqtgraph 配置 ----
        pg.setConfigOptions(antialias=True, useOpenGL=True)

        # ---- 左右两列布局 ----
        panels_layout = QtWidgets.QHBoxLayout()
        panels_layout.setSpacing(4)
        main_layout.addLayout(panels_layout, stretch=1)

        self.plots  = [[] for _ in range(2)]
        self.curves = [[] for _ in range(2)]

        for dev_id, dev_label in enumerate(DEVICE_LABELS):
            frame = QtWidgets.QGroupBox(dev_label)
            frame.setStyleSheet(
                "QGroupBox { font-size: 11pt; font-weight: bold; color: #1a6bb5; "
                "border: 2px solid #1a6bb5; border-radius: 4px; margin-top: 6px; } "
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            )
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(2, 10, 2, 2)
            frame_layout.setSpacing(0)

            gl_widget = pg.GraphicsLayoutWidget()
            gl_widget.setBackground("w")
            frame_layout.addWidget(gl_widget)
            panels_layout.addWidget(frame, stretch=1)

            for i in range(BNO_COUNT):
                global_imu_idx = dev_id * BNO_COUNT + i
                part_name = IMU_PART_NAMES.get(global_imu_idx, "")
                title_str = f"IMU{global_imu_idx}  {part_name}  Quaternion"
                plot = gl_widget.addPlot(title=title_str)
                plot.showGrid(x=True, y=True, alpha=0.15)
                plot.setLabel("left",   "Value")
                plot.setLabel("bottom", "Samples")
                plot.setTitle(title_str, color="#333", size="10pt")
                plot.getAxis("left").setPen(pg.mkPen("#999"))
                plot.getAxis("bottom").setPen(pg.mkPen("#999"))
                plot.getAxis("left").setTextPen(pg.mkPen("#555"))
                plot.getAxis("bottom").setTextPen(pg.mkPen("#555"))
                plot.enableAutoRange(axis='x', enable=False)
                plot.setXRange(0, SCROLL_WINDOW, padding=0)
                plot.enableAutoRange(axis='y', enable=False)
                plot.setYRange(-1.1, 1.1, padding=0)
                plot.setMouseEnabled(x=False, y=True)
                plot.setDownsampling(auto=True, mode='peak')
                plot.setClipToView(True)

                curves_for_imu = []
                for j in range(4):
                    curve = plot.plot(pen=pg.mkPen(color=QUAT_COLORS[j], width=1.5))
                    curves_for_imu.append(curve)
                self.plots[dev_id].append(plot)
                self.curves[dev_id].append(curves_for_imu)

                if i < BNO_COUNT - 1:
                    gl_widget.nextRow()

        # ---- 信号连接 ----
        self.btn_open.clicked.connect(self._on_open_file)
        self.scrollbar.valueChanged.connect(self._on_scroll)

    # ---------- 打开文件 ----------
    def _on_open_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 IMU 数据 CSV 文件",
            str(Path.home()),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not filepath:
            return

        self.lbl_status.setText("正在加载…")
        QtWidgets.QApplication.processEvents()

        try:
            quats, header, n_frames = load_csv(filepath)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败", f"无法读取文件：\n{e}")
            self.lbl_status.setText("加载失败")
            return

        if quats is None or n_frames == 0:
            QtWidgets.QMessageBox.warning(self, "文件为空", "CSV 文件中没有数据行。")
            self.lbl_status.setText("文件为空")
            return

        self._quats    = quats
        self._n_frames = n_frames

        # 更新 UI
        filename = Path(filepath).name
        self.lbl_file.setText(f"文件：{filename}   总帧数：{n_frames}")
        self.setWindowTitle(f"IMU CSV 查看器 — {filename}")

        # 配置滚动条
        max_scroll = max(0, n_frames - SCROLL_WINDOW)
        self.scrollbar.setMaximum(max_scroll)
        self.scrollbar.setValue(0)
        self.scrollbar.setEnabled(True)

        # 渲染第一窗
        self._current_pos = 0
        self._render(0)
        self.lbl_status.setText(f"已加载 {n_frames} 帧 | 滚动条拖动查看")

    # ---------- 滚动条事件 ----------
    def _on_scroll(self, value):
        if self._quats is None:
            return
        self._current_pos = value
        self._render(value)

    # ---------- 渲染当前窗口 ----------
    def _render(self, start_frame: int):
        end_frame = min(start_frame + SCROLL_WINDOW, self._n_frames)
        segment   = self._quats[start_frame:end_frame]  # (win, 12, 4)
        win_size  = segment.shape[0]

        # 构建完整宽度的 nan 缓冲，不足 SCROLL_WINDOW 则左侧补 nan
        buf = np.full((SCROLL_WINDOW, TOTAL_IMU, 4), np.nan, dtype=np.float32)
        buf[SCROLL_WINDOW - win_size:] = segment
        x_axis = np.arange(SCROLL_WINDOW)

        for dev_id in range(2):
            for i in range(BNO_COUNT):
                imu_global = dev_id * BNO_COUNT + i
                curve_row  = self.curves[dev_id][i]
                for j in range(4):
                    curve_row[j].setData(x_axis, buf[:, imu_global, j])

        self.lbl_frame_pos.setText(
            f"帧: {start_frame} – {end_frame - 1}  /  {self._n_frames}"
        )

    # ---------- 键盘 ----------
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_Left:
            self.scrollbar.setValue(max(0, self.scrollbar.value() - SCROLL_WINDOW // 2))
        elif event.key() == QtCore.Qt.Key_Right:
            self.scrollbar.setValue(
                min(self.scrollbar.maximum(), self.scrollbar.value() + SCROLL_WINDOW // 2)
            )
        else:
            super().keyPressEvent(event)


# ===================== 入口 =====================
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
