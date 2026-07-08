"""Direct IMU 采集 —— 双设备 12 路 BNO055 数据实时折线图 + CSV 存储。

启动后全屏显示两套设备各 6 个 IMU 的四元数 (w,x,y,z) 折线图（pyqtgraph）。
左侧面板：设备1（IMU0-5），右侧面板：设备2（IMU6-11）。
顶部左侧 [开始采集] / [停止采集] 按钮。
点击"开始采集"自动以序号命名（1.csv ~ 50.csv），两路数据合并保存在脚本同目录。
第 50 次采集停止后程序自动退出。空格键可切换采集/停止。

自动分段：采集开始后按 过渡段(TRANSITION_DURATION 秒) → 词语段(WORD_DURATION 秒) → 过渡段 ... 循环。
所有词语采集完后停留在过渡段直到手动停止。

线程设计：
  - TCP 接收线程 ×2：只管收包、解包，每一帧入 ring buffer + save queue
  - CSV 写入线程：从 save queue 取数据追加写入文件（两路合并）
  - GUI 线程：定时从两个 ring buffer 读最新数据刷新曲线
"""

import sys
import csv
import struct
import socket
import threading
import time
import queue
from pathlib import Path

# ===================== 采集序号参数 =====================
MAX_RECORDINGS = 50                              # 最多采集次数
SAVE_DIR = Path(__file__).parent.resolve()        # CSV 保存目录（脚本同目录）

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ===================== 通信参数 =====================
# 设备1（左手）
DEVICE1_IP   = "192.168.31.135"
DEVICE1_PORT = 8913
# 设备2（右手）
DEVICE2_IP   = "192.168.31.75"
DEVICE2_PORT = 8915

BUFFER_SIZE        = 65536
RECONNECT_INTERVAL = 1.0

# ===================== 自动分段时间参数 =====================
TRANSITION_DURATION = 0.7   # 过渡段持续秒数
WORD_DURATION       = 1.5   # 词语段持续秒数


# ===================== 传感器参数（单台设备）=====================
BNO_COUNT          = 6
QUAT_FLOAT_COUNT   = BNO_COUNT * 4          # 24
VECTOR_FLOAT_COUNT = BNO_COUNT * 3          # 18
SENSOR_FLOAT_COUNT = QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT * 2  # 60

PACKET_HEADER_FORMAT = "<IQ"
PACKET_HEADER_SIZE   = struct.calcsize(PACKET_HEADER_FORMAT)     # 12
PACKET_FLOAT_FORMAT  = f"<{SENSOR_FLOAT_COUNT}f"
PACKET_SIZE = PACKET_HEADER_SIZE + struct.calcsize(PACKET_FLOAT_FORMAT)  # 252

# ===================== 双设备总计 =====================
TOTAL_IMU = BNO_COUNT * 2                    # 12

# ===================== 显示参数 =====================
IMU_LABELS = [f"IMU{i}" for i in range(TOTAL_IMU)]
QUAT_LABELS = ["w", "x", "y", "z"]
QUAT_COLORS = [
    (0,   0,   0),
    (220, 50,  50),
    (34,  160, 60),
    (50,  120, 220),
]
SCROLL_WINDOW        = 500          # 滚动窗口显示的数据点数
PLOT_REFRESH_HZ      = 60           # GUI 刷新帧率（OpenGL 模式，60Hz 已流畅）
RING_BUFFER_CAPACITY = 8192         # 环形缓冲区容量

DEVICE_LABELS = ["右手 (设备2)", "左手 (设备1)"]


# ===================== 四元数 → 欧拉角 =====================
def quat_to_euler_deg(w, x, y, z):
    """返回 (roll, pitch, yaw)，单位度。ZYX 内旋顺序。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll  = np.arctan2(sinr_cosp, cosr_cosp)
    sinp  = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw   = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ===================== TCP 接收线程 =====================
class TcpReceiver:
    """TCP 接收线程：收包、解包，直接写入 ring buffer 和 CSV 队列。"""

    def __init__(self, ring, ring_lock, host, port, device_id=0):
        self.ring        = ring
        self.ring_lock   = ring_lock
        self.csv_writer  = None          # 由外部在采集开始时设置
        self.host        = host
        self.port        = port
        self.device_id   = device_id     # 0 或 1，用于 CSV 列偏移
        self.running     = False
        self.sock        = None
        self._thread     = None
        self.last_seq    = None
        self.lost_packets  = 0
        self.recv_packets  = 0
        self.last_timestamp = 0
        self.recording   = False

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        sock = self.sock
        self.sock = None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    def _connect(self):
        while self.running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            try:
                sock.connect((self.host, self.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
                sock.settimeout(1.0)
                self.sock      = sock
                self.last_seq  = None
                with self.ring_lock:
                    self.ring.clear()
                print(f"[TCP{self.device_id}] Connected to {self.host}:{self.port}")
                return True
            except OSError as exc:
                try:
                    sock.close()
                except Exception:
                    pass
                print(f"[TCP{self.device_id}] Connect failed: {exc}. Retry in {RECONNECT_INTERVAL}s")
                time.sleep(RECONNECT_INTERVAL)
        return False

    def _run(self):
        buffer = bytearray()
        while self.running:
            if self.sock is None and not self._connect():
                break
            sock = self.sock
            if sock is None:
                continue
            try:
                chunk = sock.recv(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.running:
                    break
                print(f"[TCP{self.device_id}] Receive error:", exc)
                chunk = b""

            if not chunk:
                if not self.running:
                    break
                print(f"[TCP{self.device_id}] Disconnected. Reconnecting...")
                try:
                    sock.close()
                except Exception:
                    pass
                if self.sock is sock:
                    self.sock = None
                buffer.clear()
                time.sleep(0.5)
                continue

            buffer.extend(chunk)
            while len(buffer) >= PACKET_SIZE:
                try:
                    self._process_frame(buffer)
                except Exception as exc:
                    print(f"[TCP{self.device_id}] Parse error:", exc)
                del buffer[:PACKET_SIZE]

    def _process_frame(self, buffer):
        seq, timestamp = struct.unpack_from(PACKET_HEADER_FORMAT, buffer, 0)
        floats = np.frombuffer(
            buffer, dtype=np.float32,
            count=SENSOR_FLOAT_COUNT, offset=PACKET_HEADER_SIZE,
        ).copy()
        quats  = floats[:QUAT_FLOAT_COUNT].reshape((BNO_COUNT, 4))
        accels = floats[QUAT_FLOAT_COUNT:QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT].reshape((BNO_COUNT, 3))
        gyros  = floats[QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT:].reshape((BNO_COUNT, 3))

        self.recv_packets  += 1
        self.last_timestamp = timestamp
        if self.last_seq is None:
            self.last_seq = seq
        else:
            delta = (seq - self.last_seq) & 0xFFFFFFFF
            if 1 < delta < 0x80000000:
                self.lost_packets += delta - 1
            self.last_seq = seq

        with self.ring_lock:
            self.ring.push(quats)

        if self.recording and self.csv_writer is not None:
            self.csv_writer.enqueue(self.device_id, seq, timestamp, quats, accels, gyros)


# ===================== CSV 写入线程（双设备合并单行 12 IMU）=====================
class CsvWriterThread(threading.Thread):
    """后台线程：缓冲两路设备数据，配对后一行写入全部 12 个 IMU。

    每行格式：label, seq_0, timestamp_us_0, imu0~5, seq_1, timestamp_us_1, imu6~11
    label=0 过渡段，label=词语名 词语段。
    只写入两路都有新帧的配对行，避免单路数据行。
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._queue = queue.Queue(maxsize=65536)
        self._file  = None
        self._writer = None
        self.running = True
        self.rows_written = 0
        self._buf = [None, None]   # [dev0_row, dev1_row]
        self.current_label = "0"    # 由 GUI 线程更新（线程安全：GIL）

    def open(self, filepath):
        self._file = open(filepath, "w", newline="", encoding="utf-8-sig")
        header = ["label", "seq_0", "timestamp_us_0"]
        for i in range(BNO_COUNT):
            header += [f"imu{i}_qw", f"imu{i}_qx", f"imu{i}_qy", f"imu{i}_qz"]
            header += [f"imu{i}_ax", f"imu{i}_ay", f"imu{i}_az"]
            header += [f"imu{i}_gx", f"imu{i}_gy", f"imu{i}_gz"]
        header += ["seq_1", "timestamp_us_1"]
        for i in range(BNO_COUNT):
            imu_idx = BNO_COUNT + i
            header += [f"imu{imu_idx}_qw", f"imu{imu_idx}_qx", f"imu{imu_idx}_qy", f"imu{imu_idx}_qz"]
            header += [f"imu{imu_idx}_ax", f"imu{imu_idx}_ay", f"imu{imu_idx}_az"]
            header += [f"imu{imu_idx}_gx", f"imu{imu_idx}_gy", f"imu{imu_idx}_gz"]
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self.rows_written = 0

    def close(self):
        self._queue.put(None)  # sentinel
        self.join(timeout=5.0)
        if self._file:
            self._file.close()
            self._file  = None
            self._writer = None

    def enqueue(self, device_id, seq, timestamp, quats, accels, gyros):
        """由接收线程调用，非阻塞。队列满时丢弃（极端情况）。"""
        row = [device_id, seq, timestamp, self.current_label]
        for i in range(BNO_COUNT):
            q = quats[i]; row += [q[0], q[1], q[2], q[3]]
            a = accels[i]; row += [a[0], a[1], a[2]]
            g = gyros[i];   row += [g[0], g[1], g[2]]
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            print("[CSV] Queue full, dropping frame")

    def run(self):
        while self.running:
            row = self._queue.get()
            if row is None:
                break
            if self._writer is None:
                continue
            dev_id = row[0]
            self._buf[dev_id] = row   # [device_id, seq, ts, label, imu0..5_data]
            # 两路都有新帧时合并写入
            if self._buf[0] is not None and self._buf[1] is not None:
                r0 = self._buf[0]
                r1 = self._buf[1]
                combined = [r1[3]]                  # label（取 dev1 的，因为 label 写入时两路同时刻）
                combined += [r0[1], r0[2]]           # seq_0, ts_0
                combined += r0[4:]                   # dev0 imu0-5（跳过 device_id, seq, ts, label 共4项）
                combined += [r1[1], r1[2]]           # seq_1, ts_1
                combined += r1[4:]                   # dev1 imu0-5
                try:
                    self._writer.writerow(combined)
                    self.rows_written += 1
                except Exception as exc:
                    print("[CSV] Write error:", exc)
                self._buf = [None, None]
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass


# ===================== 环形缓冲区 =====================
class RingBuffer:
    """固定长度环形缓冲，用于 GUI 显示最近 N 帧。"""

    def __init__(self, capacity, channels, fields):
        self.capacity = capacity
        self.channels = channels
        self.fields   = fields
        self.count    = 0
        self.index    = 0
        self.data = np.zeros((capacity, channels, fields), dtype=np.float32)

    def push(self, quats):
        """quats: (BNO_COUNT, 4)"""
        self.data[self.index] = quats
        self.index = (self.index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_last_n(self, n):
        """只返回最近 n 帧，避免复制整个缓冲区。"""
        n = min(n, self.count)
        if n <= 0:
            return np.empty((0, self.channels, self.fields), dtype=np.float32)
        if self.count < self.capacity:
            start = self.count - n
            return self.data[start:self.count].copy()
        end   = self.index
        start = (end - n) % self.capacity
        if start < end:
            return self.data[start:end].copy()
        result    = np.empty((n, self.channels, self.fields), dtype=np.float32)
        first_len = self.capacity - start
        result[:first_len] = self.data[start:]
        result[first_len:] = self.data[:end]
        return result

    def clear(self):
        self.count = 0
        self.index = 0


# ===================== 主窗口 =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Direct IMU 采集 —— 双设备 12 路")
        self.setMinimumSize(1400, 900)
        self.resize(1800, 1000)

        # ---- 双设备核心对象 ----
        self.rings       = []
        self.ring_locks  = []
        self.receivers   = []
        for dev_id, (ip, port) in enumerate([
            (DEVICE2_IP, DEVICE2_PORT),
            (DEVICE1_IP, DEVICE1_PORT),
        ]):
            ring  = RingBuffer(RING_BUFFER_CAPACITY, BNO_COUNT, 4)
            lock  = threading.Lock()
            recv  = TcpReceiver(ring, lock, host=ip, port=port, device_id=dev_id)
            self.rings.append(ring)
            self.ring_locks.append(lock)
            self.receivers.append(recv)

        self.csv_writer   = None
        self.recording    = False
        self._record_count = 0

        # 自动分段状态
        self._segment_names   = []      # 预设词语名称列表，如 ["你好","世界"]
        self._seg_phase       = "idle"  # idle / transition / word
        self._seg_phase_start = 0.0     # 当前阶段开始时间
        self._seg_word_index  = 0       # 下一个要播放的词索引
        self._seg_word_count  = 0       # 已完成的词语段数量

        # ---- 中央 Widget ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        # ---- 顶部工具栏 ----
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton(f"开始采集 (1/{MAX_RECORDINGS})")
        self.btn_stop  = QtWidgets.QPushButton("停止采集")
        self.btn_stop.setEnabled(False)
        # 分段状态标签（只读显示，不可点击）
        self.lbl_segment = QtWidgets.QLabel("等待采集...")
        self.lbl_segment.setStyleSheet(
            "QLabel { font-weight: bold; color: #bbb; padding: 3px 8px; "
            "border: 1px solid #ddd; border-radius: 3px; }"
        )
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)
        toolbar.addSpacing(18)
        self.btn_setup_seg = QtWidgets.QPushButton("设置段落")
        self.btn_setup_seg.setToolTip("输入这段话的词语数量和每个词的名称")
        toolbar.addWidget(self.btn_setup_seg)
        toolbar.addWidget(self.lbl_segment)
        toolbar.addSpacing(12)
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
        self.lbl_status = QtWidgets.QLabel("等待数据...")
        toolbar.addWidget(self.lbl_status)
        main_layout.addLayout(toolbar)

        # ---- pyqtgraph 配置 ----
        pg.setConfigOptions(antialias=True, useOpenGL=True)

        # ---- 左右两列布局 ----
        panels_layout = QtWidgets.QHBoxLayout()
        panels_layout.setSpacing(4)
        main_layout.addLayout(panels_layout, stretch=1)

        self.plots  = [[] for _ in range(2)]   # plots[dev_id][imu_idx]
        self.curves = [[] for _ in range(2)]   # curves[dev_id][imu_idx][quat_idx]

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
                plot = gl_widget.addPlot(title=f"IMU{global_imu_idx}  Quaternion")
                plot.showGrid(x=True, y=True, alpha=0.15)
                plot.setLabel("left",   "Value")
                plot.setLabel("bottom", "Samples")
                plot.setTitle(f"IMU{global_imu_idx}  Quaternion", color="#333", size="10pt")
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
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_setup_seg.clicked.connect(self._on_setup_segments)

        # ---- 定时刷新 GUI (60Hz，批量读取 ring buffer 重绘) ----
        self._last_status_time = 0.0
        self._x_axis = np.arange(SCROLL_WINDOW)
        self._plot_data = [
            np.full((SCROLL_WINDOW, BNO_COUNT, 4), np.nan, dtype=np.float32)
            for _ in range(2)
        ]
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self._refresh)
        self._last_frame_time = time.perf_counter()

    # ---------- 采集控制 ----------
    def _on_start(self):
        next_index = self._record_count + 1
        filepath   = SAVE_DIR / f"{next_index}.csv"
        writer = CsvWriterThread()
        writer.start()
        writer.open(filepath)
        self.csv_writer = writer
        for recv in self.receivers:
            recv.csv_writer = writer
            recv.recording  = True
        self.recording = True
        self._recording_filename = filepath.name

        # 分段复位，进入第一个过渡段
        self._seg_word_index  = 0
        self._seg_word_count  = 0
        self.csv_writer.current_label = "0"
        if self._segment_names:
            self._seg_phase       = "transition"
            self._seg_phase_start = time.monotonic()
            self._update_seg_label()
            self.btn_setup_seg.setEnabled(False)
            print(f"[SEG] Auto-seg started: {len(self._segment_names)} words -> {self._segment_names}")
            print(f"[SEG] Phase: transition ({TRANSITION_DURATION}s)")
        else:
            self._seg_phase = "idle"
            self.lbl_segment.setText("(未设置段落)")
            self.lbl_segment.setStyleSheet(
                "QLabel { font-weight: bold; color: #999; padding: 3px 8px; "
                "border: 1px solid #ddd; border-radius: 3px; }"
            )
            self.btn_setup_seg.setEnabled(False)

        # 开始/停止
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        for recv in self.receivers:
            recv.start()
        self._timer.start(int(1000 / PLOT_REFRESH_HZ))
        print(f"[ACQ] Recording to {filepath}  ({next_index}/{MAX_RECORDINGS})")

    def _on_stop(self):
        self.recording = False
        self._seg_phase = "idle"
        for recv in self.receivers:
            recv.recording  = False
            recv.csv_writer = None

        # 分段标签复位
        self.lbl_segment.setText("等待采集...")
        self.lbl_segment.setStyleSheet(
            "QLabel { font-weight: bold; color: #bbb; padding: 3px 8px; "
            "border: 1px solid #ddd; border-radius: 3px; }"
        )
        self.btn_setup_seg.setEnabled(True)

        if self.csv_writer is not None:
            self.csv_writer.close()
            rows = self.csv_writer.rows_written
            self.csv_writer = None
        else:
            rows = 0
        self._timer.stop()
        for recv in self.receivers:
            recv.stop()

        self._record_count += 1
        finished = self._record_count
        seg_count = self._seg_word_count
        print(f"[ACQ] Stopped. {rows} rows saved, {seg_count} segments. ({finished}/{MAX_RECORDINGS})")

        if finished >= MAX_RECORDINGS:
            self.lbl_status.setText(
                f"已完成全部 {MAX_RECORDINGS} 次采集，程序即将退出…"
            )
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(False)
            QtCore.QTimer.singleShot(1500, self.close)
        else:
            remaining = MAX_RECORDINGS - finished
            self.btn_start.setText(
                f"开始采集 ({finished + 1}/{MAX_RECORDINGS})"
            )
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText(
                f"第 {finished} 次已保存 {rows} 帧，{seg_count} 段，还剩 {remaining} 次"
            )

    # ---------- 设置段落名称（采集前调用）----------
    def _on_setup_segments(self):
        """弹出对话框，输入词语数量和每个词的名称。"""
        if self.recording:
            return
        # Step 1: 输入词语数量
        count, ok = QtWidgets.QInputDialog.getInt(
            self, "设置段落", "这段话包含几个词语？",
            value=max(1, len(self._segment_names)), minValue=1, maxValue=30,
        )
        if not ok:
            return
        # Step 2: 逐个输入词语名称
        names = []
        for i in range(count):
            default = self._segment_names[i] if i < len(self._segment_names) else ""
            name, ok2 = QtWidgets.QInputDialog.getText(
                self, f"词语 {i + 1}/{count}",
                f"请输入第 {i + 1} 个词语的名称：",
                text=default,
            )
            if not ok2:
                return
            names.append(name.strip() if name.strip() else f"词{i + 1}")
        self._segment_names = names
        self._seg_word_index = 0
        print(f"[SEG] Setup: {count} words -> {names}")

    # ---------- 自动分段 tick（在 _refresh 中每帧调用）----------
    def _seg_tick(self):
        """根据经过时间自动切换 transition / word 阶段并更新 label。"""
        if not self.recording or self._seg_phase == "idle":
            return
        if not self._segment_names:
            return
        now     = time.monotonic()
        elapsed = now - self._seg_phase_start

        if self._seg_phase == "transition":
            if elapsed >= TRANSITION_DURATION:
                # 过渡段结束 → 进入下一个词语段
                if self._seg_word_index < len(self._segment_names):
                    word = self._segment_names[self._seg_word_index]
                    self._seg_word_index  += 1
                    self._seg_word_count  += 1
                    self._seg_phase        = "word"
                    self._seg_phase_start  = now
                    self.csv_writer.current_label = word
                    self._update_seg_label()
                    print(f"[SEG] → word '{word}'  ({self._seg_word_index}/{len(self._segment_names)})")
                else:
                    # 所有词已采集完，保持过渡段直到手动停止
                    self._seg_phase = "done"
                    self.csv_writer.current_label = "0"
                    self.lbl_segment.setText("✓ 全部词语采集完毕")
                    self.lbl_segment.setStyleSheet(
                        "QLabel { font-weight: bold; color: #1a6bb5; padding: 3px 8px; "
                        "border: 1px solid #1a6bb5; border-radius: 3px; }"
                    )
                    print("[SEG] All words done, staying in transition until stop.")

        elif self._seg_phase == "word":
            if elapsed >= WORD_DURATION:
                # 词语段结束 → 回到过渡段
                self._seg_phase       = "transition"
                self._seg_phase_start = now
                self.csv_writer.current_label = "0"
                self._update_seg_label()
                print(f"[SEG] → transition ({TRANSITION_DURATION}s)")

    def _update_seg_label(self):
        """更新顶部分段状态标签（阶段名称 + 剩余倒计时）。"""
        if self._seg_phase == "transition":
            elapsed  = time.monotonic() - self._seg_phase_start
            remain   = max(0.0, TRANSITION_DURATION - elapsed)
            next_idx = self._seg_word_index
            if next_idx < len(self._segment_names):
                next_word = self._segment_names[next_idx]
                text = f"过渡段  {remain:.1f}s → [{next_word}]"
            else:
                text = f"过渡段  {remain:.1f}s → (结束)"
            self.lbl_segment.setText(text)
            self.lbl_segment.setStyleSheet(
                "QLabel { font-weight: bold; color: #d4380d; padding: 3px 8px; "
                "border: 1px solid #d4380d; border-radius: 3px; }"
            )
        elif self._seg_phase == "word":
            elapsed = time.monotonic() - self._seg_phase_start
            remain  = max(0.0, WORD_DURATION - elapsed)
            word    = self._segment_names[self._seg_word_index - 1]
            text    = f"[{word}]  {remain:.1f}s  ({self._seg_word_index}/{len(self._segment_names)})"
            self.lbl_segment.setText(text)
            self.lbl_segment.setStyleSheet(
                "QLabel { font-weight: bold; color: #237804; padding: 3px 8px; "
                "border: 1px solid #237804; border-radius: 3px; background: #f6ffed; }"
            )

    # ---------- GUI 批量刷新 ----------
    def _refresh(self):
        # 自动分段 tick
        self._seg_tick()
        # 每帧更新倒计时显示
        if self._seg_phase in ("transition", "word"):
            self._update_seg_label()

        for dev_id in range(2):
            with self.ring_locks[dev_id]:
                data = self.rings[dev_id].get_last_n(SCROLL_WINDOW)
            n = data.shape[0]
            if n < 1:
                continue
            pd = self._plot_data[dev_id]
            if n < SCROLL_WINDOW:
                pd[:] = np.nan
            pd[SCROLL_WINDOW - n:] = data
            x_axis = self._x_axis
            for i in range(BNO_COUNT):
                curve_row = self.curves[dev_id][i]
                for j in range(4):
                    curve_row[j].setData(x_axis, pd[:, i, j])

        # FPS 计数 + 状态栏
        t_now = time.perf_counter()
        dt = t_now - self._last_frame_time
        self._last_frame_time = t_now

        now = time.monotonic()
        if now - self._last_status_time >= 0.5:
            self._last_status_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            r0  = self.receivers[0]
            r1  = self.receivers[1]
            if self.recording:
                lbl = self.csv_writer.current_label if self.csv_writer else "0"
                seg_text = f" [{lbl}]" if lbl != "0" else " [过渡段]"
                rec_text = f"采集中 -> {self._recording_filename}{seg_text}"
            else:
                rec_text = "未采集"
            self.lbl_status.setText(
                f"FPS={fps:.0f} | "
                f"Dev1 接收={r0.recv_packets} 丢包={r0.lost_packets} | "
                f"Dev2 接收={r1.recv_packets} 丢包={r1.lost_packets} | "
                f"{rec_text}"
            )

    # ---------- 关闭 ----------
    def closeEvent(self, event):
        self._timer.stop()
        for recv in self.receivers:
            recv.recording = False
            recv.stop()
        if self.csv_writer is not None:
            self.csv_writer.close()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_Space:
            if self.btn_start.isEnabled():
                self._on_start()
            elif self.btn_stop.isEnabled():
                self._on_stop()
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
