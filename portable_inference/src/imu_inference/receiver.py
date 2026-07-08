"""
IMU 数据接收器：通过 TCP 连接两只手套的 IMU 传感器，实时接收数据。

协议：
- TCP 连接，每个数据包 248 字节
- 头部 12 字节：seq(uint32) + timestamp(uint64)
- 负载 240 字节：60 个 float32（6 IMU × 4四元数 + 6 IMU × 3加速度 + 6 IMU × 3陀螺仪）

通道映射（来自 data.md）：
- 右手 imu0-5：0=手背, 1=尾指, 2=无名指, 3=中指, 4=食指, 5=拇指
- 左手 imu6-11：11=手背, 10=尾指, 9=无名指, 8=中指, 7=食指, 6=拇指（序号反转）
"""

import socket
import struct
import threading
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

HAND_BNO_COUNT = 6
BNO_COUNT = 12
FEATURES_PER_IMU = 10
BUFFER_SIZE = 65536
RECONNECT_INTERVAL = 1.0

QUAT_FLOAT_COUNT = HAND_BNO_COUNT * 4    # 24
VECTOR_FLOAT_COUNT = HAND_BNO_COUNT * 3  # 18
SENSOR_FLOAT_COUNT = QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT * 2  # 60

PACKET_HEADER_FORMAT = "<IQ"
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)  # 12
PACKET_FLOAT_FORMAT = f"<{SENSOR_FLOAT_COUNT}f"
PACKET_SIZE = PACKET_HEADER_SIZE + struct.calcsize(PACKET_FLOAT_FORMAT)  # 252

# 默认连接参数（可在命令行覆盖）
DEFAULT_RIGHT_HOST = "192.168.31.152"
DEFAULT_RIGHT_PORT = 8911
DEFAULT_LEFT_HOST = "192.168.31.152"
DEFAULT_LEFT_PORT = 8913


def packet_floats_to_feature_row(floats):
    """
    将原始 packet float 数组重组为特征行。

    原始顺序: [quat0..5(24个), accel0..5(18个), gyro0..5(18个)]
    目标顺序: [imu0_qw,qx,qy,qz,ax,ay,az,gx,gy,gz, imu1_..., ...] (60个)

    Returns:
        np.ndarray shape (60,)
    """
    quats = floats[:QUAT_FLOAT_COUNT].reshape((HAND_BNO_COUNT, 4))
    accels = floats[QUAT_FLOAT_COUNT:QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT].reshape((HAND_BNO_COUNT, 3))
    gyros = floats[QUAT_FLOAT_COUNT + VECTOR_FLOAT_COUNT:].reshape((HAND_BNO_COUNT, 3))

    row = np.empty(SENSOR_FLOAT_COUNT, dtype=np.float32)
    idx = 0
    for i in range(HAND_BNO_COUNT):
        row[idx:idx + 4] = quats[i];  idx += 4
        row[idx:idx + 3] = accels[i]; idx += 3
        row[idx:idx + 3] = gyros[i];  idx += 3
    return row


def combine_hand_rows(right_row, left_row):
    """
    合并左右手数据为 120 维特征向量。

    右手 local_idx 0-5 → global 0-5
    左手 local_idx 0-5 → global 11-6（反转）
    """
    combined = np.zeros((BNO_COUNT * FEATURES_PER_IMU,), dtype=np.float32)
    for local_idx in range(HAND_BNO_COUNT):
        src = slice(local_idx * FEATURES_PER_IMU, (local_idx + 1) * FEATURES_PER_IMU)
        right_global = local_idx
        left_global = 11 - local_idx
        combined[right_global * FEATURES_PER_IMU:(right_global + 1) * FEATURES_PER_IMU] = right_row[src]
        combined[left_global * FEATURES_PER_IMU:(left_global + 1) * FEATURES_PER_IMU] = left_row[src]
    return combined


# ---------------------------------------------------------------------------
# 单手接收器
# ---------------------------------------------------------------------------

class HandReceiver:
    """通过 TCP 接收单只手的 IMU 数据。"""

    def __init__(self, host, port, capacity=2048):
        self.host = host
        self.port = port
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.running = False
        self.sock = None
        self.thread = None
        self.last_seq = None
        self.recv_packets = 0
        self.lost_packets = 0
        self.connected = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        sock = self.sock
        self.sock = None
        if sock:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            try: sock.close()
            except OSError: pass
        self.connected = False

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def drain_since(self, last_count):
        """获取上次调用以来的所有新帧。"""
        with self.lock:
            total = self.recv_packets
            available = list(self.buffer)
        delta = max(0, total - last_count)
        if delta <= 0:
            return [], total
        rows = available[-min(delta, len(available)):]
        return [r.copy() for r in rows], total

    def _connect(self):
        while self.running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            try:
                sock.connect((self.host, self.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
                sock.settimeout(1.0)
                self.sock = sock
                self.connected = True
                self.last_seq = None
                self.clear()
                print(f"[IMU] 已连接 {self.host}:{self.port}")
                return True
            except OSError as exc:
                self.connected = False
                try: sock.close()
                except OSError: pass
                print(f"[IMU] 连接失败 {self.host}:{self.port}: {exc}, {RECONNECT_INTERVAL}s 后重试")
                time.sleep(RECONNECT_INTERVAL)
        return False

    def _run(self):
        raw = bytearray()
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
            except OSError:
                if not self.running:
                    break
                chunk = b""
            if not chunk:
                if not self.running:
                    break
                self.connected = False
                try: sock.close()
                except OSError: pass
                if self.sock is sock:
                    self.sock = None
                raw.clear()
                time.sleep(0.5)
                continue
            raw.extend(chunk)
            while len(raw) >= PACKET_SIZE:
                self._process_packet(raw)
                del raw[:PACKET_SIZE]

    def _process_packet(self, raw):
        seq, timestamp = struct.unpack_from(PACKET_HEADER_FORMAT, raw, 0)
        floats = np.frombuffer(raw, dtype=np.float32,
                               count=SENSOR_FLOAT_COUNT,
                               offset=PACKET_HEADER_SIZE).copy()
        row = packet_floats_to_feature_row(floats)
        self.recv_packets += 1
        if self.last_seq is not None:
            delta = (seq - self.last_seq) & 0xFFFFFFFF
            if 1 < delta < 0x80000000:
                self.lost_packets += delta - 1
        self.last_seq = seq
        with self.lock:
            self.buffer.append(row)


# ---------------------------------------------------------------------------
# 双手接收器
# ---------------------------------------------------------------------------

class DualHandReceiver:
    """同时接收左右两只手的 IMU 数据，配对合并为 120 维帧。"""

    def __init__(self, right_host, right_port, left_host, left_port, capacity=2048):
        self.right = HandReceiver(right_host, right_port, capacity=max(capacity, 256))
        self.left = HandReceiver(left_host, left_port, capacity=max(capacity, 256))
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.recv_packets = 0
        self.connected = False
        self._right_count = 0
        self._left_count = 0
        self._right_queue = deque()
        self._left_queue = deque()

    def start(self):
        if self.running:
            return
        self.running = True
        self.right.start()
        self.left.start()
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.right.stop()
        self.left.stop()
        self.connected = False

    def clear(self):
        self._right_queue.clear()
        self._left_queue.clear()
        self.right.clear()
        self.left.clear()
        with self.lock:
            self.buffer.clear()

    def drain_since(self, last_count):
        """获取上次调用以来的所有配对帧。"""
        with self.lock:
            total = self.recv_packets
            available = list(self.buffer)
        delta = max(0, total - last_count)
        if delta <= 0:
            return [], total
        rows = available[-min(delta, len(available)):]
        return [r.copy() for r in rows], total

    def _sync_loop(self):
        while self.running:
            right_rows, self._right_count = self.right.drain_since(self._right_count)
            left_rows, self._left_count = self.left.drain_since(self._left_count)
            self._right_queue.extend(right_rows)
            self._left_queue.extend(left_rows)

            paired = min(len(self._right_queue), len(self._left_queue))
            if paired:
                for _ in range(paired):
                    combined = combine_hand_rows(
                        self._right_queue.popleft(),
                        self._left_queue.popleft()
                    )
                    with self.lock:
                        self.buffer.append(combined)
                        self.recv_packets += 1

            self.connected = self.right.connected and self.left.connected
            time.sleep(0.005)

    @property
    def status(self):
        return (
            f"右手={self.right.host}:{self.right.port} "
            f"接收={self.right.recv_packets} 丢包={self.right.lost_packets} | "
            f"左手={self.left.host}:{self.left.port} "
            f"接收={self.left.recv_packets} 丢包={self.left.lost_packets} | "
            f"配对={self.recv_packets}"
        )
