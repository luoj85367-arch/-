# 手语 IMU 最小推理包

这个目录是可迁移推理版本，只包含运行实时识别和 CSV 离线测试需要的代码与模型文件。

当前模型：

- 35 类：34 个手势 + `静止`
- 模型文件：`model/model.pt`
- 验证准确率：94.65%
- 测试准确率：94.71%
- 当前对话评估：178/178 句完全匹配

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r portable_inference/requirements.txt
```

CPU 可以直接运行。需要 GPU 时，按本机 CUDA 版本安装对应的 PyTorch。

## 运行

在项目根目录运行，不需要 `cd portable_inference`，也不需要手动设置 `PYTHONPATH`。

GUI 实时推理：

```bash
python -m portable_inference
```

终端实时推理：

```bash
python -m portable_inference --no-gui
```

CSV 离线推理：

```bash
python -m portable_inference --csv dataset_processed/dialogues/单手对话2：非常好，你下午吃什么/1.csv --device cpu
```

如果你选择进入 `portable_inference` 安装为包，也可以用：

```bash
cd portable_inference
pip install -e .
imu-inference --csv path/to/file.csv
```

## VSCode

已提供根目录 `.vscode/launch.json`：

- `Portable Inference GUI`
- `Portable Inference CSV`

在 VSCode 里选择项目根目录的 Python 解释器后，直接运行这两个配置即可。`.vscode/settings.json` 也把 `portable_inference/src` 加入了分析路径，编辑器能识别 `imu_inference` 包。

## 连接参数

默认连接两只手套的 TCP 数据流：

```text
右手 192.168.31.152:8911
左手 192.168.31.152:8913
```

覆盖默认地址：

```bash
python -m portable_inference \
  --right-host 192.168.1.100 --right-port 8911 \
  --left-host 192.168.1.101 --left-port 8913
```

## 推理规则

默认参数按最新模型验证过的配置设置：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--window-len` | 128 | 模型滑动窗口帧数 |
| `--step` | 8 | 滑动推理步长 |
| `--tail-frames` | 640 | 实时缓冲尾段长度 |
| `--min-confidence` | 0.70 | 帧级最低置信度 |
| `--smooth-radius` | 2 | 概率时间平滑半径 |
| `--min-token-frames` | 16 | 最短 token 帧数，过滤静止段短误触发 |
| `--merge-gap` | 12 | 同类 token 合并间隔 |
| `--commit-lag-frames` | 12 | 实时提交前等待尾部稳定 |
| `--repeat-gap-frames` | 24 | 防止滑窗重复提交同一 token |

现在的实时约定是每个动作结束后约 1 秒静止。模型会使用动作过程和结束后的短静止上下文，长静止段本身不会输出 token；纯静止文件已验证无误触发。

## 目录结构

```text
portable_inference/
├── __main__.py                 # 根目录 python -m portable_inference 入口
├── __init__.py
├── model/
│   ├── model.pt
│   ├── normalization.json
│   └── dataset_meta.json
├── src/imu_inference/
│   ├── __main__.py
│   ├── model.py
│   ├── predict.py
│   ├── receiver.py
│   └── realtime.py
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md
```

## 采集通道

模型输入为 12 个 IMU，每个 IMU 10 维：

```text
qw, qx, qy, qz, ax, ay, az, gx, gy, gz
```

右手映射到 `imu0` 到 `imu5`，左手映射到 `imu6` 到 `imu11`，共 120 维。CSV 离线测试要求包含这些标准列名。
