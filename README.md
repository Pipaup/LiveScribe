# LiveScribe — 实时语音转文字系统

中文 | [English](README_en.md)

一个 Python 实现的实时语音转文字系统，通过 Windows WASAPI Loopback 捕获系统音频（播放器、浏览器、会议等任何系统声音），经过 VAD 语音检测断句后，由可切换的 ASR 引擎实时输出文字。

> **一句话：你在电脑上播放的任何声音都能实时转成文字，支持三种识别引擎切换。**

---

## ✨ 特性

- **系统音频捕获**：通过 WASAPI Loopback 捕获系统扬声器输出，无需麦克风，任何电脑播放的声音（视频、播客、在线会议）都能转写
- **引擎可切换**：在 `config.yaml` 中一行切换 ASR 引擎，无需改代码
- **三种引擎支持**：
  | 引擎 | 特点 | 显存需求 | 推荐场景 |
  |------|------|---------|---------|
  | **faster-whisper** | CTranslate2 后端，CPU 友好 | ~500MB | 通用推荐，中英文均可 |
  | **Qwen3-ASR 0.6B** | 中文顶级准确率，52 种语言 | ~1.8GB | 中文/多语种最佳 |
  | **Qwen3-ASR 1.7B** | SOTA 开放源 ASR 模型 | ~4GB | 追求最高准确率 |
- **VAD 语音检测**：Silero VAD 自动检测语音边界，支持阈值、静音时长等参数调节
- **双线程架构**：音频采集与 ASR 推理分离，不丢帧
- **多种输出**：同时输出到控制台（实时刷新）和文件（定时保存）
- **模型本地化**：支持本地模型路径，完全离线可用

---

## 📂 架构

```
LiveScribe/
├── livescribe/                     # 主 Python 包
│   ├── asr/                        # ASR 引擎层（策略模式）
│   │   ├── base.py                 #   引擎抽象接口
│   │   ├── factory.py              #   工厂函数（按配置创建引擎）
│   │   ├── faster_whisper_engine.py
│   │   ├── qwen3_transformers_engine.py
│   │   └── qwen3_vllm_engine.py
│   ├── audio/                      # 音频捕获层
│   │   ├── capture.py              #   WASAPI Loopback 捕获
│   │   ├── device.py               #   设备枚举与选择
│   │   └── resampler.py            #   重采样工具
│   ├── vad/                        # 语音活动检测层
│   │   ├── detector.py             #   Silero VAD 封装
│   │   └── download.py             #   VAD 模型下载工具
│   ├── output/                     # 输出层
│   │   ├── console.py              #   控制台实时打印
│   │   ├── file_writer.py          #   文件写入（定时 flush）
│   │   └── output_manager.py       #   多后端管理
│   ├── config/                     # 配置模块
│   │   ├── schema.py               #   配置数据类 + 枚举
│   │   └── loader.py               #   YAML 加载 + 校验
│   ├── pipeline.py                 # 主流水线（双线程）
│   ├── exceptions.py               # 自定义异常
│   ├── logging_utils.py            # 日志工具
│   └── __main__.py                 # `python -m livescribe` 入口
├── config.example.yaml             # 配置模板
├── main.py                         # 程序入口
├── requirements.txt                # Python 依赖
├── run.bat                         # Windows 一键启动脚本
├── run-download-vad.bat            # 下载 VAD 模型
├── run-list-devices.bat            # 列出音频设备
└── setup-env.bat                   # 环境配置脚本
```

---

## 🚀 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- NVIDIA GPU（可选但推荐，RTX 3060 6GB 或以上）
- CUDA 12.4+

### 1. 安装依赖

```bash
# 使用 conda（推荐）
conda create -n livescribe python=3.11 -y
conda activate livescribe

# 安装 PyTorch（GPU 版）
pip install torch>=2.6.0 torchaudio>=2.6.0

# 安装项目依赖
pip install -r requirements.txt
pip install -e .
```

或者直接双击运行 `setup-env.bat`（Windows 批处理脚本）。

### 2. 准备模型

```bash
# 下载 Qwen3-ASR 0.6B（可选）
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./models/Qwen3-ASR-0.6B

# 下载 Silero VAD 模型（必须）
python main.py --download-vad
```

### 3. 配置

```bash
# 复制配置模板
copy config.example.yaml config.yaml

# 编辑 config.yaml，选择引擎：
#   asr.engine: "faster_whisper"  或  "qwen3_transformers"
```

### 4. 运行

```bash
# 启动转写
python main.py

# 或双击 run.bat

# 列出可用音频设备
python main.py --list-devices
```

### 5. 使用

1. 确保系统有声输出（播放视频/音频）
2. 按 `Ctrl+C` 停止

---

## ⚙️ 配置说明

所有配置在 `config.yaml` 中（完整模板见 `config.example.yaml`）：

```yaml
mode: "vad_segmented"                     # 运行模式

asr:
  engine: "faster_whisper"                # 切换引擎的唯一切换点
  
  faster_whisper:
    model_size: "large-v3-turbo"          # 推荐 large-v3-turbo
    device: "cuda"
    
  qwen3:
    model_name: "./models/Qwen3-ASR-0.6B"  # 本地路径或 HF ID
    device: "cuda"
    dtype: "bfloat16"

audio:
  chunk_duration_ms: 32                   # 帧长
  loopback_device: null                   # null=自动选默认设备

vad:
  enabled: true
  threshold: 0.5                          # 语音概率阈值
  min_silence_duration_ms: 300            # 静音多久判断断句

output:
  console:
    enabled: true                         # 控制台输出
  file:
    enabled: true                         # 文件输出
```

---

## 🔧 引擎切换

修改 `config.yaml` 中一行即可切换：

```yaml
asr:
  engine: "faster_whisper"    # ← 改这里
```

| 值 | 引擎 | 适用场景 |
|----|------|---------|
| `faster_whisper` | faster-whisper (CTranslate2) | 通用推荐，支持中文 |
| `qwen3_transformers` | Qwen3-ASR Transformers | 中文/多语种最佳 |
| `qwen3_vllm` | Qwen3-ASR vLLM | Linux 流式推理（Windows 暂不可用） |

---

## 🔌 扩展新引擎

只需三步，不需要修改 Pipeline、Audio、VAD、Output：

1. 在 `livescribe/asr/` 下新建 `xxx_engine.py`，继承 `BaseASREngine`
2. 在 `livescribe/config/schema.py` 中添加枚举值
3. 在 `livescribe/asr/factory.py` 中添加 `elif` 分支

---

## 📝 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check .
ruff format --check .

# 类型检查
mypy livescribe/
```

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — 强大的多语言 ASR 模型
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速的 Whisper
- [Silero VAD](https://github.com/snakers4/silero-vad) — 轻量级语音活动检测
- [SoundCard](https://github.com/bastibe/SoundCard) — Python WASAPI Loopback 支持
