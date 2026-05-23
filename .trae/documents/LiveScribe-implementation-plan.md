# LiveScribe 项目实现计划书

## 一、项目概述

**目标**：构建一个 Python 实时语音转文字系统，捕获 Windows 系统音频（WASAPI loopback），通过可切换的 ASR 引擎实时输出文字。

**核心原则**：
- **模型可切换**：通过配置文件一行切换引擎，无需改代码
- **高度可扩展**：新增 ASR 引擎只需实现一个抽象接口即可接入
- **分层解耦**：音频采集、VAD、ASR、输出 四层独立，互不干扰

---

## 二、目标硬件环境

| 项目 | 实际值 |
|------|--------|
| GPU | NVIDIA RTX 3060 Laptop 6GB |
| 空闲显存 | ~2.5GB（空闲）|
| Python | 3.12.7 (conda) |
| 操作系统 | Windows |
| 默认扬声器 | Senary Audio（板载声卡）|

---

## 三、ASR 引擎矩阵（可切换）

系统通过 `config.yaml` 中的 `asr.engine` 字段切换引擎，运行时无需改代码。

### 3.1 引擎一览

| 引擎标识符 | 后端 | 特点 | 显存需求 | 安装方式 |
|-----------|------|------|---------|---------|
| `qwen3_transformers` | Qwen3ASRModel.from_pretrained() | 轻量，分段转写，稳定 | 0.6B: ~3GB / 1.7B: ~6GB | `pip install qwen-asr` |
| `qwen3_vllm` | Qwen3ASRModel.LLM() | 流式推理，高吞吐 | 需额外 1-2GB | `pip install qwen-asr[vllm]` |
| `faster_whisper` | faster-whisper CTranslate2 | CPU 友好，140MB 模型 | ~500MB | `pip install faster-whisper` |
| `whisper`（预留）| openai-whisper | 生态成熟 | ~1-5GB | `pip install openai-whisper` |
| `vosk`（预留）| vosk | 超轻量流式 | ~50MB | `pip install vosk` |

### 3.2 引擎特性对比

| 特性 | qwen3_transformers | qwen3_vllm | faster_whisper |
|------|-------------------|------------|---------------|
| 流式支持 | ❌（分段+VAD） | ✅（原生流式） | ❌（分段+VAD） |
| 时间戳 | ✅（需 ForcedAligner） | ✅（需 ForcedAligner） | ✅ |
| 中文准确率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 模型大小 | 0.6B: 1.8GB / 1.7B: 4GB | 同左 | base: 140MB |
| 离线可用 | ✅ | ✅ | ✅ |
| 需要 GPU | 推荐 | 必须 | 可选 |

---

## 四、项目架构设计

### 4.1 目录结构

```
LiveScribe/
├── livescribe/                     # 主 Python 包
│   ├── __init__.py
│   │
│   ├── config/                     # 配置模块
│   │   ├── __init__.py
│   │   ├── schema.py               # 配置数据类 + 引擎枚举
│   │   └── loader.py               # YAML 加载 + 校验
│   │
│   ├── audio/                      # 音频捕获层
│   │   ├── __init__.py
│   │   ├── capture.py              # AudioCapture 类（WASAPI loopback）
│   │   ├── device.py               # 设备枚举与选择
│   │   └── resampler.py            # 重采样工具（48k/44.1k → 16k）
│   │
│   ├── vad/                        # 语音活动检测层
│   │   ├── __init__.py
│   │   └── detector.py             # Silero VAD 封装（VADIterator + 状态机）
│   │
│   ├── asr/                        # ASR 引擎层（策略模式 + 工厂模式）
│   │   ├── __init__.py
│   │   ├── base.py                 # 抽象基类 BaseASREngine（含流式接口）
│   │   ├── factory.py              # 工厂函数 create_engine(config)
│   │   ├── qwen3_transformers_engine.py   # Qwen3ASR Transformers 后端
│   │   ├── qwen3_vllm_engine.py           # Qwen3ASR vLLM 后端
│   │   └── faster_whisper_engine.py       # faster-whisper 引擎
│   │
│   ├── output/                     # 输出层
│   │   ├── __init__.py
│   │   ├── base_output.py          # 抽象基类 BaseOutput
│   │   ├── console.py              # 控制台实时打印（流式支持同行刷新）
│   │   └── file_writer.py          # 写入 txt 文件
│   │
│   ├── logging_utils.py            # 日志工具（统一 logger 配置）
│   └── pipeline.py                 # 主流水线（多线程，串联所有组件）
│
├── config.yaml                     # 用户配置文件
├── main.py                         # 程序入口
├── requirements.txt                # Python 依赖
└── .gitignore                      # 已有
```

### 4.2 分层架构图

```
┌──────────────────────────────────────────────────────────┐
│                      main.py                              │
│                  读取 config.yaml                          │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│                   Pipeline (pipeline.py)                  │
│  协调各层，管理运行生命周期                                  │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌──────────┐ ┌──────────┐
│ Audio │ │  VAD  │ │   ASR    │ │  Output  │
│Capture│ │Detect │ │ Engine   │ │  Layer   │
│       │ │(可选) │ │(可切换)  │ │          │
└───────┘ └───────┘ └──────────┘ └──────────┘
   │          │          │           │
   ▼          ▼          ▼           ▼
 WASAPI    Silero   策略模式     Console
 loopback  VAD     ┌────┴────┐  FileWriter
   │               │ factory │
 PCM 流            └────┬────┘
                   ┌────┼────┐
                   ▼    ▼    ▼
               qwen3 qwen3 faster
               trans vllm  whisper
```

### 4.3 核心设计模式

#### 4.3.1 策略模式：ASR 引擎

所有 ASR 引擎实现同一个抽象接口，Pipeline 只依赖接口，不依赖具体实现：

```python
# asr/base.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseASREngine(ABC):
    """所有 ASR 引擎的统一接口

    引擎分为两类，通过 supports_streaming 属性区分：

    分段引擎（supports_streaming=False）：
        - 只实现 transcribe()
        - Pipeline 走 VAD 断句 → 累积音频 → 整段转写 的路径

    流式引擎（supports_streaming=True）：
        - 额外实现 start_stream / feed_chunk / end_stream
        - Pipeline 走逐帧喂入 → 增量输出 的实时路径
        - transcribe() 仍然实现（用于 fallback 或测试）
    """

    # ============ 所有引擎必须实现 ============

    @abstractmethod
    def initialize(self) -> None:
        """加载模型、分配显存"""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """音频 → 文字（分段模式核心方法）
        - audio: float32 数组, shape=(num_samples,), 值域 [-1.0, 1.0]
        - sample_rate: 采样率, 默认 16000
        - 返回：识别出的文字字符串
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """释放显存、清理资源"""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """是否支持原生流式推理"""
        ...

    @property
    @abstractmethod
    def supports_timestamps(self) -> bool:
        """是否支持返回时间戳"""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """引擎名称（用于日志）"""
        ...

    # ============ 流式引擎额外实现（supports_streaming=True 时） ============

    def start_stream(self) -> None:
        """初始化流式会话。流式引擎必须 override。"""

    def feed_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """喂入一帧音频，返回增量文字（可能为空）。
        - chunk: float32 数组, shape=(512,), 值域 [-1.0, 1.0]
        - 返回：本轮新增的识别文字，无新结果时返回 None
        流式引擎必须 override。
        """

    def end_stream(self) -> str:
        """结束流式会话，冲刷缓冲区，返回最终结果。
        流式引擎必须 override。
        """

    # ============ 可选实现 ============

    def estimate_vram_usage(self) -> Optional[int]:
        """估算当前配置下的显存使用量（MB）。可选 override。"""
        return None
```

#### 4.3.2 工厂模式：按配置创建引擎

```python
# asr/factory.py
from livescribe.config.schema import ASRConfig
from livescribe.asr.base import BaseASREngine

def create_asr_engine(config: ASRConfig) -> BaseASREngine:
    """根据配置创建 ASR 引擎实例
    新增引擎只需在此函数中添加一个 elif 分支
    """
    engine_type = config.engine  # "qwen3_transformers" | "qwen3_vllm" | "faster_whisper"

    if engine_type == "qwen3_transformers":
        from livescribe.asr.qwen3_transformers_engine import Qwen3TransformersEngine
        return Qwen3TransformersEngine(config)

    elif engine_type == "qwen3_vllm":
        from livescribe.asr.qwen3_vllm_engine import Qwen3VllmEngine
        return Qwen3VllmEngine(config)

    elif engine_type == "faster_whisper":
        from livescribe.asr.faster_whisper_engine import FasterWhisperEngine
        return FasterWhisperEngine(config)

    else:
        raise ValueError(f"Unknown ASR engine type: {engine_type}")
```

#### 4.3.3 新增引擎的步骤（保证可扩展性）

1. 在 `livescribe/asr/` 下创建 `xxx_engine.py`
2. 继承 `BaseASREngine`，实现全部抽象方法
3. 在 `factory.py` 中添加 `elif` 分支
4. 在 `config/schema.py` 的枚举中添加引擎标识符
5. 完成。**不需要修改 Pipeline、Audio、VAD、Output 任何其他代码。**

---

## 五、配置文件设计

### 5.1 config.yaml（完整版）

```yaml
# ====================================================
#  LiveScribe 配置文件
#  修改 asr.engine 即可切换 ASR 引擎
# ====================================================

# -------- ASR 引擎配置（核心切换点）--------
asr:
  # 引擎选择: "qwen3_transformers" | "qwen3_vllm" | "faster_whisper"
  engine: "qwen3_transformers"

  # --- Qwen3 通用参数 ---
  qwen3:
    model_name: "Qwen/Qwen3-ASR-0.6B"     # 0.6B 或 1.7B 或本地路径 "./models/Qwen3-ASR-0.6B"
    device: "cuda:0"                       # cuda:0 或 cpu
    dtype: "bfloat16"                      # float16 / bfloat16 / auto

  # --- Qwen3 vLLM 专用参数 ---
  qwen3_vllm:
    gpu_memory_utilization: 0.7            # GPU 显存占比 (0.0~1.0)
    max_inference_batch_size: 128
    max_new_tokens: 4096

  # --- faster-whisper 专用参数 ---
  faster_whisper:
    model_size: "base"                     # tiny / base / small / medium / large-v3
    device: "cuda"                         # cuda 或 cpu
    compute_type: "float16"                # float16 / int8_float16 / int8

  # --- 通用推理参数 ---
  language: null                           # null=自动检测, 或指定 "zh"/"en"
  max_inference_batch_size: 32             # 批次大小上限
  max_new_tokens: 256                      # 最大生成 token 数

  # --- 时间戳（Qwen3 需要 ForcedAligner）---
  return_timestamps: false
  forced_aligner: null                     # "Qwen/Qwen3-ForcedAligner-0.6B" 启用时间戳

# -------- 音频捕获配置 --------
audio:
  sample_rate: 16000                       # 16kHz（固定，ASR 标准）
  channels: 1                              # 单声道
  chunk_duration_ms: 32                    # 每帧 32ms = 512 采样点
  loopback_device: null                    # null=自动选默认设备, 或指定关键词如 "Senary"

# -------- VAD 语音活动检测 --------
vad:
  enabled: true                            # Qwen3 vLLM 流式模式下建议 false
  threshold: 0.5                           # 语音概率阈值 (0.0~1.0)
  min_silence_duration_ms: 500             # 静音多久算断句结束
  min_speech_duration_ms: 300              # 最短有效语音
  speech_pad_ms: 100                       # 语音段前后填充

# -------- 输出配置 --------
output:
  console:
    enabled: true                          # 是否控制台实时打印
    show_timestamp: true                   # 是否显示时间戳前缀
  file:
    enabled: true                          # 是否保存到文件
    output_dir: "./output"                 # 输出目录
    filename_pattern: "session_{timestamp}.txt"  # 文件名模板
```

---

## 六、各层模块详细设计

### 6.1 音频捕获层（audio/）

**文件**：`livescribe/audio/capture.py`

```
类：AudioCapture
  方法：
    __init__(config: AudioConfig)
        - 获取默认扬声器
        - 通过 soundcard 获取对应 loopback 设备
        - 创建 recorder

    open() → None
        - 启动录制流

    read_chunk() → np.ndarray
        - 返回 shape=(512,) float32 数组，范围 [-1.0, 1.0]
        - 阻塞直到读取一帧 (32ms)

    close() → None
        - 停止录制

    list_devices() → List[dict]  [静态方法]
        - 枚举所有可用 loopback 设备
```

**文件**：`livescribe/audio/device.py`

```
函数：
    get_default_loopback_device() → soundcard.Microphone
    list_loopback_devices() → List[dict]
    find_device_by_keyword(keyword: str) → Optional[soundcard.Microphone]
```

**PCM 规格**：16000 Hz, 1 channel, float32（内部）→ 需要时转为 int16

---

### 6.2 VAD 语音检测层（vad/）

**文件**：`livescribe/vad/detector.py`

```
类：VADDetector
  方法：
    __init__(config: VADConfig)
        - 通过 torch.hub 加载 silero-vad 模型
        - 创建 VADIterator（流式状态管理）

    process(chunk: np.ndarray) → Optional[Literal["start", "end"]]
        - 输入 shape=(512,) float32 数组
        - 返回 "start"（语音开始）/ "end"（语音结束）/ None（无事件）

    reset() → None
        - 重置 LSTM 状态

    close() → None
        - 释放资源
```

**VAD 状态机流程**：
```
静音状态 ──(检测到语音)──▶ 语音状态 ──(静音超过 min_silence)──▶ 触发 "end"
                            │
                            └──(持续语音)── 保持语音状态
```

---

### 6.3 ASR 引擎层（asr/）

#### 6.3.1 Qwen3 Transformers 引擎

**文件**：`livescribe/asr/qwen3_transformers_engine.py`

```
类：Qwen3TransformersEngine(BaseASREngine)

  initialize():
    1. Qwen3ASRModel.from_pretrained(config.qwen3.model_name, ...)
    2. 加载可选的 ForcedAligner

  transcribe(audio, sample_rate) → str:
    1. 确保音频是 16kHz float32
    2. 调用 model.transcribe(audio=(audio, 16000), language=...)
    3. 返回 results[0].text

  shutdown():
    1. model.cpu()
    2. del model
    3. torch.cuda.empty_cache()

  supports_streaming → False
  engine_name → "qwen3_transformers"
```

#### 6.3.2 Qwen3 vLLM 引擎

**文件**：`livescribe/asr/qwen3_vllm_engine.py`

```
类：Qwen3VllmEngine(BaseASREngine)

  initialize():
    1. Qwen3ASRModel.LLM(model=..., gpu_memory_utilization=..., ...)
    （注意：必须在 if __name__ == "__main__": 保护下调用）

  transcribe(audio, sample_rate) → str:
    1. 调用 model.transcribe(audio=(audio, 16000), ...)
    2. 返回结果

  shutdown():
    1. 释放 vLLM 资源

  supports_streaming → True
  engine_name → "qwen3_vllm"

  注意：
    - vLLM 原生支持流式推理，但需要 WebSocket 接口
    - 简单模式下退化为分段 transcribe()
    - 如果需要真正的 chunk-by-chunk 流式，需额外实现
```

#### 6.3.3 faster-whisper 引擎

**文件**：`livescribe/asr/faster_whisper_engine.py`

```
类：FasterWhisperEngine(BaseASREngine)

  initialize():
    1. from faster_whisper import WhisperModel
    2. model = WhisperModel(model_size, device=device, compute_type=...)

  transcribe(audio, sample_rate) → str:
    1. 确保 16kHz float32
    2. segments, info = model.transcribe(audio, language=...)
    3. 拼接所有 segment.text，返回完整文本

  shutdown():
    1. del model
    2. gc.collect()

  supports_streaming → False
  engine_name → "faster_whisper"
```

---

### 6.4 输出层（output/）

**文件**：`livescribe/output/base_output.py`

```
类：BaseOutput (ABC)
  abstract emit(text: str) → None
  abstract close() → None
```

**文件**：`livescribe/output/console.py`

```
类：ConsoleOutput(BaseOutput)
  emit(text): print(f"[{timestamp}] {text}")
```

**文件**：`livescribe/output/file_writer.py`

```
类：FileWriter(BaseOutput)
  emit(text): 追加写入 output/session_xxx.txt
  支持 open / flush / close
```

**输出层管理**：
```
类：OutputManager
  - 持有 List[BaseOutput]
  - emit_all(text): 遍历所有输出器，逐个调用 emit()
  - 支持同时输出到控制台 + 文件，或仅其中一个
```

---

### 6.5 主流水线（pipeline.py）

```
类：Pipeline

  __init__(config_path: str):
    1. 加载 config.yaml
    2. 创建 AudioCapture
    3. 创建 VADDetector（如果启用）
    4. create_asr_engine(config.asr) → 工厂创建引擎
    5. 创建 OutputManager

  run():
    buffer = []          # 累积当前语音段的音频帧
    is_speaking = False  # 当前是否在说话

    while 未收到停止信号:
      chunk = audio_capture.read_chunk()   # (512,) float32

      if vad_enabled:
        event = vad.process(chunk)
        if event == "start":
          is_speaking = True
          # 保留回溯 buffer
        elif event == "end" and is_speaking:
          # 语音段结束 → 转写
          full_audio = concat(buffer)
          text = asr_engine.transcribe(full_audio)
          output_manager.emit_all(text)
          buffer = []
          is_speaking = False
          vad.reset()

      if is_speaking or not vad_enabled:
        buffer.append(chunk)

      # 无 VAD 模式：累积到一定时长就转写
      if not vad_enabled and len(buffer) * 32 / 1000 >= MAX_CHUNK_SEC:
        full_audio = concat(buffer)
        text = asr_engine.transcribe(full_audio)
        output_manager.emit_all(text)
        buffer = []

  stop():
    audio_capture.close()
    asr_engine.shutdown()
    output_manager.close()
```

---

## 七、数据流总览

```
系统音频播放
      │
      ▼
┌──────────────────────┐
│  WASAPI Loopback      │  soundcard 库
│  默认扬声器 → 采集     │
│  16kHz, mono, float32 │
└──────────┬───────────┘
           │ 每 32ms 一帧 (512 采样点)
           ▼
┌──────────────────────┐
│  VAD 语音检测         │  Silero VAD (可选)
│  判断是否有人声        │
│  → "start" / "end"   │
└──────────┬───────────┘
           │ 语音段边界事件
           ▼
┌──────────────────────┐
│  音频缓冲 Buffer       │  累积 PCM 帧
│  list[np.ndarray]     │  直到 VAD 触发 "end"
└──────────┬───────────┘
           │ concat → np.ndarray(num_samples,)
           ▼
┌──────────────────────┐
│  ASR 引擎 (策略模式)   │
│  ┌─────────────────┐ │
│  │ Qwen3 Transf    │ │  ← 工厂选择
│  │ Qwen3 vLLM      │ │
│  │ faster-whisper   │ │
│  └─────────────────┘ │
│  model.transcribe()   │
│   → text: str         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  输出层               │
│  ├→ Console: 实时打印 │
│  └→ File: 写入 .txt   │
└──────────────────────┘
```

---

## 八、实施步骤

### Phase 1：环境搭建（目标：可用的 Python 环境 + 依赖）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1.1 | 创建 conda 环境 `livescribe` (Python 3.12) | 干净环境 |
| 1.2 | 安装核心依赖：`qwen-asr`, `torch`, `soundcard`, `numpy`, `pyyaml` | 基础包可用 |
| 1.3 | 安装备选引擎依赖：`faster-whisper` | 备选引擎可用 |
| 1.4 | 预下载 Qwen3-ASR-0.6B 模型（HuggingFace / ModelScope）| 离线可用 |
| 1.5 | 编写 `requirements.txt` | 依赖清单 |

### Phase 2：项目骨架（目标：可运行的目录结构）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | 创建全部目录和 `__init__.py` | 包结构就绪 |
| 2.2 | 实现 `config/schema.py`（数据类 + 引擎枚举） | 配置类型定义 |
| 2.3 | 实现 `config/loader.py`（YAML 加载 + 校验） | 配置系统可用 |
| 2.4 | 编写 `config.yaml` | 配置文件就绪 |

### Phase 3：ASR 引擎层（目标：工厂模式 + 三种引擎可切换）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 3.1 | 实现 `asr/base.py`（抽象基类） | 引擎接口定义 |
| 3.2 | 实现 `asr/qwen3_transformers_engine.py` | Qwen3 Transformers 可用 |
| 3.3 | 实现 `asr/qwen3_vllm_engine.py` | Qwen3 vLLM 可用 |
| 3.4 | 实现 `asr/faster_whisper_engine.py` | faster-whisper 可用 |
| 3.5 | 实现 `asr/factory.py`（工厂函数） | 引擎切换可用 |
| 3.6 | 写单元测试：加载每个引擎 → transcribe 一个测试音频 → 输出文字 | 验证各引擎正常 |

### Phase 4：音频捕获层（目标：能抓到系统声音）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 4.1 | 实现 `audio/device.py`（设备枚举 + 选择） | 设备管理可用 |
| 4.2 | 实现 `audio/capture.py`（loopback PCM 流采集） | 音频流可用 |
| 4.3 | 写测试：录制 10 秒系统音频 → 保存 wav → 人耳验证 | 确认捕获正常 |

### Phase 5：VAD 语音检测层（目标：正确检测语音边界）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 5.1 | 实现 `vad/detector.py`（Silero VAD 封装 + 状态机） | VAD 模块可用 |
| 5.2 | 写测试：播放一段含静音的音频 → 验证 VAD 断句正确 | 确认检测正常 |

### Phase 6：输出层（目标：文字输出到控制台和文件）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 6.1 | 实现 `output/base_output.py`（抽象基类） | 接口定义 |
| 6.2 | 实现 `output/console.py` | 控制台输出可用 |
| 6.3 | 实现 `output/file_writer.py` | 文件输出可用 |
| 6.4 | 实现 `OutputManager`（管理多个输出器） | 输出管理可用 |

### Phase 7：主流水线 + 入口（目标：端到端可运行）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 7.1 | 实现 `pipeline.py`（串联 capture → VAD → ASR → output） | 流水线可用 |
| 7.2 | 实现 `main.py`（入口 + 信号处理 + 键盘停止） | 程序入口 |
| 7.3 | 端到端测试：播放中文视频 → 看控制台输出 | 验收通过 |

### Phase 8：打磨（可选）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 8.1 | 添加 GPU 显存监控日志 | 资源可见 |
| 8.2 | 添加设备热切换检测（蓝牙耳机断开等） | 健壮性提升 |
| 8.3 | 添加对 whisper（openai-whisper）的引擎适配 | 引擎矩阵+1 |
| 8.4 | 添加对 vosk 的引擎适配 | 引擎矩阵+2 |

---

## 九、扩展性验证

新增一个引擎（如 `vosk`）需要的改动范围：

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `livescribe/asr/vosk_engine.py` | 新建 ~60行 | 实现 BaseASREngine |
| `livescribe/asr/factory.py` | +3行 | 添加 elif 分支 |
| `livescribe/config/schema.py` | +1行 | 添加枚举值 |
| `config.yaml` | 无需改动 | 用户手动改 `engine` 字段 |
| 其他所有文件 | **0 行** | 完全不需要动 |

这证明了架构的可扩展性。

---

## 十、关键风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| RTX 3060 6GB 跑 1.7B 显存不够 | 高 | 无法加载模型 | 默认使用 0.6B，1.7B 作为可选 |
| vLLM 后端 Windows 兼容性 | 中 | 流式模式不可用 | 退回 Transformers + VAD 分段模式 |
| soundcard WASAPI loopback 不工作 | 低 | 无法捕获系统音频 | 备选 sounddevice 库 |
| 蓝牙耳机 loopback 静音 | 中 | 录到空音频 | 代码检测默认设备变更，提示用户 |
| bitsandbytes 4bit Windows 兼容性 | 高 | 1.7B 无法量化 | 放弃 4bit 量化，使用 0.6B 或显存更大的 GPU |

---

## 十一、依赖清单

```
# requirements.txt
qwen-asr>=1.0.0          # Qwen3 ASR 核心包
torch>=2.4.0             # PyTorch
faster-whisper>=1.0.0    # faster-whisper 引擎
soundcard>=0.4.3         # WASAPI loopback 音频捕获
numpy>=1.26.0            # 数组运算
pyyaml>=6.0              # 配置文件解析
soundfile>=0.12.0        # 音频文件读写（测试用）
silero-vad>=5.0          # 或通过 torch.hub 加载（内置于 PyTorch）
```
