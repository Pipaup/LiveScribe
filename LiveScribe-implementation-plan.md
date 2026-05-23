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
#  修改 mode 切换运行模式，修改 asr.engine 切换引擎
# ====================================================

# -------- 运行模式（核心切换点）--------
# "vad_segmented": VAD 断句 → 整段转写（适用非流式引擎：qwen3_transformers / faster_whisper）
# "streaming":     逐帧流式 → 增量输出（仅 supports_streaming=True 的引擎可用）
mode: "vad_segmented"

# -------- ASR 引擎配置 --------
asr:
  # 引擎选择: "qwen3_transformers" | "qwen3_vllm" | "faster_whisper"
  engine: "qwen3_transformers"

  # --- Qwen3 通用参数 ---
  qwen3:
    model_name: "Qwen/Qwen3-ASR-0.6B"     # 0.6B (~3GB 显存) 或 1.7B (~6GB)，支持本地路径
    device: "cuda:0"                       # cuda:0 或 cpu（显存不够时改为 cpu）
    dtype: "bfloat16"                      # float16 / bfloat16 / auto

  # --- Qwen3 vLLM 专用参数 ---
  qwen3_vllm:
    gpu_memory_utilization: 0.7            # GPU 显存占比 (0.0~1.0)
    max_inference_batch_size: 128
    max_new_tokens: 4096

  # --- faster-whisper 专用参数 ---
  faster_whisper:
    model_size: "medium"                   # tiny/base 中文差，建议 medium 起步
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
  sample_rate: 16000                       # 内部统一采样率（ASR 标准）
  channels: 1                              # 单声道
  chunk_duration_ms: 32                    # 每帧 32ms = 512 采样点
  loopback_device: null                    # null=自动选默认设备, 或指定关键词如 "Senary"
  resample_method: "linear"                # 重采样算法：linear / sinc（48k→16k 必经步骤）

# -------- VAD 语音活动检测（仅 mode=vad_segmented 时生效）--------
vad:
  enabled: true
  threshold: 0.5                           # 语音概率阈值 (0.0~1.0)
  min_silence_duration_ms: 400             # 静音多久算断句结束（200=激进, 800=保守）
  min_speech_duration_ms: 300              # 最短有效语音
  speech_pad_ms: 100                       # 语音段前后填充
  max_segment_duration_s: 15               # 单段最长秒数，防止 buffer 无限增长 / 超 token 限制
  model: "silero_vad"                      # 固定使用 silero_vad（PyTorch 版）

# -------- 流式模式参数（仅 mode=streaming 时生效）--------
streaming:
  result_interval_ms: 200                  # 增量结果刷新间隔

# -------- 输出配置 --------
output:
  console:
    enabled: true
    show_timestamp: true
    streaming_mode: "overwrite"            # "overwrite": 同行刷新 / "append": 逐行追加
  file:
    enabled: true
    output_dir: "./output"
    filename_pattern: "session_{timestamp}.txt"
    flush_interval_s: 5                    # 文件 flush 间隔（秒），平衡性能与数据安全

# -------- 日志配置 --------
logging:
  level: "INFO"                            # DEBUG / INFO / WARNING / ERROR
  file: "logs/livescribe.log"              # 日志文件路径，null 为仅控制台
```

---

## 六、各层模块详细设计

### 6.1 音频捕获层（audio/）

**文件**：`livescribe/audio/capture.py`

```
类：AudioCapture
  职责：从 loopback 设备持续读取 PCM 帧，输出统一格式的 float32 数组。

  关键设计决策：
    - 内部统一输出 16kHz / mono / float32，调用方不需要处理格式差异
    - capture 层内部自行处理 重采样（48k→16k）和 声道转换（stereo→mono）
    - 设备断开时 read_chunk() 抛 AudioDeviceError（而非静默返回零值）

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
        - 内部自动完成重采样 + 声道转换

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

**文件**：`livescribe/audio/resampler.py`

```
函数：
    resample(audio: np.ndarray, src_rate: int, dst_rate: int, method: str) → np.ndarray
        - 使用 scipy.signal.resample 或线性插值将音频重采样到目标采样率
        - src_rate: 原始采样率（如 48000）
        - dst_rate: 目标采样率（16000）
        - method: "linear" | "sinc"

    to_mono(audio: np.ndarray, channels: int) → np.ndarray
        - 多声道 → 单声道（均值混合）
```

**PCM 数据流（带格式转换链路）**：

```
soundcard 原生帧 (48kHz / stereo / int16)
        │
        ▼
  [resampler.resample]  48kHz → 16kHz
        │
        ▼
  [resampler.to_mono]   stereo → mono
        │
        ▼
  [类型转换]  int16 → float32 / 归一化到 [-1.0, 1.0]
        │
        ▼
  输出: (512,) float32  ← 各层统一使用此格式
```

---

### 6.2 VAD 语音检测层（vad/）

**文件**：`livescribe/vad/detector.py`

```
类：VADDetector
  职责：逐帧分析语音概率，通过状态机管理语音段边界。

  API 选型决策：
    选用 Silero VAD 的 VADIterator（流式逐帧模式），而非 get_speech_timestamps（整段分析）。
    原因：Pipeline 需要每 32ms 处理一帧，不能等整段音频。
    VADIterator 要求每帧 512 采样点（16kHz），与 AudioCapture 的 chunk 大小完全对齐。

  方法：
    __init__(config: VADConfig)
        - 通过 torch.hub 加载 silero_vad 模型（jit 版）
        - 创建 VADIterator（管理 LSTM 隐状态）
        - VADIterator 初始化参数：sample_rate=16000, threshold, min_silence_duration_ms,
          min_speech_duration_ms, speech_pad_ms

    process(chunk: np.ndarray) → Optional[Literal["start", "end"]]
        - chunk: shape=(512,) float32, 值域 [-1.0, 1.0]
        - 内部调用 VADIterator(chunk)，返回语音概率 + 状态
        - 返回 "start"（语音开始）/ "end"（语音结束）/ None（无事件）

    reset() → None
        - 重置 VADIterator 的 LSTM 隐状态
        - 每次 detect 到 "end" 后调用

    close() → None
        - 释放模型资源

  状态机内部流程：
    Silent ──(speech_prob > threshold, 持续 min_speech_duration_ms)──▶ Speaking
    Speaking ──(speech_prob < threshold, 持续 min_silence_duration_ms)──▶ 触发 "end" → Silent
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

  __init__(config, streaming_mode: str = "append"):
      streaming_mode:
        - "append":    每次 emit 新起一行，适合分段模式
        - "overwrite": 用 \r + sys.stdout.flush() 同行刷新，适合流式模式

  emit(text, is_partial: bool = False):
    分段模式：print(f"[{timestamp}] {text}")
    流式模式：
      当 is_partial=True 时用 overwrite 模式（同位置刷新）
      当 is_partial=False（最终结果）时换行输出
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

#### 6.5.1 线程模型（关键设计）

Pipeline 单线程会有一个致命问题：**ASR 推理阻塞 `read_chunk()`，推理期间的音频永久丢失。**

解决方案：**双线程 + 线程安全队列**。

```
┌─────────────────────────────────────────────────────┐
│                     Main Thread                       │
│  while running:                                      │
│    chunk = audio.read_chunk()  ← 永不阻塞（32ms）    │
│    audio_queue.put(chunk)                            │
│                                                      │
│    # 流式模式：也可以在这里处理 UI 输出               │
│    # 分段模式：VAD 状态机在主线程运行                 │
└──────────────────┬──────────────────────────────────┘
                   │ audio_queue (threading.Queue)
                   ▼
┌─────────────────────────────────────────────────────┐
│                   ASR Worker Thread                   │
│  while running:                                      │
│    # 分段模式：                                       │
│      从 queue 累积音频 → VAD 触发后 → transcribe()   │
│      → 结果放入 result_queue                         │
│                                                      │
│    # 流式模式：                                       │
│      从 queue 取帧 → feed_chunk() → 增量结果入 queue │
└──────────────────┬──────────────────────────────────┘
                   │ result_queue (threading.Queue)
                   ▼
┌─────────────────────────────────────────────────────┐
│               Output (主线程消费结果)                  │
│  从 result_queue 取结果 → emit_all()                 │
└─────────────────────────────────────────────────────┘
```

**Queue 设计要点**：
- `audio_queue`：容量上限 3000 帧（约 96 秒），防止内存爆炸。满了就丢弃最旧的帧（覆盖而非阻塞）
- `result_queue`：无容量上限，结果一帧不能丢
- 主线程从 `result_queue` 非阻塞取结果

#### 6.5.2 完整 Pipeline 伪代码

```
类：Pipeline

  __init__(config_path: str):
    1. 加载 config.yaml
    2. 创建 AudioCapture
    3. 创建 VADDetector（如果 mode == "vad_segmented"）
    4. create_asr_engine(config.asr) → 工厂创建引擎
    5. 创建 OutputManager
    6. 初始化 audio_queue (Queue, maxsize=3000), result_queue (Queue)
    7. running = False

  run():
    running = True

    # 启动 ASR 工作线程
    asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
    asr_thread.start()

    # 主线程：音频采集 + VAD（分段模式）
    try:
      if config.mode == "vad_segmented":
        self._run_segmented_capture()
      elif config.mode == "streaming":
        self._run_streaming_capture()
    finally:
      self.stop()

  # ==================== 分段模式 ====================

  _run_segmented_capture():
    """主线程：读帧 + VAD 状态机"""
    buffer = []          # 当前语音段音频帧
    is_speaking = False
    silent_frames = 0

    while running:
      chunk = audio.read_chunk()     # (512,) float32
      audio_queue.put(chunk)         # 喂给 ASR 线程（ASR 线程只累积不推理）

      if not vad_enabled:
        # 无 VAD 模式：盲切
        buffer.append(chunk)
        if len(buffer) * 32 / 1000 >= config.vad.max_segment_duration_s:
          self._send_to_asr(concat(buffer))
          buffer = []
        continue

      # 有 VAD 模式
      event = vad.process(chunk)

      if event == "start":
        is_speaking = True
        # 回溯：保留 start 前 speech_pad_ms 的帧
        pad_frames = int(config.vad.speech_pad_ms / 32)
        buffer = buffer[-pad_frames:] if pad_frames > 0 else buffer

      elif event == "end" and is_speaking:
        # 语音段结束 → 发给 ASR 线程
        self._send_to_asr(concat(buffer))
        buffer = []
        is_speaking = False
        vad.reset()

      # 强制截断保护（max_segment_duration_s）
      if is_speaking and len(buffer) * 32 / 1000 >= config.vad.max_segment_duration_s:
        self._send_to_asr(concat(buffer))
        buffer = buffer[-pad_frames:]  # 保留末尾作为下一段上下文
        is_speaking = True             # 保持说话状态

      if is_speaking or not vad_enabled:
        buffer.append(chunk)

      # 消费 ASR 结果（非阻塞）
      self._drain_results()

    # 循环结束：处理最后一段
    if buffer:
      self._send_to_asr(concat(buffer))
    self._send_to_asr(None)  # 哨兵：通知 ASR 线程退出

  # ==================== 流式模式 ====================

  _run_streaming_capture():
    """主线程：读帧 → 喂入队列，逐帧不等待"""
    engine.start_stream()

    while running:
      chunk = audio.read_chunk()
      audio_queue.put(chunk)
      self._drain_results()

    self._send_to_asr(None)  # 哨兵
    final = engine.end_stream()
    if final:
      result_queue.put(("final", final))

  # ==================== ASR 工作线程 ====================

  _asr_worker():
    """独立线程：从 audio_queue 取帧 → 推理 → 结果放入 result_queue"""
    buffer = []

    while True:
      chunk = audio_queue.get()  # 阻塞等待

      if chunk is None:  # 哨兵：退出
        if buffer:
          text = engine.transcribe(concat(buffer))
          result_queue.put(("final", text))
        break

      if config.mode == "vad_segmented":
        # 分段模式：等待主线程组装好的完整段
        # 主线程通过 _send_to_asr() 发送 (is_segment, data) 标记
        # 实际实现中 audio_queue 存的是带标记的数据
        ...
      elif config.mode == "streaming":
        partial = engine.feed_chunk(chunk)
        if partial:
          result_queue.put(("partial", partial))

  _send_to_asr(segment: np.ndarray):
    """主线程 → ASR 线程：发送完整语音段"""
    audio_queue.put(("segment", segment))

  _drain_results():
    """主线程：非阻塞消费 ASR 结果"""
    while True:
      try:
        tag, text = result_queue.get_nowait()
        if tag == "partial":
          output_manager.emit_all(text, is_partial=True)
        else:
          output_manager.emit_all(text, is_partial=False)
      except queue.Empty:
        break

  # ==================== 生命周期 ====================

  stop():
    running = False
    audio.close()
    engine.shutdown()
    output_manager.close()
```

#### 6.5.3 两种模式的流程图

**分段模式 (vad_segmented)**：
```
[Audio Loopback]
      │ 每 32ms 一帧
      ▼
[主线程: AudioCapture + VAD 状态机]
      │ VAD 触发 "end" 或 超 max_segment_duration_s
      │ concat(buffer) → 完整语音段
      ▼
[audio_queue] ──────▶ [ASR 线程: engine.transcribe(segment)]
                              │
                              ▼
                       [result_queue] ──▶ [主线程: Output]
```

**流式模式 (streaming)**：
```
[Audio Loopback]
      │ 每 32ms 一帧
      ▼
[主线程: AudioCapture]
      │ 逐帧直接放入 queue
      ▼
[audio_queue] ──────▶ [ASR 线程: engine.feed_chunk(chunk)]
                              │ 每 ~200ms 返回增量文字
                              ▼
                       [result_queue] ──▶ [主线程: Output 同行刷新]
```

---

## 七、错误处理策略

### 7.1 分层异常定义

```
livescribe/
├── exceptions.py          # 项目自定义异常
│   ├── LiveScribeError    # 基类
│   ├── AudioDeviceError   # 音频设备断开/不可用
│   ├── ASREngineError     # ASR 推理失败
│   ├── ConfigError        # 配置校验失败
│   └── PipelineError      # 流水线运行时错误
```

### 7.2 各层错误处理策略

| 层级 | 错误场景 | 处理方式 |
|------|---------|---------|
| **Audio** | 设备断开（蓝牙耳机拔出） | 抛 `AudioDeviceError` → Pipeline 捕获 → 日志告警 → 尝试重新打开设备（最多 3 次，间隔 1s） |
| **Audio** | `read_chunk()` 返回空帧 | 丢弃，计数。连续 100 帧空帧 → 可能设备静音/断开 → 日志 Warning |
| **VAD** | 模型加载失败 | 抛 `ConfigError` → 程序退出（VAD 是可选模块，可降级为盲切模式运行） |
| **ASR** | `initialize()` 加载 OOM | 抛 `ASREngineError("显存不足")` → 提示用户切换 CPU 模式或更小模型 |
| **ASR** | `transcribe()` 推理失败 | 抛 `ASREngineError` → Pipeline 捕获 → 日志 Error → 跳过该段，继续运行（不崩溃） |
| **ASR** | `transcribe()` 超时（30s） | 抛 `ASREngineError("timeout")` → 同上处理 |
| **ASR** | `transcribe()` 返回空字符串 | 不抛异常，日志 Warning → 正常 emit（可能是静音段被误判） |
| **Pipeline** | audio_queue 满 | 丢弃最旧帧（覆盖），日志 Warning 一次 |
| **Config** | `loopback_device` 填了但不存在的设备名 | 抛 `ConfigError` → 列出所有可用设备，让用户选 |
| **Config** | 数值校验失败（负数、越界） | 抛 `ConfigError` → 指出具体字段和允许范围 |

### 7.3 引擎加载失败降级链

```
尝试加载首选引擎 (config.asr.engine)
    │
    ├── 成功 → 运行
    │
    └── 失败（OOM / 模型文件不存在）
        │
        ├── 如果是 GPU 模式 → 提示改为 device: "cpu" 重试
        ├── 如果是 1.7B 模型 → 建议降级到 0.6B
        └── 如果 0.6B GPU 仍失败 → 建议改为 faster_whisper + CPU
```

---

## 八、日志系统设计

### 8.1 设计原则

- 统一使用 Python `logging` 标准库，不引入第三方日志框架
- 各模块通过 `logging.getLogger(__name__)` 获取 logger
- `livescribe/logging_utils.py` 提供 `setup_logging(config)` 集中配置

### 8.2 模块设计

**文件**：`livescribe/logging_utils.py`

```
函数：
  setup_logging(config: LoggingConfig) → None:
    1. 设置 root logger 的 level
    2. 添加 StreamHandler（控制台输出，INFO 级别以上）
    3. 如果配置了 log_file，添加 RotatingFileHandler（5MB 轮转，保留 3 个备份）
    4. 设置格式："%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### 8.3 日志级别使用约定

| 级别 | 使用场景 | 示例 |
|------|---------|------|
| DEBUG | 每帧 VAD 概率、音频 RMS 值 | `VAD prob=0.23, RMS=-42dB` |
| INFO | 生命周期事件、引擎切换 | `"ASR engine loaded: qwen3_transformers (0.6B, cuda:0)"` |
| WARNING | 可恢复的异常、降级 | `"Audio queue full, dropping 50 frames"` |
| ERROR | 推理失败、设备断开 | `"ASR transcribe failed: CUDA OOM"` |

---

## 九、更新后的数据流总览

### 分段模式 (mode=vad_segmented)

```
系统音频播放
      │
      ▼
┌─────────────────────────┐
│  WASAPI Loopback         │  soundcard 库
│  48kHz stereo → 重采样    │  resampler.py
│  → 16kHz, mono, float32  │
└──────────┬──────────────┘
           │ 每 32ms 一帧 (512 采样点)
           ▼
┌─────────────────────────┐
│  [主线程]                │
│  VAD 状态机 + Buffer      │  Silero VADIterator
│  判断语音边界             │
│  → 累积音频帧             │
│  → 触发 "end" 或超时截断  │
└──────────┬──────────────┘
           │ concat → np.ndarray(num_samples,)
           │ 放入 audio_queue
           ▼
      ┌────────────────────┐
      │   audio_queue       │  threading.Queue (maxsize=3000)
      └────────┬───────────┘
               │
               ▼
┌─────────────────────────┐
│  [ASR 线程]              │
│  策略模式引擎             │
│  ┌───────────────────┐  │
│  │ Qwen3 Transf      │  │  ← 工厂选择
│  │ Qwen3 vLLM        │  │
│  │ faster-whisper     │  │
│  └───────────────────┘  │
│  engine.transcribe()     │
│  → text: str             │
└──────────┬──────────────┘
           │ 放入 result_queue
           ▼
      ┌────────────────────┐
      │   result_queue      │  threading.Queue
      └────────┬───────────┘
               │
               ▼
┌─────────────────────────┐
│  [主线程] 输出层          │
│  ├→ Console: print 换行  │
│  └→ File: 追加写入 .txt   │
└─────────────────────────┘
```

### 流式模式 (mode=streaming)

```
系统音频播放
      │
      ▼
┌─────────────────────────┐
│  WASAPI Loopback         │
│  16kHz, mono, float32    │
└──────────┬──────────────┘
           │ 每 32ms 一帧
           ▼
      ┌────────────────────┐
      │   audio_queue       │
      └────────┬───────────┘
               │
               ▼
┌─────────────────────────┐
│  [ASR 线程]              │
│  engine.feed_chunk()     │  逐帧增量解码
│  → 每 ~200ms 出增量文字  │
└──────────┬──────────────┘
           │ "partial" / "final"
           ▼
      ┌────────────────────┐
      │   result_queue      │
      └────────┬───────────┘
               │
               ▼
┌─────────────────────────┐
│  [主线程] 输出层          │
│  ├→ Console: 同行刷新     │  \r + flush
│  └→ File: 追加写入 .txt   │
└─────────────────────────┘
```

---

## 十、实施步骤

### Phase 1：环境搭建（目标：可用的 Python 环境 + 依赖）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1.1 | 创建 conda 环境 `livescribe` (Python 3.12) | 干净环境 |
| 1.2 | 安装 PyTorch（确认 CUDA 版本匹配，conda 安装 GPU 版） | torch 可用 |
| 1.3 | 安装核心依赖：`qwen-asr`, `soundcard`, `numpy`, `pyyaml`, `scipy` | 基础包可用 |
| 1.4 | 安装备选引擎依赖：`faster-whisper` | 备选引擎可用 |
| 1.5 | 预下载 Qwen3-ASR-0.6B 模型（HuggingFace / ModelScope）| 离线可用 |
| 1.6 | 编写 `requirements.txt`（含版本号 pin） | 依赖清单 |

### Phase 2：项目骨架（目标：可运行的目录结构）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | 创建全部目录和 `__init__.py` | 包结构就绪 |
| 2.2 | 实现 `config/schema.py`（数据类 + 引擎枚举 + mode 枚举） | 配置类型定义 |
| 2.3 | 实现 `config/loader.py`（YAML 加载 + 数值范围校验 + 设备存在性校验） | 配置系统可用 |
| 2.4 | 实现 `exceptions.py`（自定义异常类） | 异常体系就绪 |
| 2.5 | 实现 `logging_utils.py`（统一日志配置） | 日志系统可用 |
| 2.6 | 编写 `config.yaml` | 配置文件就绪 |

### Phase 3：ASR 引擎层（目标：工厂模式 + 三种引擎可切换）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 3.1 | 实现 `asr/base.py`（抽象基类，含流式接口） | 引擎接口定义 |
| 3.2 | 实现 `asr/qwen3_transformers_engine.py` | Qwen3 Transformers 可用 |
| 3.3 | 实现 `asr/qwen3_vllm_engine.py` | Qwen3 vLLM 可用 |
| 3.4 | 实现 `asr/faster_whisper_engine.py` | faster-whisper 可用 |
| 3.5 | 实现 `asr/factory.py`（工厂函数 + 加载失败降级逻辑） | 引擎切换可用 |
| 3.6 | 写单元测试：加载每个引擎 → transcribe 一个测试音频 → 输出文字 | 验证各引擎正常 |

### Phase 4：音频捕获层（目标：能抓到系统声音）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 4.1 | 实现 `audio/device.py`（设备枚举 + 关键词匹配） | 设备管理可用 |
| 4.2 | 实现 `audio/resampler.py`（重采样 + 声道转换） | 格式转换可用 |
| 4.3 | 实现 `audio/capture.py`（loopback PCM 流采集，内嵌重采样） | 音频流可用 |
| 4.4 | 写测试：录制 10 秒系统音频 → 保存 wav → 人耳验证 | 确认捕获正常 |

### Phase 5：VAD 语音检测层（目标：正确检测语音边界）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 5.1 | 实现 `vad/detector.py`（Silero VADIterator + 状态机） | VAD 模块可用 |
| 5.2 | 写测试：播放一段含静音的音频 → 验证 VAD 断句正确 | 确认检测正常 |

### Phase 6：输出层（目标：文字输出到控制台和文件）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 6.1 | 实现 `output/base_output.py`（抽象基类，emit 加 is_partial 参数） | 接口定义 |
| 6.2 | 实现 `output/console.py`（分段逐行 + 流式同行刷新） | 控制台输出可用 |
| 6.3 | 实现 `output/file_writer.py`（定时 flush） | 文件输出可用 |
| 6.4 | 实现 `OutputManager`（管理多个输出器） | 输出管理可用 |

### Phase 7：主流水线 + 入口（目标：端到端可运行）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 7.1 | 实现 `pipeline.py`（双线程 + 双模式 + 结果消费） | 流水线可用 |
| 7.2 | 实现 `main.py`（入口 + Ctrl+C 信号处理 + 优雅退出） | 程序入口 |
| 7.3 | 端到端测试：播放中文视频 → 看控制台输出 | 验收通过 |
| 7.4 | 模式切换测试：vad_segmented / streaming 两种模式都跑通 | 双模式验证 |

### Phase 8：打磨（可选）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 8.1 | 添加 GPU 显存监控日志 | 资源可见 |
| 8.2 | 添加设备热切换检测（蓝牙耳机断开等） | 健壮性提升 |
| 8.3 | 添加对 whisper（openai-whisper）的引擎适配 | 引擎矩阵+1 |
| 8.4 | 添加对 vosk 的引擎适配 | 引擎矩阵+2 |
| 8.5 | 添加对 SenseVoice / FunASR 的引擎适配 | 引擎矩阵+3/4 |

---

## 十一、扩展性验证

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

## 十二、关键风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| RTX 3060 6GB 空闲仅 2.5GB，0.6B 模型 ~3GB 显存不够 | 高 | 无法加载模型 | 首先设为 `device: cpu` 验证可用，然后尝试 `dtype: float16` + 减少其他 GPU 占用 |
| 0.6B CPU 推理太慢（>5 秒每段） | 中 | 实时性差 | 切换到 faster_whisper（CTranslate2 CPU 推理快很多） |
| vLLM 后端 Windows 兼容性 | 中 | 流式模式不可用 | 退回 Transformers + VAD 分段模式；流式需求走 vosk |
| soundcard WASAPI loopback 不工作 | 低 | 无法捕获系统音频 | 备选 sounddevice 库 |
| 蓝牙耳机 loopback 静音 | 中 | 录到空音频 | 代码检测默认设备变更 + 连续空帧计数 → 日志 Warning + 提示用户 |
| bitsandbytes 4bit Windows 兼容性 | 高 | 1.7B 无法量化 | 放弃 4bit 量化，使用 0.6B 或显存更大的 GPU |
| ASR 推理时间过长导致 audio_queue 堆积 | 中 | 丢帧 / 延迟累积 | audio_queue 设容量上限 + 满时丢弃策略 + 日志告警；排查是否模型/显存瓶颈 |
| faster-whisper base 模型中文 WER ~15-20%，体验差 | 中 | 识别不准确 | 默认改为 medium 模型（1.5GB）；在配置注释中说明各模型预期准确率 |
| torch CUDA 版本与系统驱动不匹配 | 中 | GPU 推理不可用 | Phase 1 中强调 conda 安装 PyTorch GPU 版并验证 `torch.cuda.is_available()` |
| Qwen3ASR transcribe() 偶发卡死（HF transformers 已知问题） | 低 | 单段永远不出结果 | ASR 线程设 30s 超时，超时则跳过该段 |

---

## 十三、依赖清单

```
# requirements.txt
qwen-asr>=1.0.0          # Qwen3 ASR 核心包
torch>=2.4.0             # PyTorch（conda 安装 GPU 版）
faster-whisper>=1.0.0    # faster-whisper 引擎（CTranslate2）
soundcard>=0.4.3         # WASAPI loopback 音频捕获
numpy>=1.26.0            # 数组运算
pyyaml>=6.0              # 配置文件解析
scipy>=1.12.0            # 重采样（resample / decimate）
soundfile>=0.12.0        # 音频文件读写（测试用）
# silero-vad: 通过 torch.hub 加载，不通过 pip 安装
# 注意：PyTorch 必须通过 conda 安装 GPU 版（cuda 版本与 nvidia-smi 匹配）
```

---

## 十四、成功标准（Definition of Done）

| 阶段 | 验收条件 |
|------|---------|
| **Phase 3 完成** | 每个引擎独立加载，用一个 5 秒中文测试音频跑 `transcribe()`，输出文字可读 |
| **Phase 4 完成** | 录制 10 秒系统音频（如播放 YouTube 视频）→ 保存 wav → 人耳确认内容完整、无静音段 |
| **Phase 5 完成** | 播放一段"说话-静音-说话-静音"的音频，日志确认 VAD 在正确边界触发 start/end |
| **Phase 7 完成** | 播放一段 1 分钟中文播客 → 控制台看到实时文字输出 → 文件有对应 txt → 延迟 < 3s（分段模式）/ < 500ms（流式模式） |
| **双模式验证** | 改 config mode 切换 → 重启程序 → 两种模式都能正常运行到输出 |

---

## 十五、不做的事情（Scope Exclusion）

以下功能明确不在当前范围内，防止开发时 scope creep：

- ❌ GUI 界面（桌面窗口、Web UI）
- ❌ 麦克风输入（只做系统音频 loopback 捕获）
- ❌ 实时翻译（只做 ASR 转写，不做翻译流水线）
- ❌ 热词/自定义词典
- ❌ 多路同时转写
- ❌ 配置热更新（改 config.yaml 需重启程序）
- ❌ 打包为 exe（PyInstaller / Nuitka）
- ❌ Docker 部署

---

## 十六、测试策略

### 测试框架

- **pytest** 作为测试框架
- 测试文件放在 `tests/` 目录下，镜像源码结构

```
tests/
├── test_config/
│   ├── test_schema.py      # 枚举值、数据类字段校验
│   └── test_loader.py      # YAML 加载、边界值、错误配置
├── test_audio/
│   └── test_resampler.py   # 重采样正确性
├── test_asr/
│   ├── test_base.py        # 接口契约（子类必须实现抽象方法）
│   ├── test_qwen3_transformer.py
│   ├── test_faster_whisper.py
│   └── test_factory.py     # 工厂：正确引擎 + 未知引擎报错 + 降级逻辑
├── test_vad/
│   └── test_detector.py    # 状态机逻辑、边界条件下行为
├── test_output/
│   └── test_console.py     # 输出格式
└── test_pipeline.py        # 集成测试（mock 音频源）
```

### 测试分级

| 级别 | 内容 | 运行频率 |
|------|------|---------|
| **单元测试** | 工厂函数、配置校验、VAD 状态机、resampler 正确性 | 每次 git commit 前 |
| **集成测试** | mock 音频源 → Pipeline → 验证 ASR 被正确调用、输出格式正确 | 每次 push 前 |
| **端到端测试** | 播放真实音频 → 运行 LiveScribe → 人工检查输出 | Phase 完成时 / 发版前 |

### 不测试的内容

- 不 mock ASR 模型内部行为（那是 Qwen/whisper 库的职责）
- 不测试 soundcard 库的 WASAPI 行为（硬件/驱动差异，靠人工验证）
