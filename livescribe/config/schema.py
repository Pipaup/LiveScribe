"""Configuration data classes and enums for LiveScribe."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EngineType(str, Enum):
    """Supported ASR engine identifiers."""

    QWEN3_TRANSFORMERS = "qwen3_transformers"
    QWEN3_VLLM = "qwen3_vllm"
    FASTER_WHISPER = "faster_whisper"


class RunMode(str, Enum):
    """Pipeline operation mode."""

    VAD_SEGMENTED = "vad_segmented"  # VAD segments → batch transcribe
    STREAMING = "streaming"  # Frame-by-frame incremental decode


@dataclass
class Qwen3Config:
    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"


@dataclass
class Qwen3VllmConfig:
    gpu_memory_utilization: float = 0.7
    max_inference_batch_size: int = 128
    max_new_tokens: int = 4096


@dataclass
class FasterWhisperConfig:
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class ASRConfig:
    engine: EngineType = EngineType.QWEN3_TRANSFORMERS
    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    qwen3_vllm: Qwen3VllmConfig = field(default_factory=Qwen3VllmConfig)
    faster_whisper: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)
    language: Optional[str] = None
    max_inference_batch_size: int = 32
    max_new_tokens: int = 256
    return_timestamps: bool = False
    forced_aligner: Optional[str] = None


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 32
    loopback_device: Optional[str] = None
    resample_method: str = "linear"


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_silence_duration_ms: int = 400
    min_speech_duration_ms: int = 300
    speech_pad_ms: int = 100
    max_segment_duration_s: int = 15
    model: str = "silero_vad"
    model_path: str = "./models/silero_vad.jit"


@dataclass
class StreamingConfig:
    result_interval_ms: int = 200


@dataclass
class OutputConsoleConfig:
    enabled: bool = True
    show_timestamp: bool = True
    streaming_mode: str = "overwrite"


@dataclass
class OutputFileConfig:
    enabled: bool = True
    output_dir: str = "./output"
    filename_pattern: str = "session_{timestamp}.txt"
    flush_interval_s: int = 5


@dataclass
class OutputConfig:
    console: OutputConsoleConfig = field(default_factory=OutputConsoleConfig)
    file: OutputFileConfig = field(default_factory=OutputFileConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = "logs/livescribe.log"


@dataclass
class LiveScribeConfig:
    mode: RunMode = RunMode.VAD_SEGMENTED
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
