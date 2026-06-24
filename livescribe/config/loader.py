"""YAML configuration loading and validation."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from livescribe.config.schema import (
    ASRConfig,
    AudioConfig,
    EngineType,
    FasterWhisperConfig,
    LiveScribeConfig,
    LoggingConfig,
    OutputConfig,
    OutputConsoleConfig,
    OutputFileConfig,
    Qwen3Config,
    Qwen3VllmConfig,
    RunMode,
    StreamingConfig,
    VADConfig,
)
from livescribe.exceptions import ConfigError

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_VALID_RESAMPLE_METHODS = {"linear", "sinc"}
_VALID_STREAMING_MODES = {"overwrite", "append"}
_VALID_DTYPES = {"float16", "bfloat16", "auto"}


def _get_str(data: dict, key: str, default: str) -> str:
    val = data.get(key, default)
    if not isinstance(val, str):
        raise ConfigError(f"'{key}' must be a string, got {type(val).__name__}")
    return val


def _get_optional_str(data: dict, key: str) -> Optional[str]:
    val = data.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ConfigError(f"'{key}' must be a string or null, got {type(val).__name__}")
    return val


def _get_int(data: dict, key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    val = data.get(key, default)
    if not isinstance(val, int) or isinstance(val, bool):
        raise ConfigError(f"'{key}' must be an integer, got {type(val).__name__}")
    if min_val is not None and val < min_val:
        raise ConfigError(f"'{key}' must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise ConfigError(f"'{key}' must be <= {max_val}, got {val}")
    return val


def _get_float(data: dict, key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    val = data.get(key, default)
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ConfigError(f"'{key}' must be a number, got {type(val).__name__}")
    val = float(val)
    if min_val is not None and val < min_val:
        raise ConfigError(f"'{key}' must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise ConfigError(f"'{key}' must be <= {max_val}, got {val}")
    return val


def _get_bool(data: dict, key: str, default: bool) -> bool:
    val = data.get(key, default)
    if not isinstance(val, bool):
        raise ConfigError(f"'{key}' must be a boolean, got {type(val).__name__}")
    return val


def _parse_qwen3(data: dict) -> Qwen3Config:
    dtype = _get_str(data, "dtype", "bfloat16")
    if dtype not in _VALID_DTYPES:
        raise ConfigError(f"asr.qwen3.dtype must be one of {_VALID_DTYPES}, got '{dtype}'")
    return Qwen3Config(
        model_name=_get_str(data, "model_name", "Qwen/Qwen3-ASR-0.6B"),
        device=_get_str(data, "device", "cuda:0"),
        dtype=dtype,
    )


def _parse_qwen3_vllm(data: dict) -> Qwen3VllmConfig:
    return Qwen3VllmConfig(
        gpu_memory_utilization=_get_float(data, "gpu_memory_utilization", 0.7, 0.0, 1.0),
        max_inference_batch_size=_get_int(data, "max_inference_batch_size", 128, 1),
        max_new_tokens=_get_int(data, "max_new_tokens", 4096, 1),
    )


def _parse_faster_whisper(data: dict) -> FasterWhisperConfig:
    compute_type = _get_str(data, "compute_type", "float16")
    return FasterWhisperConfig(
        model_size=_get_str(data, "model_size", "medium"),
        device=_get_str(data, "device", "cuda"),
        compute_type=compute_type,
    )


def _parse_asr(data: dict) -> ASRConfig:
    engine_str = _get_str(data, "engine", "qwen3_transformers").strip()
    try:
        engine = EngineType(engine_str)
    except ValueError:
        valid = [e.value for e in EngineType]
        raise ConfigError(f"asr.engine must be one of {valid}, got '{engine_str}'")

    return ASRConfig(
        engine=engine,
        qwen3=_parse_qwen3(data.get("qwen3", {})),
        qwen3_vllm=_parse_qwen3_vllm(data.get("qwen3_vllm", {})),
        faster_whisper=_parse_faster_whisper(data.get("faster_whisper", {})),
        language=_get_optional_str(data, "language"),
        max_inference_batch_size=_get_int(data, "max_inference_batch_size", 32, 1),
        max_new_tokens=_get_int(data, "max_new_tokens", 256, 1),
        return_timestamps=_get_bool(data, "return_timestamps", False),
        forced_aligner=_get_optional_str(data, "forced_aligner"),
    )


def _parse_audio(data: dict) -> AudioConfig:
    method = _get_str(data, "resample_method", "linear")
    if method not in _VALID_RESAMPLE_METHODS:
        raise ConfigError(f"audio.resample_method must be one of {_VALID_RESAMPLE_METHODS}, got '{method}'")
    return AudioConfig(
        sample_rate=_get_int(data, "sample_rate", 16000, 8000, 192000),
        channels=_get_int(data, "channels", 1, 1, 8),
        chunk_duration_ms=_get_int(data, "chunk_duration_ms", 32, 10, 500),
        loopback_device=_get_optional_str(data, "loopback_device"),
        resample_method=method,
    )


def _parse_vad(data: dict) -> VADConfig:
    return VADConfig(
        enabled=_get_bool(data, "enabled", True),
        threshold=_get_float(data, "threshold", 0.5, 0.0, 1.0),
        min_silence_duration_ms=_get_int(data, "min_silence_duration_ms", 400, 50, 5000),
        min_speech_duration_ms=_get_int(data, "min_speech_duration_ms", 300, 50, 5000),
        speech_pad_ms=_get_int(data, "speech_pad_ms", 100, 0, 2000),
        max_segment_duration_s=_get_int(data, "max_segment_duration_s", 15, 1, 300),
        model=_get_str(data, "model", "silero_vad"),
        model_path=_get_str(data, "model_path", "./models/silero_vad.jit"),
    )


def _parse_streaming(data: dict) -> StreamingConfig:
    return StreamingConfig(
        result_interval_ms=_get_int(data, "result_interval_ms", 200, 10, 5000),
    )


def _parse_output(data: dict) -> OutputConfig:
    console_data = data.get("console", {})
    file_data = data.get("file", {})

    streaming_mode = _get_str(console_data, "streaming_mode", "overwrite")
    if streaming_mode not in _VALID_STREAMING_MODES:
        raise ConfigError(f"output.console.streaming_mode must be one of {_VALID_STREAMING_MODES}, got '{streaming_mode}'")

    return OutputConfig(
        console=OutputConsoleConfig(
            enabled=_get_bool(console_data, "enabled", True),
            show_timestamp=_get_bool(console_data, "show_timestamp", True),
            streaming_mode=streaming_mode,
        ),
        file=OutputFileConfig(
            enabled=_get_bool(file_data, "enabled", True),
            output_dir=_get_str(file_data, "output_dir", "./output"),
            filename_pattern=_get_str(file_data, "filename_pattern", "session_{timestamp}.txt"),
            flush_interval_s=_get_int(file_data, "flush_interval_s", 5, 1, 300),
        ),
    )


def _parse_logging(data: dict) -> LoggingConfig:
    level = _get_str(data, "level", "INFO").upper()
    if level not in _VALID_LEVELS:
        raise ConfigError(f"logging.level must be one of {_VALID_LEVELS}, got '{level}'")
    return LoggingConfig(
        level=level,
        file=_get_optional_str(data, "file"),
    )


def load_config(config_path: str | Path) -> LiveScribeConfig:
    """Load and validate a LiveScribe YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Validated LiveScribeConfig instance.

    Raises:
        ConfigError: If the file is missing, unparseable, or contains invalid values.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML: {e}") from e

    if raw is None:
        raw = {}

    mode_str = _get_str(raw, "mode", "vad_segmented")
    try:
        mode = RunMode(mode_str)
    except ValueError:
        valid = [m.value for m in RunMode]
        raise ConfigError(f"mode must be one of {valid}, got '{mode_str}'")

    return LiveScribeConfig(
        mode=mode,
        asr=_parse_asr(raw.get("asr", {})),
        audio=_parse_audio(raw.get("audio", {})),
        vad=_parse_vad(raw.get("vad", {})),
        streaming=_parse_streaming(raw.get("streaming", {})),
        output=_parse_output(raw.get("output", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )
