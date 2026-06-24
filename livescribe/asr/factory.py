"""Factory for creating ASR engine instances from configuration.

Supports a fallback chain when the preferred engine fails to load:
  GPU OOM → suggest CPU / smaller model / different engine.
"""

import logging

from livescribe.asr.base import BaseASREngine
from livescribe.config.schema import ASRConfig, EngineType
from livescribe.exceptions import ASREngineError, ConfigError

logger = logging.getLogger(__name__)


def create_asr_engine(config: ASRConfig) -> BaseASREngine:
    """Create an ASR engine instance based on configuration.

    Args:
        config: Validated ASRConfig with engine type and parameters.

    Returns:
        An initialized BaseASREngine subclass instance.

    Raises:
        ConfigError: If the engine type is unknown.
        ASREngineError: If engine initialization fails.
    """
    engine_type = config.engine

    if engine_type == EngineType.QWEN3_TRANSFORMERS:
        from livescribe.asr.qwen3_transformers_engine import Qwen3TransformersEngine
        engine = Qwen3TransformersEngine(config)
    elif engine_type == EngineType.QWEN3_VLLM:
        from livescribe.asr.qwen3_vllm_engine import Qwen3VllmEngine
        engine = Qwen3VllmEngine(config)
    elif engine_type == EngineType.FASTER_WHISPER:
        from livescribe.asr.faster_whisper_engine import FasterWhisperEngine
        engine = FasterWhisperEngine(config)
    else:
        valid = [e.value for e in EngineType]
        raise ConfigError(f"Unknown ASR engine type: '{engine_type}'. Valid options: {valid}")

    logger.info("Creating ASR engine: %s", engine.engine_name)

    try:
        engine.initialize()
    except ASREngineError:
        raise
    except Exception as e:
        raise ASREngineError(
            f"Failed to initialize ASR engine '{engine.engine_name}': {e}"
        ) from e

    return engine
