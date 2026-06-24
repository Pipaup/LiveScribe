"""faster-whisper engine (CTranslate2 backend, CPU-friendly).

Model sizes: tiny (75 MB), base (140 MB), small (480 MB), medium (1.5 GB), large-v3 (3 GB).
Recommended: medium or above for acceptable Chinese accuracy.
"""

import gc
import logging
from typing import Optional

import numpy as np

from livescribe.asr.base import BaseASREngine
from livescribe.config.schema import ASRConfig
from livescribe.exceptions import ASREngineError

logger = logging.getLogger(__name__)


class FasterWhisperEngine(BaseASREngine):
    """faster-whisper via CTranslate2 WhisperModel."""

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_timestamps(self) -> bool:
        # faster-whisper natively returns word-level timestamps
        return True

    @property
    def engine_name(self) -> str:
        return "faster_whisper"

    def initialize(self) -> None:
        fw_config = self._config.faster_whisper
        logger.info(
            "Loading faster-whisper model: size=%s, device=%s, compute=%s",
            fw_config.model_size,
            fw_config.device,
            fw_config.compute_type,
        )

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ASREngineError(
                "faster-whisper package not installed. Run: pip install faster-whisper"
            )

        try:
            self._model = WhisperModel(
                fw_config.model_size,
                device=fw_config.device,
                compute_type=fw_config.compute_type,
            )
        except Exception as e:
            raise ASREngineError(f"Failed to load faster-whisper model: {e}") from e

        logger.info("faster-whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            raise ASREngineError("Engine not initialized. Call initialize() first.")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._config.language,
                beam_size=5,
            )
        except Exception as e:
            raise ASREngineError(f"faster-whisper transcription failed: {e}") from e

        # Concatenate all segment texts
        texts = []
        for segment in segments:
            if segment.text:
                texts.append(segment.text.strip())

        result = "".join(texts)
        if not result:
            logger.warning("Transcription returned empty result (detected language: %s)", info.language)
            return result

        # Convert Traditional → Simplified Chinese when language is zh
        if info.language == "zh":
            try:
                from zhconv import convert
                result = convert(result, "zh-cn")
            except ImportError:
                pass

        return result

    def shutdown(self) -> None:
        if self._model is not None:
            logger.info("Shutting down faster-whisper engine")
            del self._model
            self._model = None
            gc.collect()

    def estimate_vram_usage(self) -> Optional[int]:
        sizes = {
            "tiny": 400,
            "base": 500,
            "small": 800,
            "medium": 1500,
            "large-v3": 3000,
        }
        return sizes.get(self._config.faster_whisper.model_size)
