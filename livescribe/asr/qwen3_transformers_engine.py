"""Qwen3-ASR engine via transformers backend (offline / VAD-segmented mode).

Model: Qwen3-ASR-0.6B (~1.8 GB VRAM) or 1.7B (~4 GB VRAM).
"""

import logging
from typing import Optional

import numpy as np
import torch

from livescribe.asr.base import BaseASREngine
from livescribe.config.schema import ASRConfig
from livescribe.exceptions import ASREngineError

logger = logging.getLogger(__name__)


class Qwen3TransformersEngine(BaseASREngine):
    """Qwen3-ASR via Qwen3ASRModel.from_pretrained()."""

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_timestamps(self) -> bool:
        return self._config.forced_aligner is not None

    @property
    def engine_name(self) -> str:
        return "qwen3_transformers"

    def initialize(self) -> None:
        logger.info(
            "Loading Qwen3-ASR model: %s (device=%s, dtype=%s)",
            self._config.qwen3.model_name,
            self._config.qwen3.device,
            self._config.qwen3.dtype,
        )

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ASREngineError(
                "qwen-asr package not installed. Run: pip install qwen-asr"
            )

        dtype = getattr(torch, self._config.qwen3.dtype)

        init_kwargs: dict = {
            "dtype": dtype,
            "device_map": self._config.qwen3.device,
            "max_inference_batch_size": self._config.max_inference_batch_size,
            "max_new_tokens": self._config.max_new_tokens,
        }

        if self._config.forced_aligner:
            init_kwargs["forced_aligner"] = self._config.forced_aligner
            init_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=self._config.qwen3.device,
            )

        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self._config.qwen3.model_name,
                **init_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            raise ASREngineError(
                "GPU out of memory loading Qwen3-ASR model. "
                "Try setting asr.qwen3.device to 'cpu' or switching to faster_whisper engine."
            )
        except FileNotFoundError:
            raise ASREngineError(
                f"Model not found: {self._config.qwen3.model_name}. "
                "Download it or check the path in asr.qwen3.model_name."
            )

        logger.info("Qwen3-ASR model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            raise ASREngineError("Engine not initialized. Call initialize() first.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            results = self._model.transcribe(
                audio=(audio, sample_rate),
                language=self._config.language,
                return_time_stamps=self._config.return_timestamps,
            )
        except Exception as e:
            raise ASREngineError(f"Transcription failed: {e}") from e

        if not results:
            logger.warning("Transcription returned empty result")
            return ""

        text = results[0].text
        if text is None:
            return ""
        return text.strip()

    def shutdown(self) -> None:
        if self._model is not None:
            logger.info("Shutting down Qwen3-ASR engine")
            try:
                self._model = self._model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def estimate_vram_usage(self) -> Optional[int]:
        # 0.6B ~1.8 GB, 1.7B ~4 GB
        if "0.6B" in self._config.qwen3.model_name or "0.6b" in self._config.qwen3.model_name:
            return 1800
        elif "1.7B" in self._config.qwen3.model_name or "1.7b" in self._config.qwen3.model_name:
            return 4000
        return None
