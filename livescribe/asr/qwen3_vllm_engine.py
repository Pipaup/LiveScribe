"""Qwen3-ASR engine via vLLM backend (high-throughput batch + streaming).

Note: vLLM on Windows has limited support. This engine requires the vLLM
backend to be installed: pip install qwen-asr[vllm]
"""

import logging
from typing import Optional

import numpy as np
import torch

from livescribe.asr.base import BaseASREngine
from livescribe.config.schema import ASRConfig
from livescribe.exceptions import ASREngineError

logger = logging.getLogger(__name__)


class Qwen3VllmEngine(BaseASREngine):
    """Qwen3-ASR via Qwen3ASRModel.LLM() (vLLM backend)."""

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None

    @property
    def supports_streaming(self) -> bool:
        # vLLM backend supports streaming via WebSocket, but the simple
        # chunk-by-chunk streaming path requires extra wiring.
        # For now, this engine operates in segmented mode via transcribe().
        return False

    @property
    def supports_timestamps(self) -> bool:
        return self._config.forced_aligner is not None

    @property
    def engine_name(self) -> str:
        return "qwen3_vllm"

    def initialize(self) -> None:
        logger.info(
            "Loading Qwen3-ASR vLLM: %s (gpu_mem=%.0f%%)",
            self._config.qwen3.model_name,
            self._config.qwen3_vllm.gpu_memory_utilization * 100,
        )

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ASREngineError(
                "qwen-asr package not installed. Run: pip install qwen-asr[vllm]"
            )

        init_kwargs: dict = {
            "model": self._config.qwen3.model_name,
            "gpu_memory_utilization": self._config.qwen3_vllm.gpu_memory_utilization,
            "max_inference_batch_size": self._config.qwen3_vllm.max_inference_batch_size,
            "max_new_tokens": self._config.qwen3_vllm.max_new_tokens,
        }

        if self._config.forced_aligner:
            dtype = getattr(torch, self._config.qwen3.dtype)
            init_kwargs["forced_aligner"] = self._config.forced_aligner
            init_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=self._config.qwen3.device,
            )

        try:
            self._model = Qwen3ASRModel.LLM(**init_kwargs)
        except torch.cuda.OutOfMemoryError:
            raise ASREngineError(
                "GPU out of memory loading Qwen3-ASR vLLM. "
                "Try reducing gpu_memory_utilization or switching to qwen3_transformers engine."
            )
        except Exception as e:
            raise ASREngineError(f"Failed to initialize vLLM backend: {e}") from e

        logger.info("Qwen3-ASR vLLM engine loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            raise ASREngineError("Engine not initialized. Call initialize() first.")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            results = self._model.transcribe(
                audio=[(audio, sample_rate)],
                language=[self._config.language] if self._config.language else None,
                return_time_stamps=self._config.return_timestamps,
            )
        except Exception as e:
            raise ASREngineError(f"vLLM transcription failed: {e}") from e

        if not results:
            logger.warning("Transcription returned empty result")
            return ""

        text = results[0].text
        if text is None:
            return ""
        return text.strip()

    def shutdown(self) -> None:
        if self._model is not None:
            logger.info("Shutting down Qwen3-ASR vLLM engine")
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def estimate_vram_usage(self) -> Optional[int]:
        if "0.6B" in self._config.qwen3.model_name or "0.6b" in self._config.qwen3.model_name:
            return 2500  # vLLM has extra overhead
        elif "1.7B" in self._config.qwen3.model_name or "1.7b" in self._config.qwen3.model_name:
            return 5000
        return None
