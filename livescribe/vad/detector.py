"""Voice Activity Detection via Silero VAD.

Loads the Silero VAD TorchScript model from a local .jit file and implements
the VADIterator state machine inline, avoiding the torch.hub / GitHub dependency.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from livescribe.config.schema import VADConfig
from livescribe.exceptions import LiveScribeError

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FRAME_SIZE = 512  # samples per frame at 16kHz = 32ms


class _VADIterator:
    """Minimal VADIterator implementing the Silero speech boundary state machine.

    Equivalent to the upstream silero-vad VADIterator but self-contained so we
    don't depend on torch.hub for utils.
    """

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._sampling_rate = sampling_rate

        self._min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self._speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)

        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0

    def reset_states(self) -> None:
        self._model.reset_states()
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0

    def __call__(self, chunk: np.ndarray, return_seconds: bool = False) -> dict:
        """Process one audio frame and return {'start': ts} or {'end': ts} or {}."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        t = torch.from_numpy(chunk)
        window_size_samples = t.shape[0] if t.dim() == 1 else t.shape[-1]
        self._current_sample += window_size_samples

        speech_prob = self._model(t, self._sampling_rate).item()

        # Speech start detection
        if speech_prob >= self._threshold:
            self._temp_end = 0
            if not self._triggered:
                self._triggered = True
                speech_start = max(
                    self._current_sample - self._speech_pad_samples - window_size_samples, 0
                )
                ts = speech_start if not return_seconds else round(speech_start / self._sampling_rate, 1)
                return {"start": ts}

        # Speech end detection (hysteresis: threshold - 0.15)
        if speech_prob < self._threshold - 0.15 and self._triggered:
            if self._temp_end == 0:
                self._temp_end = self._current_sample
            if self._current_sample - self._temp_end >= self._min_silence_samples:
                speech_end = self._temp_end + self._speech_pad_samples - window_size_samples
                ts = speech_end if not return_seconds else round(speech_end / self._sampling_rate, 1)
                self._triggered = False
                self._temp_end = 0
                return {"end": ts}

        # False alarm check
        if speech_prob >= self._threshold - 0.15 and self._triggered:
            self._temp_end = 0

        return {}


class VADDetector:
    """Silero VAD wrapper with local model loading and state-machine interface.

    Loads the model from a local .jit file (configurable via vad.model_path).
    Falls back to torch.hub only when the local file is missing and the user
    confirms they can reach GitHub.
    """

    def __init__(self, config: VADConfig) -> None:
        self._config = config
        self._model = None
        self._iterator: Optional[_VADIterator] = None
        self._initialize()

    def _initialize(self) -> None:
        logger.info(
            "Loading Silero VAD model (threshold=%.2f, min_silence=%dms, pad=%dms)",
            self._config.threshold,
            self._config.min_silence_duration_ms,
            self._config.speech_pad_ms,
        )

        model_path = Path(self._config.model_path)

        if model_path.exists():
            self._model = self._load_local(model_path)
        else:
            self._model = self._load_via_hub()

        self._iterator = _VADIterator(
            self._model,
            threshold=self._config.threshold,
            sampling_rate=_SAMPLE_RATE,
            min_silence_duration_ms=self._config.min_silence_duration_ms,
            speech_pad_ms=self._config.speech_pad_ms,
        )

        logger.info("Silero VAD model loaded successfully")

    def _load_local(self, path: Path) -> torch.jit.ScriptModule:
        logger.info("Loading VAD from local file: %s", path)
        try:
            model = torch.jit.load(str(path))
            model.eval()
            return model
        except Exception as e:
            raise LiveScribeError(
                f"Failed to load VAD model from {path}: {e}"
            ) from e

    def _load_via_hub(self) -> torch.jit.ScriptModule:
        logger.info("Local VAD model not found, trying torch.hub (GitHub)...")
        logger.info(
            "If this fails, download silero_vad.jit to %s",
            self._config.model_path,
        )
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            return model
        except Exception as e:
            raise LiveScribeError(
                f"Failed to download Silero VAD model from GitHub.\n"
                f"  Reason: {e}\n"
                f"  Quick fix:  python main.py --download-vad"
            ) from e

    def process(self, chunk: np.ndarray) -> Optional[Literal["start", "end"]]:
        """Feed one audio frame to the VAD state machine.

        Args:
            chunk: float32 array, shape=(512,), range [-1.0, 1.0].

        Returns:
            "start" when speech begins, "end" when speech ends, or None.
        """
        if self._iterator is None:
            raise LiveScribeError("VAD not initialized")

        if chunk.shape != (_FRAME_SIZE,):
            raise ValueError(
                f"VAD expects chunk shape ({_FRAME_SIZE},), got {chunk.shape}"
            )

        speech_dict = self._iterator(chunk, return_seconds=False)

        if not speech_dict:
            return None

        if "start" in speech_dict:
            return "start"
        if "end" in speech_dict:
            return "end"

        return None

    def reset(self) -> None:
        """Reset the VAD LSTM hidden state after each speech segment ends."""
        if self._iterator is not None:
            self._iterator.reset_states()
            logger.debug("VAD state reset")

    def close(self) -> None:
        """Release the VAD model."""
        if self._model is not None:
            logger.info("Releasing Silero VAD model")
            del self._model
            self._model = None
            self._iterator = None

    @property
    def threshold(self) -> float:
        return self._config.threshold
