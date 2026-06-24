"""Audio capture via WASAPI loopback.

Captures system audio (what you hear) and outputs uniform float32 frames
at the target sample rate (default 16kHz).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import soundcard as sc

from livescribe.audio.device import find_device_by_keyword, get_default_loopback_device
from livescribe.audio.resampler import resample, to_mono
from livescribe.config.schema import AudioConfig
from livescribe.exceptions import AudioDeviceError

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures system audio via WASAPI loopback and outputs uniform float32 chunks.

    Requests the recorder at the target sample rate directly. If the device
    delivers a different rate, a resampling step is applied.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._mic: Optional[sc.Microphone] = None
        self._recorder = None  # RecorderContext, type is private
        self._device_channels: int = 0
        self._actual_rate: int = 0

        # Samples per output chunk at target rate
        self._chunk_samples = int(config.sample_rate * config.chunk_duration_ms / 1000)

        # Consecutive silent frame counter
        self._empty_count = 0

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def actual_rate(self) -> int:
        """The actual sample rate delivered by the device."""
        return self._actual_rate or self._config.sample_rate

    def open(self) -> None:
        """Open the loopback device and start recording."""
        if self._config.loopback_device:
            self._mic = find_device_by_keyword(self._config.loopback_device)
            if self._mic is None:
                available = [d["name"] for d in self._list_devices()]
                raise AudioDeviceError(
                    f"Loopback device '{self._config.loopback_device}' not found. "
                    f"Available devices: {available}"
                )
        else:
            self._mic = get_default_loopback_device()

        self._device_channels = self._mic.channels
        self._actual_rate = self._config.sample_rate

        logger.info(
            "Opening loopback device: name=%s, target_rate=%d Hz, channels=%d",
            self._mic.name,
            self._actual_rate,
            self._device_channels,
        )

        try:
            self._recorder = self._mic.recorder(
                samplerate=self._actual_rate,
                channels=self._device_channels,
            )
            self._recorder.__enter__()
        except Exception as e:
            raise AudioDeviceError(f"Failed to open loopback device: {e}") from e

        self._empty_count = 0

    def read_chunk(self) -> np.ndarray:
        """Read one chunk of audio, normalized to target format.

        Returns:
            float32 array, shape=(chunk_samples,), range [-1.0, 1.0].

        Raises:
            AudioDeviceError: If the device is not open or encounters an error.
        """
        if self._recorder is None:
            raise AudioDeviceError("Device not open. Call open() first.")

        # Number of frames to request from the device
        num_frames = int(self._actual_rate * self._config.chunk_duration_ms / 1000)

        try:
            raw = self._recorder.record(numframes=num_frames)
        except Exception as e:
            raise AudioDeviceError(f"Failed to read audio frame: {e}") from e

        # raw shape: (num_frames, channels), float32, [-1, 1]

        # Convert to mono
        audio = to_mono(raw, self._device_channels)

        # Trim or pad to exact chunk size
        if len(audio) > self._chunk_samples:
            audio = audio[:self._chunk_samples]
        elif len(audio) < self._chunk_samples:
            audio = np.pad(audio, (0, self._chunk_samples - len(audio)))

        # Detect silent/empty condition
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-6:
            self._empty_count += 1
            if self._empty_count >= 100:
                logger.debug(
                    "100 consecutive silent frames — possible device mute/disconnect"
                )
                self._empty_count = 0
        else:
            self._empty_count = 0

        return audio.astype(np.float32)

    def close(self) -> None:
        """Stop recording and release the device."""
        if self._recorder is not None:
            try:
                self._recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._recorder = None

        self._mic = None
        logger.info("Audio capture closed")

    @staticmethod
    def _list_devices() -> list[dict]:
        from livescribe.audio.device import list_loopback_devices
        return list_loopback_devices()
