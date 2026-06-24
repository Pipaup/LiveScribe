"""Abstract base class for all ASR engines.

Defines the unified interface that every engine must implement.
Pipeline depends only on this interface — never on concrete engine types.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseASREngine(ABC):
    """Unified interface for all ASR engines.

    Engines fall into two categories, distinguished by supports_streaming:

    Segmented engines (supports_streaming=False):
        - Only implement transcribe()
        - Pipeline runs VAD → accumulate audio → transcribe whole segment

    Streaming engines (supports_streaming=True):
        - Additionally implement start_stream / feed_chunk / end_stream
        - Pipeline feeds frame-by-frame → incremental output
        - transcribe() should still work (for fallback or testing)
    """

    # ============ All engines must implement ============

    @abstractmethod
    def initialize(self) -> None:
        """Load model, allocate VRAM."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Audio → text (core method for segmented mode).

        Args:
            audio: float32 array, shape=(num_samples,), range [-1.0, 1.0].
            sample_rate: Sample rate in Hz, default 16000.

        Returns:
            Recognized text string.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release VRAM, free resources."""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this engine supports native streaming inference."""
        ...

    @property
    @abstractmethod
    def supports_timestamps(self) -> bool:
        """Whether this engine can return word-level timestamps."""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine identifier for logging."""
        ...

    # ============ Streaming engines override these ============

    def start_stream(self) -> None:
        """Initialize a streaming session. Streaming engines must override."""

    def feed_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """Feed one audio frame, return incremental text (may be None).

        Args:
            chunk: float32 array, shape=(512,), range [-1.0, 1.0].

        Returns:
            Newly recognized text since last call, or None if no new result.
        """

    def end_stream(self) -> str:
        """End streaming session, flush buffers, return final result."""

    # ============ Optional ============

    def estimate_vram_usage(self) -> Optional[int]:
        """Estimate VRAM usage in MB for the current config. Optional."""
        return None
