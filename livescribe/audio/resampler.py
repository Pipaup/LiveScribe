"""Audio resampling and channel conversion utilities."""

import logging

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def resample(
    audio: np.ndarray,
    src_rate: int,
    dst_rate: int,
    method: str = "linear",
) -> np.ndarray:
    """Resample audio from source rate to destination rate.

    Args:
        audio: Input audio array, shape=(num_samples,) or (num_samples, channels).
        src_rate: Source sample rate in Hz.
        dst_rate: Destination sample rate in Hz.
        method: "linear" for fast linear interpolation, "sinc" for scipy FFT resample.

    Returns:
        Resampled audio at dst_rate.
    """
    if src_rate == dst_rate:
        return audio

    num_src = audio.shape[-1]
    num_dst = int(np.ceil(num_src * dst_rate / src_rate))

    if method == "linear":
        xs = np.linspace(0, num_src - 1, num_src)
        xd = np.linspace(0, num_src - 1, num_dst)

        if audio.ndim == 1:
            return np.interp(xd, xs, audio).astype(audio.dtype)
        else:
            # Multi-channel: interpolate each channel
            out = np.zeros((audio.shape[0], num_dst), dtype=audio.dtype)
            for ch in range(audio.shape[0]):
                out[ch] = np.interp(xd, xs, audio[ch])
            return out

    elif method == "sinc":
        if audio.ndim == 1:
            return scipy_signal.resample(audio, num_dst)
        else:
            return scipy_signal.resample(audio, num_dst, axis=-1)

    else:
        raise ValueError(f"Unknown resample method: {method}. Use 'linear' or 'sinc'.")


def to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging.

    Args:
        audio: Input array, shape=(num_frames, channels) or (channels, num_frames).
        channels: Number of interleaved channels.

    Returns:
        Mono audio, shape=(num_frames,).
    """
    if channels == 1:
        if audio.ndim == 2 and audio.shape[-1] == 1:
            return audio[:, 0]
        return audio

    # Handle (num_frames, channels) layout
    if audio.ndim == 2 and audio.shape[-1] == channels:
        return audio.mean(axis=1)
    elif audio.ndim == 2 and audio.shape[0] == channels:
        return audio.mean(axis=0)
    elif audio.ndim == 1:
        # Interleaved channels: reshape
        total = len(audio) // channels * channels
        return audio[:total].reshape(-1, channels).mean(axis=1)
    else:
        raise ValueError(f"Unexpected audio shape {audio.shape} with channels={channels}")
