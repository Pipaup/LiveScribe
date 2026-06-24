"""WASAPI loopback device enumeration and selection."""

from __future__ import annotations

import logging
from typing import Optional

import soundcard as sc

logger = logging.getLogger(__name__)


def list_loopback_devices() -> list[dict]:
    """Enumerate all available loopback input devices.

    Returns:
        List of dicts with keys: name, id, channels, default_samplerate.
    """
    devices = []
    for mic in sc.all_microphones(include_loopback=True):
        devices.append({
            "name": mic.name,
            "id": mic.id,
            "channels": mic.channels,
        })
    return devices


def get_default_loopback_device() -> sc.Microphone:
    """Get the default system speaker's loopback microphone.

    Returns:
        A soundcard Microphone for loopback capture.

    Raises:
        RuntimeError: If no loopback device is found.
    """
    try:
        speaker = sc.default_speaker()
    except Exception:
        raise RuntimeError("No default speaker found. Is the audio driver installed?")

    speaker_name = speaker.name
    logger.info("Default speaker: %s", speaker_name)

    for mic in sc.all_microphones(include_loopback=True):
        if speaker_name in mic.name or mic.name in speaker_name:
            logger.info("Matched loopback device: %s", mic.name)
            return mic

    # Fallback: return first loopback device
    loopback_mics = [m for m in sc.all_microphones(include_loopback=True)
                     if "loopback" in m.name.lower()]
    if loopback_mics:
        logger.warning("No exact match for speaker, using first loopback: %s", loopback_mics[0].name)
        return loopback_mics[0]

    raise RuntimeError(
        "No loopback device found. "
        "Enable 'Stereo Mix' in Sound settings → Recording devices, "
        "or install audio drivers that support WASAPI loopback."
    )


def find_device_by_keyword(keyword: str) -> Optional[sc.Microphone]:
    """Find a loopback device whose name contains the given keyword.

    Args:
        keyword: Case-insensitive substring to match against device names.

    Returns:
        Matching Microphone, or None if no match.
    """
    keyword_lower = keyword.lower()
    for mic in sc.all_microphones(include_loopback=True):
        if keyword_lower in mic.name.lower():
            logger.info("Found device by keyword '%s': %s", keyword, mic.name)
            return mic
    return None
