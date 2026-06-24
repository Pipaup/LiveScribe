"""LiveScribe custom exception hierarchy."""


class LiveScribeError(Exception):
    """Base exception for all LiveScribe errors."""


class ConfigError(LiveScribeError):
    """Configuration validation or loading failure."""


class AudioDeviceError(LiveScribeError):
    """Audio device disconnected, unavailable, or producing invalid data."""


class ASREngineError(LiveScribeError):
    """ASR model loading failure or inference error."""


class PipelineError(LiveScribeError):
    """Pipeline runtime error (e.g., queue overflow, thread crash)."""
