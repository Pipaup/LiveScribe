"""Centralized logging setup for LiveScribe."""

import logging
import logging.handlers
from pathlib import Path

from livescribe.config.schema import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure the root logger with console and optional file output.

    Args:
        config: LoggingConfig with level and optional file path.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level, logging.INFO))

    # Avoid adding duplicate handlers on re-configuration
    if root.handlers:
        for handler in root.handlers:
            handler.close()
        root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(root.level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler with rotation
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(root.level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
