#!/usr/bin/env python3
"""LiveScribe — Real-time Windows speech-to-text system.

Captures system audio (WASAPI loopback) and transcribes it using
a configurable ASR engine.
"""

import argparse
import os
import signal
import sys
import warnings
from pathlib import Path

# Resolve OpenMP DLL conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress soundcard data-discontinuity warnings (normal on loaded systems)
warnings.filterwarnings("ignore", message="data discontinuity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiveScribe — Real-time speech-to-text from system audio"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in current directory)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available loopback devices and exit",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available ASR engines and exit",
    )
    parser.add_argument(
        "--download-vad",
        action="store_true",
        help="Download Silero VAD model (.jit) to the path in config.yaml and exit",
    )
    args = parser.parse_args()

    # Add project root to path so livescribe package is importable
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from livescribe.config.loader import load_config
    from livescribe.config.schema import EngineType
    from livescribe.exceptions import LiveScribeError
    from livescribe.logging_utils import setup_logging

    # --- Subcommands ---

    if args.list_devices:
        from livescribe.audio.device import list_loopback_devices
        devices = list_loopback_devices()
        print(f"\nAvailable loopback devices ({len(devices)}):\n")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d['name']}")
            print(f"      id={d['id']}, channels={d['channels']}")
        return

    if args.list_engines:
        print("\nAvailable ASR engines:\n")
        for e in EngineType:
            print(f"  - {e.value}")
        print("\nSet asr.engine in config.yaml to switch.\n")
        return

    if args.download_vad:
        from livescribe.vad.download import download_vad_model
        download_vad_model(args.config)
        return

    # --- Normal run ---

    try:
        config = load_config(args.config)
    except LiveScribeError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    setup_logging(config.logging)

    import logging
    logger = logging.getLogger(__name__)

    from livescribe.pipeline import Pipeline

    pipeline = Pipeline(config)

    # Handle Ctrl+C gracefully
    def _signal_handler(signum, frame):
        logger.info("Received signal %s, stopping pipeline", signum)
        pipeline.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("LiveScribe v%s starting", __import__("livescribe").__version__)
    logger.info("Press Ctrl+C to stop")

    try:
        pipeline.run()
    except LiveScribeError as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
