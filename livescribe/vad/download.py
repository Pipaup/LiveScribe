"""Download Silero VAD model from accessible mirrors.

Tries multiple sources in order:
  1. HF Mirror (hf-mirror.com) — accessible inside China
  2. HuggingFace (huggingface.co)
  3. GitHub (raw.githubusercontent.com, blocked in China but kept as fallback)
"""

import logging
import sys
from pathlib import Path
from urllib.request import urlopen

from livescribe.config.loader import load_config
from livescribe.exceptions import LiveScribeError

logger = logging.getLogger(__name__)

_SOURCES = [
    # (label, url)
    ("HF Mirror", "https://hf-mirror.com/snakers4/silero-vad/resolve/main/src/silero_vad/data/silero_vad.jit"),
    ("HuggingFace", "https://huggingface.co/snakers4/silero-vad/resolve/main/src/silero_vad/data/silero_vad.jit"),
    ("GitHub", "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit"),
]

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_vad_model(config_path: str) -> None:
    """Download silero_vad.jit to the path configured in config.yaml."""
    try:
        config = load_config(config_path)
    except LiveScribeError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    target = Path(config.vad.model_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"VAD model already exists: {target}")
        print(f"  Size: {target.stat().st_size / 1024:.1f} KB")
        print("  Delete it first if you want to re-download.")
        return

    print(f"Downloading Silero VAD model → {target}")
    print(f"  File size: ~2.5 MB\n")

    for label, url in _SOURCES:
        print(f"Trying {label}...")
        try:
            with urlopen(url, timeout=30) as resp:
                if resp.status != 200:
                    print(f"  HTTP {resp.status}, skipping")
                    continue

                content = resp.read()
                target.write_bytes(content)
                print(f"  Done. Saved {len(content) / 1024:.1f} KB to {target}")
                return
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("\nAll sources failed. Please download the file manually:")
    print(f"  1. Visit: {_SOURCES[0][1]}")
    print(f"  2. Save as: {target.resolve()}")
    print(f"  3. Then re-run LiveScribe normally.")
    sys.exit(1)
