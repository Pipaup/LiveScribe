# LiveScribe — Real-time Speech-to-Text System

[中文](README.md) | English

A Python-based real-time speech-to-text system that captures system audio via Windows WASAPI Loopback and transcribes it using switchable ASR engines with VAD-based voice segmentation.

> **TL;DR: Anything playing on your PC gets transcribed in real-time. No microphone needed.**

---

## ✨ Features

- **System Audio Capture**: Captures speaker output via WASAPI Loopback — no microphone required. Works with any audio source: videos, podcasts, online meetings, music.
- **Switchable ASR Engines**: Toggle engines with one line in `config.yaml` — no code changes needed.
- **Three Engines Supported**:
  | Engine | Highlights | VRAM | Best For |
  |--------|-----------|------|----------|
  | **faster-whisper** | CTranslate2 backend, CPU-friendly | ~500MB | General purpose, best balance |
  | **Qwen3-ASR 0.6B** | Top Chinese accuracy, 52 languages | ~1.8GB | Chinese/multilingual, high quality |
  | **Qwen3-ASR 1.7B** | SOTA open-source ASR model | ~4GB | Maximum accuracy |
- **VAD Voice Detection**: Silero VAD detects speech boundaries with configurable threshold, silence duration, and padding
- **Dual-Thread Architecture**: Audio capture and ASR inference run on separate threads — zero frame loss
- **Multi-Backend Output**: Simultaneous console output (live refresh) and file output (periodic flush)
- **Fully Offline**: Local model paths supported — no internet needed after model download

---

## 📂 Architecture

```
LiveScribe/
├── livescribe/                     # Main Python package
│   ├── asr/                        # ASR engine layer (Strategy pattern)
│   │   ├── base.py                 #   Abstract engine interface
│   │   ├── factory.py              #   Factory: create engine from config
│   │   ├── faster_whisper_engine.py
│   │   ├── qwen3_transformers_engine.py
│   │   └── qwen3_vllm_engine.py
│   ├── audio/                      # Audio capture layer
│   │   ├── capture.py              #   WASAPI Loopback capture
│   │   ├── device.py               #   Device enumeration & selection
│   │   └── resampler.py            #   Resampling utilities
│   ├── vad/                        # Voice Activity Detection layer
│   │   ├── detector.py             #   Silero VAD wrapper + state machine
│   │   └── download.py             #   VAD model downloader
│   ├── output/                     # Output layer
│   │   ├── console.py              #   Console real-time output
│   │   ├── file_writer.py          #   File writer with periodic flush
│   │   └── output_manager.py       #   Multi-backend manager
│   ├── config/                     # Configuration module
│   │   ├── schema.py               #   Dataclasses + enums
│   │   └── loader.py               #   YAML loader + validation
│   ├── pipeline.py                 # Main pipeline (dual-threaded)
│   ├── exceptions.py               # Custom exception hierarchy
│   ├── logging_utils.py            # Logging setup
│   └── __main__.py                 # `python -m livescribe` entry
├── config.example.yaml             # Configuration template
├── main.py                         # Program entry point
├── requirements.txt                # Python dependencies
├── run.bat                         # Windows one-click launcher
├── run-download-vad.bat            # Download VAD model
├── run-list-devices.bat            # List audio devices
└── setup-env.bat                   # Environment setup script
```

**Data Flow (VAD-Segmented Mode)**:

```
System Audio → WASAPI Loopback → [AudioCapture: 48kHz→16kHz, stereo→mono]
                                        │
                                 32ms float32 frames
                                        ▼
                          [Main Thread: VAD State Machine]
                                        │
                            Speech segment detected
                                        ▼
                            audio_queue (Thread-safe)
                                        ▼
                          [ASR Thread: engine.transcribe()]
                                        │
                            result_queue (Thread-safe)
                                        ▼
                     [Main Thread: Console + File Output]
```

---

## 🚀 Quick Start

### Prerequisites

- Windows 10/11
- Python 3.11+
- NVIDIA GPU (optional but recommended, RTX 3060 6GB+)
- CUDA 12.4+

### 1. Install Dependencies

```bash
# Using conda (recommended)
conda create -n livescribe python=3.11 -y
conda activate livescribe

# Install PyTorch (GPU version)
pip install torch>=2.6.0 torchaudio>=2.6.0

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

Or double-click `setup-env.bat` on Windows.

### 2. Download Models

```bash
# Download Qwen3-ASR 0.6B (optional)
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./models/Qwen3-ASR-0.6B

# Download Silero VAD model (required)
python main.py --download-vad
```

### 3. Configure

```bash
# Copy template
copy config.example.yaml config.yaml

# Edit config.yaml to select your engine:
#   asr.engine: "faster_whisper"  or  "qwen3_transformers"
```

### 4. Run

```bash
# Start transcription
python main.py

# Or double-click run.bat

# List available audio devices
python main.py --list-devices
```

### 5. Usage

1. Play any audio on your PC (video, podcast, meeting)
2. Watch the transcription appear in real-time
3. Press `Ctrl+C` to stop

---

## ⚙️ Configuration

All settings in `config.yaml` (see `config.example.yaml` for the full template):

```yaml
mode: "vad_segmented"                     # Operating mode

asr:
  engine: "faster_whisper"                # Switch engine here (one line!)
  
  faster_whisper:
    model_size: "large-v3-turbo"          # Recommended for best quality/speed
    device: "cuda"
    
  qwen3:
    model_name: "./models/Qwen3-ASR-0.6B"  # Local path or HuggingFace ID
    device: "cuda"
    dtype: "bfloat16"

audio:
  chunk_duration_ms: 32                   # Frame duration
  loopback_device: null                   # null = auto-detect default speaker

vad:
  enabled: true
  threshold: 0.5                          # Speech probability threshold
  min_silence_duration_ms: 300            # Silence duration before cutting

output:
  console:
    enabled: true                         # Terminal output
  file:
    enabled: true                         # File output
```

---

## 🔧 Engine Switching

Change one line in `config.yaml`:

```yaml
asr:
  engine: "faster_whisper"    # ← Change this
```

| Value | Engine | Best For |
|-------|--------|----------|
| `faster_whisper` | faster-whisper (CTranslate2) | General purpose, good for both English & Chinese |
| `qwen3_transformers` | Qwen3-ASR Transformers | Best Chinese/multilingual accuracy |
| `qwen3_vllm` | Qwen3-ASR vLLM | Linux streaming mode (not available on Windows) |

---

## 🔌 Adding a New Engine

Only 3 steps — no changes to Pipeline, Audio, VAD, or Output:

1. Create `livescribe/asr/xxx_engine.py` extending `BaseASREngine`
2. Add the enum value in `livescribe/config/schema.py`
3. Add an `elif` branch in `livescribe/asr/factory.py`

---

## 📝 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check .
ruff format --check .

# Type check
mypy livescribe/
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — State-of-the-art multilingual ASR model family
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-accelerated Whisper
- [Silero VAD](https://github.com/snakers4/silero-vad) — Lightweight voice activity detection
- [SoundCard](https://github.com/bastibe/SoundCard) — Python WASAPI Loopback support
