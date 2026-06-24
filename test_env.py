"""LiveScribe 环境集成测试

验证：PyTorch CUDA / Qwen3-ASR 加载 & 推理 / faster-whisper / VAD
"""

import os
import sys
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── PyTorch CUDA ──────────────────────────────────────────
import torch

print("=" * 50)
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    total_gb = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
else:
    print("WARNING: CUDA not available, running on CPU")

# ── Qwen3-ASR Transformers ────────────────────────────────
from qwen_asr import Qwen3ASRModel

print("=" * 50)
print("Loading Qwen3-ASR 0.6B (float16, cuda:0)...")
t0 = time.time()

model = Qwen3ASRModel.from_pretrained(
    "./models/Qwen3-ASR-0.6B",
    dtype=torch.float16,
    device_map="cuda:0",
    max_inference_batch_size=16,
    max_new_tokens=256,
)

load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"VRAM after load: {free_gb:.1f} GB free")

# 用一段 3 秒静音测试推理
test_audio = np.zeros(16000 * 3, dtype=np.float32)
print("Running test inference...")
t0 = time.time()

results = model.transcribe(audio=(test_audio, 16000), language="Chinese")

infer_time = time.time() - t0
print(f"Inference: {infer_time:.1f}s")
print(f"Result: '{results[0].text}'")

# 清理
del model
torch.cuda.empty_cache()

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"VRAM after cleanup: {free_gb:.1f} GB free")

# ── faster-whisper ────────────────────────────────────────
from faster_whisper import WhisperModel

print("=" * 50)
print("faster-whisper: OK (import only, model not loaded)")

# ── Silero VAD ────────────────────────────────────────────
model_vad, utils_vad = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", trust_repo=True
)
print("Silero VAD: OK")

# ── soundcard ─────────────────────────────────────────────
import soundcard

speakers = soundcard.all_speakers()
print(f"Audio devices: {len(speakers)} speaker(s)")
for s in speakers:
    print(f"  {s.name}")

# ── 总结 ──────────────────────────────────────────────────
print("=" * 50)
print("All checks passed!")
print(f"  PyTorch:           {torch.__version__}")
print(f"  CUDA:              OK ({torch.cuda.get_device_name(0)})")
print(f"  Qwen3-ASR 0.6B:    OK (load {load_time:.1f}s, infer {infer_time:.1f}s)")
print(f"  faster-whisper:    OK")
print(f"  Silero VAD:        OK")
print(f"  soundcard:         OK ({len(speakers)} devices)")
