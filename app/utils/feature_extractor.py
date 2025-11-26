# app/utils/feature_extractor.py
"""
ECAPA embedding extractor using ONLY local models.
NO internet, NO HuggingFace, NO torch.hub.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
from speechbrain.inference import EncoderClassifier

from .vad import apply_vad

# Force CPU
DEVICE = "cpu"

# LOCAL ECAPA MODEL DIRECTORY
MODEL_DIR = "app/models/ecapa"   # <-- MUST exist in your repo

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"❌ ECAPA model folder missing: {MODEL_DIR}")

# Load ECAPA model from local folder
ECAPA = EncoderClassifier.from_hparams(
    source=MODEL_DIR,
    savedir=MODEL_DIR,
    run_opts={"device": DEVICE},
)

# Minimum audio for ECAPA
MIN_SAMPLES = 16000


def _load_audio(path: str, target_sr: int = 16000):
    """Load audio file & resample."""
    data, sr = sf.read(path, dtype="float32")

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr


def get_ecapa_embedding_from_file(path: str):
    """Full embedding extractor pipeline."""

    # 1. LOAD AUDIO
    waveform, sr = _load_audio(path)
    if len(waveform) < 1:
        raise RuntimeError("❌ Empty audio")

    # 2. VAD (local only)
    waveform_vad = apply_vad(waveform, sr)

    if len(waveform_vad) < 160:
        waveform_vad = waveform  # fallback to raw audio

    # 3. PAD
    if len(waveform_vad) < MIN_SAMPLES:
        pad = MIN_SAMPLES - len(waveform_vad)
        waveform_vad = np.pad(waveform_vad, (0, pad), mode="reflect")

    # 4. TENSOR
    tensor = torch.tensor(waveform_vad, dtype=torch.float32).unsqueeze(0)

    # 5. ECAPA FORWARD
    with torch.no_grad():
        emb_tensor = ECAPA.encode_batch(tensor)

    # 6. NUMPY
    emb = emb_tensor.squeeze().cpu().numpy().astype(np.float32)

    # 7. FIX SHAPE
    if emb.ndim != 1:
        emb = emb.flatten()

    if emb.shape[0] != 192:
        raise RuntimeError(f"❌ Embedding wrong shape: {emb.shape}")

    # 8. NORMALIZE
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    return emb