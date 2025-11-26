import os
import numpy as np
import soundfile as sf
import librosa
import torch
from speechbrain.inference import EncoderClassifier
from .vad import apply_vad

DEVICE = "cpu"
MODEL_DIR = "app/models"

# Ensure model exists
required_files = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
]

for f in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, f)):
        raise RuntimeError(f"❌ Missing model file: {f} in {MODEL_DIR}")

# Load SpeechBrain model FROM LOCAL FILES
ECAPA = EncoderClassifier.from_hparams(
    source=MODEL_DIR,
    savedir=MODEL_DIR,
    run_opts={"device": DEVICE}
)

MIN_SAMPLES = 16000

def _load_audio(path, target_sr=16000):
    data, sr = sf.read(path, dtype="float32")

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr

def get_ecapa_embedding_from_file(path: str):
    waveform, sr = _load_audio(path)

    if len(waveform) < 1:
        raise RuntimeError("❌ Empty audio file")

    # VAD (local-only)
    vad_audio = apply_vad(waveform, sr)

    if len(vad_audio) < 160:
        vad_audio = waveform

    if len(vad_audio) < MIN_SAMPLES:
        vad_audio = np.pad(vad_audio, (0, MIN_SAMPLES - len(vad_audio)), mode="reflect")

    tensor = torch.tensor(vad_audio, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        emb = ECAPA.encode_batch(tensor).squeeze().cpu().numpy().astype(np.float32)

    if emb.ndim != 1:
        emb = emb.flatten()

    if emb.shape[0] != 192:
        raise RuntimeError(f"❌ Invalid ECAPA embedding size: {emb.shape}")

    emb = emb / (np.linalg.norm(emb) + 1e-9)

    return emb
