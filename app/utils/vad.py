# app/utils/vad.py
import numpy as np

def _energy_trim(audio, sr, frame_ms=30):
    frame = int(sr * frame_ms / 1000)
    if frame <= 0:
        return audio

    n = int(np.ceil(len(audio) / frame))
    padded = np.pad(audio, (0, n * frame - len(audio)))

    frames = padded.reshape(n, frame)
    energy = frames.mean(axis=1)

    thresh = max(1e-6, energy.mean() * 0.5)
    mask = energy > thresh

    if not mask.any():
        return audio

    start = np.argmax(mask) * frame
    end = (len(mask) - np.argmax(mask[::-1])) * frame

    return audio[start:end]


def apply_vad(audio, sr=16000, use_silero=False):
    return _energy_trim(audio, sr)
