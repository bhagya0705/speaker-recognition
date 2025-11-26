# app/utils/vad.py
"""
Energy-based VAD ONLY.
No Silero, no internet.
"""

import numpy as np


def apply_vad(audio: np.ndarray, sr: int = 16000, frame_ms: int = 30, thr_ratio: float = 0.5):
    """Simple energy VAD."""

    audio = np.asarray(audio, dtype=np.float32)
    frame_len = int(sr * frame_ms / 1000)

    if len(audio) < frame_len:
        return audio

    # pad to full frames
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)

    energy = frames.mean(axis=1)
    threshold = energy.mean() * thr_ratio

    mask = energy > threshold
    if not mask.any():
        return audio  # no speech found

    start = np.argmax(mask) * frame_len
    end = (len(mask) - np.argmax(mask[::-1])) * frame_len

    end = min(end, len(audio))

    return audio[start:end]