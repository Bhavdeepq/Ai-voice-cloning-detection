"""Audio preparation for AASIST-L V3 inference."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000


def load_mono_16khz(audio_path: str | Path) -> np.ndarray:
    """Load an audio file, mix channels to mono, and resample to 16 kHz."""
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    waveform = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(waveform, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def split_into_windows(waveform: np.ndarray, window_samples: int) -> list[np.ndarray]:
    """Split audio into fixed AASIST windows; repeat-pad only a final short window."""
    if waveform.size == 0:
        raise ValueError("Cannot create inference windows from empty audio.")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")
    windows: list[np.ndarray] = []
    for start in range(0, waveform.size, window_samples):
        window = waveform[start : start + window_samples]
        if window.size < window_samples:
            window = np.tile(window, (window_samples + window.size - 1) // window.size)[:window_samples]
        windows.append(np.ascontiguousarray(window, dtype=np.float32))
    return windows
