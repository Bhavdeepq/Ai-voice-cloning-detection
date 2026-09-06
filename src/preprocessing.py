"""Audio preparation shared by AASIST-L training and inference."""

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
    if sample_rate <= 0:
        raise ValueError(f"Audio file has an invalid sample rate: {audio_path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio file contains non-finite samples: {audio_path}")
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


def split_into_overlapping_windows(
    waveform: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> list[np.ndarray]:
    """Create fixed, end-aligned windows with overlap for robust long-file inference.

    Short recordings retain the detector's repeat-padding behavior. For longer
    recordings, the last window is aligned to the end of the waveform so that
    no trailing speech is discarded. This intentionally avoids padded tails
    when a complete window is available.
    """
    if waveform.size == 0:
        raise ValueError("Cannot create inference windows from empty audio.")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")
    if hop_samples <= 0 or hop_samples > window_samples:
        raise ValueError("hop_samples must be between 1 and window_samples.")
    if waveform.size <= window_samples:
        return split_into_windows(waveform, window_samples)

    final_start = waveform.size - window_samples
    starts = list(range(0, final_start + 1, hop_samples))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        np.ascontiguousarray(waveform[start : start + window_samples], dtype=np.float32)
        for start in starts
    ]
