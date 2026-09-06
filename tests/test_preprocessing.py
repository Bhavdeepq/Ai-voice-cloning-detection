import numpy as np

from src.preprocessing import split_into_overlapping_windows, split_into_windows


def test_short_audio_is_repeat_padded() -> None:
    windows = split_into_windows(np.array([1.0, 2.0], dtype=np.float32), 5)
    assert len(windows) == 1
    assert np.array_equal(windows[0], np.array([1, 2, 1, 2, 1], dtype=np.float32))


def test_long_audio_is_fully_segmented() -> None:
    windows = split_into_windows(np.arange(7, dtype=np.float32), 3)
    assert len(windows) == 3
    assert all(window.shape == (3,) for window in windows)


def test_overlapping_windows_are_end_aligned() -> None:
    windows = split_into_overlapping_windows(np.arange(15, dtype=np.float32), 6, 3)
    assert [window[0] for window in windows] == [0, 3, 6, 9]
    assert all(window.shape == (6,) for window in windows)
