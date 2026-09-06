"""Robust inference interface for the AASIST-L-SIH-v4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.AASIST import Model
from .preprocessing import load_mono_16khz, split_into_overlapping_windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "AASIST-L-SIH-v4.pth"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "finetune.conf"


class VoiceDetector:
    """Classify audio as ``REAL`` or ``AI_SPOOF`` with the fixed V4 model.

    The detector uses end-aligned 50%-overlapping windows for recordings longer
    than a model window. A short recording is still repeat-padded exactly as in
    the original inference path. The model architecture and V4 weights remain
    unchanged.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | torch.device | None = None,
        batch_size: int = 8,
        window_overlap: float = 0.5,
        fake_threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.model_path.is_file():
            raise FileNotFoundError(f"V4 checkpoint not found: {self.model_path}")
        if self.model_path.name != "AASIST-L-SIH-v4.pth":
            raise ValueError("VoiceDetector only accepts models/AASIST-L-SIH-v4.pth.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 <= window_overlap < 1:
            raise ValueError("window_overlap must be in the range [0, 1).")
        if not 0 < fake_threshold < 1:
            raise ValueError("fake_threshold must be between 0 and 1.")
        self.batch_size = batch_size
        self.window_overlap = window_overlap
        self.fake_threshold = fake_threshold

        with self.config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)["model_config"]
        self.window_samples = int(model_config["nb_samp"])
        self.window_hop_samples = max(1, round(self.window_samples * (1 - self.window_overlap)))

        self.model = Model(model_config).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @staticmethod
    def _aggregate_fake_probability(segment_probabilities: torch.Tensor) -> tuple[float, str]:
        """Aggregate window evidence while limiting the effect of isolated outliers.

        A 10% trimmed mean is used only for five or more windows. Short clips
        retain the ordinary mean so that their score stays directly comparable
        to the original one-window path.
        """
        fake_probabilities = segment_probabilities[:, 0]
        if fake_probabilities.numel() < 5:
            return float(fake_probabilities.mean()), "mean"
        trim_count = max(1, int(fake_probabilities.numel() * 0.10))
        ordered = torch.sort(fake_probabilities).values
        return float(ordered[trim_count:-trim_count].mean()), "trimmed_mean_10pct"

    def _predict_segments(self, windows: list[np.ndarray]) -> torch.Tensor:
        probabilities: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(np.stack(windows[start : start + self.batch_size])).to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                _, logits = self.model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probabilities, dim=0)

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Return a V4 prediction and stable aggregate evidence for one audio file."""
        audio = load_mono_16khz(audio_path)
        windows = split_into_overlapping_windows(
            audio,
            self.window_samples,
            self.window_hop_samples,
        )
        segment_probabilities = self._predict_segments(windows)
        fake_probability, aggregation = self._aggregate_fake_probability(segment_probabilities)
        real_probability = 1.0 - fake_probability
        is_fake = fake_probability >= self.fake_threshold
        prediction = "AI_SPOOF" if is_fake else "REAL"
        confidence = fake_probability if is_fake else real_probability

        segment_fake_probabilities = segment_probabilities[:, 0]
        segment_votes_fake = segment_fake_probabilities >= self.fake_threshold
        agreement = float((segment_votes_fake == is_fake).float().mean())
        decision_margin = abs(fake_probability - self.fake_threshold)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "segments_analyzed": len(windows),
            "decision_margin": decision_margin,
            "segment_agreement": agreement,
            "segment_fake_probability_std": float(segment_fake_probabilities.std(unbiased=False)),
            "aggregation": aggregation,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AASIST-L-SIH-v4 inference on an audio file.")
    parser.add_argument("audio_path", help="Path to an audio file supported by SoundFile.")
    parser.add_argument("--device", help="Override automatic CUDA/CPU selection.")
    args = parser.parse_args()
    print(json.dumps(VoiceDetector(device=args.device).predict(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
