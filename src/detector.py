"""Reusable inference interface for the final AASIST-L-SIH-v3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.AASIST import Model
from .preprocessing import load_mono_16khz, split_into_windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "AASIST-L-SIH-v3.pth"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "finetune.conf"


class VoiceDetector:
    """Classify audio as ``REAL`` or ``AI_SPOOF`` using the final V3 model only."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, config_path: str | Path = DEFAULT_CONFIG_PATH, device: str | torch.device | None = None, batch_size: int = 8) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        if not self.model_path.is_file():
            raise FileNotFoundError(f"V3 checkpoint not found: {self.model_path}")
        if self.model_path.name != "AASIST-L-SIH-v3.pth":
            raise ValueError("VoiceDetector only accepts models/AASIST-L-SIH-v3.pth.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        with self.config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)["model_config"]
        self.window_samples = int(model_config["nb_samp"])
        self.model = Model(model_config).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Return mean segment probabilities and the final detector prediction."""
        windows = split_into_windows(load_mono_16khz(audio_path), self.window_samples)
        probabilities: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(np.stack(windows[start : start + self.batch_size])).to(self.device)
                _, logits = self.model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu())
        mean_probabilities = torch.cat(probabilities).mean(dim=0)
        fake_probability = float(mean_probabilities[0])
        real_probability = float(mean_probabilities[1])
        prediction = "AI_SPOOF" if fake_probability >= real_probability else "REAL"
        confidence = fake_probability if prediction == "AI_SPOOF" else real_probability
        return {"prediction": prediction, "confidence": confidence, "real_probability": real_probability, "fake_probability": fake_probability, "segments_analyzed": len(windows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AASIST-L-SIH-v3 inference on an audio file.")
    parser.add_argument("audio_path", help="Path to an audio file supported by SoundFile.")
    parser.add_argument("--device", help="Override automatic CUDA/CPU selection.")
    args = parser.parse_args()
    print(json.dumps(VoiceDetector(device=args.device).predict(args.audio_path), indent=2))


if __name__ == "__main__":
    main()
