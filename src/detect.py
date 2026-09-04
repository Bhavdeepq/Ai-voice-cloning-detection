import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from models.AASIST import Model


BASE_DIR = Path(__file__).resolve().parent.parent
# The final FoR-tuned checkpoint is the model used by the project demo.
# AASIST-L.pth remains in models/ as the base checkpoint for re-training.
MODEL_PATH = BASE_DIR / "models" / "AASIST-L-FoR-finetuned.pth"
CONFIG_PATH = BASE_DIR / "configs" / "finetune.conf"

NUM_SAMPLES = 64600


def load_model():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(config["model_config"]).to(device)

    weights = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(weights)
    model.eval()

    return model, device


def preprocess_audio(audio_path):
    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, got {sample_rate} Hz"
        )

    if len(audio) < NUM_SAMPLES:
        repeats = int(np.ceil(NUM_SAMPLES / len(audio)))
        audio = np.tile(audio, repeats)

    audio = audio[:NUM_SAMPLES]

    return torch.tensor(
        audio,
        dtype=torch.float32
    )


def detect(audio_path):
    model, device = load_model()

    audio = preprocess_audio(audio_path)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        _, output = model(audio)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    spoof_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    if spoof_probability > real_probability:
        prediction = "SPOOF"
        confidence = spoof_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "spoof_probability": spoof_probability,
        "real_probability": real_probability,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.detect <audio_file>")
        raise SystemExit(1)

    result = detect(sys.argv[1])

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Spoof probability: {result['spoof_probability'] * 100:.2f}%")
    print(f"Real probability: {result['real_probability'] * 100:.2f}%")
