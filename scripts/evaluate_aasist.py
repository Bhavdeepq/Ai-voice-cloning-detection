"""Evaluate an AASIST checkpoint on a reproducible ASVspoof 2019 LA dev sample."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Allow `python scripts/evaluate_aasist.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.AASIST import Model


NUM_SAMPLES = 64600


def read_protocol(protocol_path: Path, samples_per_class: int, seed: int):
    real, spoof = [], []
    for line in protocol_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        (real if parts[4] == "bonafide" else spoof).append((parts[1], 1 if parts[4] == "bonafide" else 0))

    rng = np.random.default_rng(seed)
    real = [real[i] for i in rng.permutation(len(real))[:samples_per_class]]
    spoof = [spoof[i] for i in rng.permutation(len(spoof))[:samples_per_class]]
    if len(real) != samples_per_class or len(spoof) != samples_per_class:
        raise ValueError("The requested sample is larger than the available dev set.")
    return real + spoof


def load_audio(path: Path):
    audio, sample_rate = sf.read(path)
    if sample_rate != 16000:
        raise ValueError(f"{path.name}: expected 16 kHz audio, found {sample_rate} Hz")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if len(audio) < NUM_SAMPLES:
        audio = np.tile(audio, int(np.ceil(NUM_SAMPLES / len(audio))))
    return audio[:NUM_SAMPLES].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/AASIST.conf")
    parser.add_argument("--checkpoint", default="models/AASIST.pth")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    dataset_root = Path(args.dataset_root or config["database_path"])
    protocol = dataset_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    audio_dir = dataset_root / "ASVspoof2019_LA_dev" / "flac"
    examples = read_protocol(protocol, args.samples_per_class, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config["model_config"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    correct = total = true_real = false_spoof = false_real = true_spoof = 0
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            audio = np.stack([load_audio(audio_dir / f"{utt_id}.flac") for utt_id, _ in batch])
            labels = np.array([label for _, label in batch])
            _, logits = model(torch.from_numpy(audio).to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()

            correct += int((predictions == labels).sum())
            total += len(labels)
            true_real += int(((labels == 1) & (predictions == 1)).sum())
            false_spoof += int(((labels == 1) & (predictions == 0)).sum())
            false_real += int(((labels == 0) & (predictions == 1)).sum())
            true_spoof += int(((labels == 0) & (predictions == 0)).sum())

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {total} ({args.samples_per_class} real, {args.samples_per_class} spoof), seed={args.seed}")
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("             REAL  SPOOF")
    print(f"REAL       {true_real:4d}  {false_spoof:5d}")
    print(f"SPOOF      {false_real:4d}  {true_spoof:5d}")


if __name__ == "__main__":
    main()
