from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class FoRDataset(Dataset):
    """
    Fake-or-Real dataset loader for AASIST-L.

    Folder structure:
        training/
        ├── real/
        └── fake/

    Labels:
        1 = real
        0 = fake
    """

    def __init__(
        self,
        root_dir,
        max_real=None,
        max_fake=None,
        num_samples=64600,
    ):
        self.root_dir = Path(root_dir)
        self.num_samples = num_samples

        real_files = sorted(
            (self.root_dir / "real").glob("*.wav")
        )

        fake_files = sorted(
            (self.root_dir / "fake").glob("*.wav")
        )

        if max_real is not None:
            real_files = real_files[:max_real]

        if max_fake is not None:
            fake_files = fake_files[:max_fake]

        self.files = (
            [(f, 1) for f in real_files]
            + [(f, 0) for f in fake_files]
        )

        print(f"REAL files: {len(real_files)}")
        print(f"FAKE files: {len(fake_files)}")
        print(f"Total files: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, label = self.files[index]

        # Load and resample to 16 kHz mono
        audio, _ = librosa.load(
            path,
            sr=16000,
            mono=True,
        )

        # Pad short audio
        if len(audio) < self.num_samples:
            repeats = int(
                np.ceil(
                    self.num_samples / len(audio)
                )
            )
            audio = np.tile(audio, repeats)

        # Crop long audio
        audio = audio[:self.num_samples]

        audio = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return audio, label