import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.AASIST import Model
from src.for_dataset import FoRDataset


CONFIG_PATH = Path("configs/finetune.conf")

FOR_TRAIN = Path(
    r"C:/Users/Bhavdeep/Downloads/archive/for-original/for-original/training"
)

FOR_VAL = Path(
    r"C:/Users/Bhavdeep/Downloads/archive/for-original/for-original/validation"
)

PRETRAINED = Path("models/AASIST-L.pth")

# Final checkpoint produced by the FoR fine-tuning workflow.
OUTPUT = Path("models/AASIST-L-FoR-finetuned.pth")

TRAIN_REAL = 2000
TRAIN_FAKE = 2000

VAL_REAL = 100
VAL_FAKE = 100

BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-5


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError("CUDA GPU not detected.")

print("Device:", device)


# ---------------------------------------------------------
# Datasets
# ---------------------------------------------------------

train_dataset = FoRDataset(
    FOR_TRAIN,
    max_real=TRAIN_REAL,
    max_fake=TRAIN_FAKE,
)

val_dataset = FoRDataset(
    FOR_VAL,
    max_real=VAL_REAL,
    max_fake=VAL_FAKE,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Model(
    config["model_config"]
).to(device)

print("Loading pretrained:", PRETRAINED)

state_dict = torch.load(
    PRETRAINED,
    map_location=device,
)

model.load_state_dict(state_dict)

print("Pretrained AASIST-L loaded.")


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [0.5, 0.5],
        device=device,
    )
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print(
        f"\n========== Epoch {epoch + 1}/{EPOCHS} =========="
    )

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True,
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, output = model(batch_x)

        loss = criterion(
            output,
            batch_y,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_x.size(0)
        )

        predictions = torch.argmax(
            output,
            dim=1,
        )

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_x.size(0)

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    print(
        f"Train Loss: {train_loss:.5f}"
    )

    print(
        f"Train Accuracy: {train_acc:.4f}"
    )


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_x, batch_y in val_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True,
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True,
        )

        _, output = model(batch_x)

        predictions = torch.argmax(
            output,
            dim=1,
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

val_acc = correct / total

print(
    f"Validation Accuracy: {val_acc:.4f}"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

torch.save(
    model.state_dict(),
    OUTPUT,
)

print(
    f"Model saved to: {OUTPUT}"
)
