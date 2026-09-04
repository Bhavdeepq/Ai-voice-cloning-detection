"""
AASIST-L fine-tuning script for ASVspoof 2019 LA.
Based on the official AASIST training pipeline.
"""

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2019_devNeval,
    genSpoof_list,
)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    with open(args.config, "r") as f:
        config = json.load(f)

    model_config = config["model_config"]
    optim_config = config["optim_config"]

    optim_config["epochs"] = config["num_epochs"]

    track = config["track"]

    if track not in ["LA", "PA", "DF"]:
        raise ValueError(f"Invalid track: {track}")

    config.setdefault("eval_all_best", "False")
    config.setdefault("freq_aug", "False")

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    set_seed(args.seed, config)

    # ---------------------------------------------------------
    # Database paths
    # ---------------------------------------------------------
    database_path = Path(config["database_path"])

    prefix = f"ASVspoof2019.{track}"

    dev_trial_path = (
        database_path
        / f"ASVspoof2019_{track}_cm_protocols/"
          f"{prefix}.cm.dev.trl.txt"
    )

    eval_trial_path = (
        database_path
        / f"ASVspoof2019_{track}_cm_protocols/"
          f"{prefix}.cm.eval.trl.txt"
    )

    # ---------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------
    output_dir = Path(args.output_dir)

    model_tag = (
        f"{track}_"
        f"{Path(args.config).stem}_"
        f"ep{config['num_epochs']}_"
        f"bs{config['batch_size']}"
    )

    if args.comment:
        model_tag += f"_{args.comment}"

    model_tag = output_dir / model_tag

    model_save_path = model_tag / "weights"
    metric_path = model_tag / "metrics"

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    writer = SummaryWriter(model_tag)

    copy(args.config, model_tag / "config.conf")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cpu":
        raise RuntimeError("CUDA GPU not detected.")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    model = get_model(model_config, device)

    # ---------------------------------------------------------
    # Load pretrained AASIST-L weights
    # ---------------------------------------------------------
    pretrained_path = config.get("pretrained_model_path")

    if pretrained_path:
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}"
            )

        print(f"Loading pretrained weights: {pretrained_path}")

        state_dict = torch.load(
            pretrained_path,
            map_location=device
        )

        model.load_state_dict(state_dict)

        print("Loaded pretrained AASIST-L weights successfully.")

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------
    train_loader, dev_loader, eval_loader = get_loader(
        database_path,
        args.seed,
        config
    )

    # ---------------------------------------------------------
    # Evaluation-only mode
    # ---------------------------------------------------------
    if args.eval:

        eval_weights = args.eval_model_weights

        if eval_weights is None:
            eval_weights = config["model_path"]

        print(f"Loading evaluation weights: {eval_weights}")

        model.load_state_dict(
            torch.load(
                eval_weights,
                map_location=device
            )
        )

        print("Starting evaluation...")

        eval_score_path = model_tag / "eval_scores.txt"

        produce_evaluation_file(
            eval_loader,
            model,
            device,
            eval_score_path,
            eval_trial_path
        )

        print("Evaluation complete.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optim_config["steps_per_epoch"] = len(train_loader)

    optimizer, scheduler = create_optimizer(
        model.parameters(),
        optim_config
    )

    optimizer_swa = SWA(optimizer)

    best_dev_eer = float("inf")
    best_dev_tdcf = float("inf")

    n_swa_update = 0

    log_file = open(
        model_tag / "metric_log.txt",
        "a"
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(config["num_epochs"]):

        print(
            f"\n========== Epoch {epoch + 1}/{config['num_epochs']} =========="
        )

        running_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scheduler,
            config
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        dev_score_path = metric_path / "dev_score.txt"

        produce_evaluation_file(
            dev_loader,
            model,
            device,
            dev_score_path,
            dev_trial_path
        )

        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=dev_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path /
            f"dev_t-DCF_EER_epoch_{epoch}.txt",
            printout=False
        )

        print(
            f"Loss: {running_loss:.5f}"
        )

        print(
            f"Dev EER: {dev_eer:.3f}"
        )

        print(
            f"Dev t-DCF: {dev_tdcf:.5f}"
        )

        writer.add_scalar(
            "loss",
            running_loss,
            epoch
        )

        writer.add_scalar(
            "dev_eer",
            dev_eer,
            epoch
        )

        writer.add_scalar(
            "dev_tdcf",
            dev_tdcf,
            epoch
        )

        # -----------------------------------------------------
        # Save best model
        # -----------------------------------------------------
        if dev_eer < best_dev_eer:

            best_dev_eer = dev_eer

            print(
                f"New best model! EER = {dev_eer:.3f}"
            )

            torch.save(
                model.state_dict(),
                model_save_path /
                f"best_epoch_{epoch}.pth"
            )

            # SWA update
            optimizer_swa.update_swa()

            n_swa_update += 1

        best_dev_tdcf = min(
            best_dev_tdcf,
            dev_tdcf
        )

        writer.add_scalar(
            "best_dev_eer",
            best_dev_eer,
            epoch
        )

        writer.add_scalar(
            "best_dev_tdcf",
            best_dev_tdcf,
            epoch
        )

    # ---------------------------------------------------------
    # Final model
    # ---------------------------------------------------------
    final_path = model_save_path / "finetuned_AASIST_L.pth"

    torch.save(
        model.state_dict(),
        final_path
    )

    print(
        f"\nFine-tuned model saved to: {final_path}"
    )

    log_file.write(
        f"Best Dev EER: {best_dev_eer:.4f}\n"
    )

    log_file.write(
        f"Best Dev t-DCF: {best_dev_tdcf:.5f}\n"
    )

    log_file.close()

    writer.close()

    print("\nTraining complete.")


# =============================================================
# MODEL
# =============================================================

def get_model(
    model_config: Dict,
    device: torch.device
):

    module = import_module(
        f"models.{model_config['architecture']}"
    )

    model_class = getattr(
        module,
        "Model"
    )

    model = model_class(
        model_config
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {num_params}"
    )

    return model


# =============================================================
# DATA LOADERS
# =============================================================

def get_loader(
    database_path: Path,
    seed: int,
    config: dict
) -> List[torch.utils.data.DataLoader]:

    track = config["track"]

    prefix = f"ASVspoof2019.{track}"

    train_database_path = (
        database_path /
        f"ASVspoof2019_{track}_train"
    )

    dev_database_path = (
        database_path /
        f"ASVspoof2019_{track}_dev"
    )

    eval_database_path = (
        database_path /
        f"ASVspoof2019_{track}_eval"
    )

    train_protocol = (
        database_path /
        f"ASVspoof2019_{track}_cm_protocols/"
        f"{prefix}.cm.train.trn.txt"
    )

    dev_protocol = (
        database_path /
        f"ASVspoof2019_{track}_cm_protocols/"
        f"{prefix}.cm.dev.trl.txt"
    )

    eval_protocol = (
        database_path /
        f"ASVspoof2019_{track}_cm_protocols/"
        f"{prefix}.cm.eval.trl.txt"
    )

    # ---------------------------------------------------------
    # Training files
    # ---------------------------------------------------------
    labels, file_train = genSpoof_list(
        dir_meta=train_protocol,
        is_train=True,
        is_eval=False
    )

    # Limit training set
    max_train_samples = config.get(
        "max_train_samples"
    )

    if max_train_samples is not None:

        max_train_samples = int(
            max_train_samples
        )

        file_train = file_train[
            :max_train_samples
        ]

        labels = {
            key: labels[key]
            for key in file_train
        }

    print(
        f"Training files: {len(file_train)}"
    )

    train_set = Dataset_ASVspoof2019_train(
        list_IDs=file_train,
        labels=labels,
        base_dir=train_database_path
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    # ---------------------------------------------------------
    # Development
    # ---------------------------------------------------------
    _, file_dev = genSpoof_list(
        dir_meta=dev_protocol,
        is_train=False,
        is_eval=False
    )

    print(
        f"Validation files: {len(file_dev)}"
    )

    dev_set = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_dev,
        base_dir=dev_database_path
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    file_eval = genSpoof_list(
        dir_meta=eval_protocol,
        is_train=False,
        is_eval=True
    )

    eval_set = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_eval,
        base_dir=eval_database_path
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    return (
        train_loader,
        dev_loader,
        eval_loader
    )


# =============================================================
# EVALUATION
# =============================================================

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str
):

    model.eval()

    with open(trial_path, "r") as f:
        trial_lines = f.readlines()

    filenames = []
    scores = []

    for batch_x, utt_ids in data_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        with torch.no_grad():

            _, batch_out = model(
                batch_x
            )

            batch_score = (
                batch_out[:, 1]
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

        filenames.extend(
            utt_ids
        )

        scores.extend(
            batch_score.tolist()
        )

    assert len(trial_lines) == len(filenames)

    with open(save_path, "w") as f:

        for filename, score, trial in zip(
            filenames,
            scores,
            trial_lines
        ):

            _, utt_id, _, src, key = (
                trial.strip().split()
            )

            if filename != utt_id:
                raise RuntimeError(
                    f"ID mismatch: {filename} != {utt_id}"
                )

            f.write(
                f"{utt_id} {src} {key} {score}\n"
            )

    print(
        f"Scores saved: {save_path}"
    )


# =============================================================
# TRAINING EPOCH
# =============================================================

def train_epoch(
    train_loader: DataLoader,
    model,
    optimizer,
    device,
    scheduler,
    config
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    class_weights = torch.FloatTensor(
        [0.1, 0.9]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        batch_y = (
            batch_y
            .long()
            .to(
                device,
                non_blocking=True
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, outputs = model(
            batch_x,
            Freq_aug=str_to_bool(
                config["freq_aug"]
            )
        )

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        if (
            config["optim_config"]["scheduler"]
            in ["cosine", "keras_decay"]
        ):
            scheduler.step()

        total_loss += (
            loss.item()
            * batch_x.size(0)
        )

        total_samples += (
            batch_x.size(0)
        )

    return (
        total_loss
        / total_samples
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AASIST-L Fine-Tuning"
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str
    )

    parser.add_argument(
        "--output_dir",
        default="./exp_result",
        type=str
    )

    parser.add_argument(
        "--seed",
        default=1234,
        type=int
    )

    parser.add_argument(
        "--eval",
        action="store_true"
    )

    parser.add_argument(
        "--comment",
        default=None,
        type=str
    )

    parser.add_argument(
        "--eval_model_weights",
        default=None,
        type=str
    )

    args = parser.parse_args()

    main(args)