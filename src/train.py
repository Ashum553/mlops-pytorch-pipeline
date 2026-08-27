import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch import nn

from src.dataset import get_dataloaders
from src.model import get_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


def main():

    config_path = Path(
        os.getenv(
            "TRAINING_CONFIG",
            "configs/training_config.yaml",
        )
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    early_config = config["early_stopping"]
    checkpoint_config = config["checkpoint"]

    train_loader, test_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    checkpoint_dir = Path(
        os.getenv(
            "CHECKPOINT_DIR",
            checkpoint_config["dir"],
        )
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / checkpoint_config["filename"]

    epochs = training_config["epochs"]
    best_accuracy = 0.0
    epochs_without_improvement = 0

    mlflow.set_experiment("cifar10_cnn")

    with mlflow.start_run(run_name="cnn_training"):

        mlflow.log_params(
            {
                "architecture": model_config["architecture"],
                "epochs": training_config["epochs"],
                "batch_size": data_config["batch_size"],
                "learning_rate": training_config["learning_rate"],
                "weight_decay": training_config["weight_decay"],
                "seed": seed,
                "device": str(device),
            }
        )

        for epoch in range(1, epochs + 1):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                step=epoch,
            )

            if test_accuracy > best_accuracy + early_config["min_delta"]:

                best_accuracy = test_accuracy

                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "architecture": model_config["architecture"],
                        "num_classes": model_config["num_classes"],
                        "test_accuracy": test_accuracy,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

                print(
                    f"  ✓ Saved best model → {checkpoint_path}"
                )

            else:

                epochs_without_improvement += 1

            if (
                early_config["enabled"]
                and epochs_without_improvement
                >= early_config["patience"]
            ):

                print(
                    f"Early stopping triggered after epoch {epoch}."
                )

                break

        mlflow.log_metric(
            "best_test_accuracy",
            best_accuracy,
        )

        mlflow.log_artifact(
            str(checkpoint_path),
            artifact_path="checkpoints",
        )

        print()
        print("Training complete.")
        print(f"Best test accuracy: {best_accuracy:.4f}")
        print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
