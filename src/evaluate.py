from pathlib import Path

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.dataset import get_dataloaders
from src.model import get_model


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    _, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=64,
        num_workers=2,
    )

    checkpoint_path = Path(
        "artifacts/checkpoints/best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Test Accuracy: {accuracy:.4f}")

    print()
    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        f"Checkpoint test accuracy: "
        f"{checkpoint['test_accuracy']:.4f}"
    )

    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "evaluation.txt"

    with open(metrics_file, "w") as f:
        f.write("CIFAR-10 MODEL EVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n")

    print(f"\nEvaluation saved to: {metrics_file}")


if __name__ == "__main__":
    main()
