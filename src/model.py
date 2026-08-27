import torch
from torch import nn


class CIFAR10CNN(nn.Module):
    """Simple CNN for CIFAR-10 image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_model(
    architecture: str = "cnn",
    num_classes: int = 10,
) -> nn.Module:
    """Create the configured image-classification model."""

    if architecture.lower() == "cnn":
        return CIFAR10CNN(num_classes=num_classes)

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        "Supported architecture: cnn"
    )
