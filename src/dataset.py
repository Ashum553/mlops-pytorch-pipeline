from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transform():
    """Transform and augment CIFAR-10 training images."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_test_transform():
    """Transform CIFAR-10 test images."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
):
    """Download CIFAR-10 and return training and test DataLoaders."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=get_train_transform(),
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=get_test_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
