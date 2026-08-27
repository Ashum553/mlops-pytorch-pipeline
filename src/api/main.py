from pathlib import Path

import os
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

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


CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "artifacts/checkpoints/best_model.pt",
    )
)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


app = FastAPI(
    title="CIFAR-10 Image Classification API",
    version="1.0.0",
)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)


def load_model():
    """Load the trained CIFAR-10 CNN checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = get_model(
        architecture=checkpoint.get("architecture", "cnn"),
        num_classes=checkpoint.get("num_classes", 10),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 Image Classification API",
        "model": "CNN",
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the CIFAR-10 class of an uploaded image."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()

        image = Image.open(
            __import__("io").BytesIO(contents)
        ).convert("RGB")

        image = image.resize((32, 32))

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1,
            )

        class_index = predicted_class.item()

        return {
            "filename": file.filename,
            "prediction": CLASS_NAMES[class_index],
            "class_index": class_index,
            "confidence": round(confidence.item(), 4),
            "model": "CIFAR10CNN",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
        ) from exc
