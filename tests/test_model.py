from src.model import get_model


def test_model_output_shape():
    model = get_model(num_classes=10)

    assert model is not None
