import torch

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE: int = 64
EPOCHS: int = 10
LEARNING_RATE: float = 0.001

MODEL_PATH: str = "model_pipeline/pretrained_models/mnist_fashion_model.pth"