import torch
from .model_architecture import FashionMNIST_CNN
from .config import DEVICE, MODEL_PATH
from .decorators import time_function

@time_function
def load_model() -> FashionMNIST_CNN:
    model: FashionMNIST_CNN = FashionMNIST_CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model