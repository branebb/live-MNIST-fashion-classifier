import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .dataset import train_loader, classes
from .model_architecture import FashionMNIST_CNN
from .decorators import time_function

@time_function
def calculate_accuracy(model: FashionMNIST_CNN, loader: DataLoader, 
                       device: torch.device) -> float:
    model.eval()
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def show_sample_images() -> None:
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(classes[labels[i].item()])
        axes[i].axis("off")
    plt.show()