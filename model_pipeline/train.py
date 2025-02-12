import torch
import torch.optim as optim
import torch.nn as nn
from .model_architecture import FashionMNIST_CNN
from .dataset import train_loader, test_loader
from .utils import calculate_accuracy
from .config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH
from .decorators import time_function

@time_function
def train_model() -> None:
    model: FashionMNIST_CNN = FashionMNIST_CNN().to(DEVICE)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss: float = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc: float = calculate_accuracy(model, 
                                              train_loader, 
                                              DEVICE)
        test_acc: float = calculate_accuracy(model, 
                                             test_loader, 
                                             DEVICE)

        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")