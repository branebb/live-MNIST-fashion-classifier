from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DEVICE
from .decorators import (
    time_function,
    validate_model_loaded,
    log_function_call
)


class BaseModel(ABC, nn.Module):
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def predict(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        pass


class FashionMNIST_CNN(BaseModel):
    def __init__(self) -> None:
        super(FashionMNIST_CNN, self).__init__()
        
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, 
                                          out_channels=32, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(32)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=32, 
                                          out_channels=64, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, 
                                               stride=2)
        self.dropout: nn.Dropout = nn.Dropout(0.3)
        self.fc1: nn.Linear = nn.Linear(64 * 7 * 7, 128)
        self.fc2: nn.Linear = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    # @log_function_call
    # @validate_model_loaded
    # @time_function
    def predict(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        image = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output: torch.Tensor = self(image)
            probabilities: torch.Tensor = F.softmax(output, dim=1).squeeze()
            predicted_class: int = torch.argmax(probabilities).item()
            return predicted_class, probabilities.round(decimals=4)
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())