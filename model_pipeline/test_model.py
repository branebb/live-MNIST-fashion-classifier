import unittest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .model_architecture import FashionMNIST_CNN
from .dataset import get_transforms, get_dataloader
from .config import DEVICE


class TestModel(unittest.TestCase):
    
    def test_forward_pass(self) -> None:
        model = FashionMNIST_CNN().to(DEVICE)
        input_tensor: torch.Tensor = torch.randn(1, 1, 28, 28).to(DEVICE)  
        output: torch.Tensor = model(input_tensor)
        self.assertEqual(
            output.shape,
            (1, 10),
            f"Expected output shape (1, 10), but got {output.shape}",
        )

    def test_get_transforms(self) -> None:
        transform = get_transforms()
        self.assertIsInstance(
            transform, 
            transforms.Compose, 
            "Expected a transforms.Compose object"
        )

    def test_get_dataloader(self) -> None:
        train_loader: DataLoader = get_dataloader(32, train=True)
        self.assertIsInstance(
            train_loader, 
            DataLoader, 
            "Expected DataLoader, but got something else"
        )
        self.assertEqual(
            len(train_loader.dataset), 
            60000, 
            f"Expected dataset size 60000, but got {len(train_loader.dataset)}"
        )

    def test_model_parameter_count(self) -> None:
        model = FashionMNIST_CNN()
        expected_parameters: int = 421834
        self.assertEqual(
            model.num_parameters,
            expected_parameters,
            f"Expected {expected_parameters} parameters," 
            f"but got {model.num_parameters}"
        )

    def test_model_prediction(self) -> None:
        model = FashionMNIST_CNN().to(DEVICE)
        model.load_state_dict(
            torch.load("pretrained_models/mnist_fashion_model.pth")
        )
        model.eval()

        sample_image: torch.Tensor = torch.randn(1, 28, 28).to(DEVICE)
        predicted_class, probabilities = model.predict(sample_image)

        self.assertIsInstance(
            predicted_class, 
            int, 
            "Predicted class is not an integer."
        )
        self.assertEqual(
            predicted_class, 
            int(predicted_class), 
            "Predicted class is not an integer value."
        )
        self.assertIsInstance(
            probabilities, 
            torch.Tensor, 
            "Probabilities should be a tensor."
        )


if __name__ == "__main__":
    unittest.main()