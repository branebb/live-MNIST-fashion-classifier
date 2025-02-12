import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .config import BATCH_SIZE

def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_dataloader(batch_size: int, train: bool) -> DataLoader:
    dataset = datasets.FashionMNIST(
        root="./data", 
        train=train, 
        transform=get_transforms(), 
        download=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


train_loader: DataLoader = get_dataloader(BATCH_SIZE, train=True)
test_loader: DataLoader = get_dataloader(BATCH_SIZE, train=False)

classes: list[str] = train_loader.dataset.classes