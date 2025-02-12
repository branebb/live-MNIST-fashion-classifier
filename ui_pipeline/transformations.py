from torchvision import transforms
from torchvision.transforms import Compose

transformation: Compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])