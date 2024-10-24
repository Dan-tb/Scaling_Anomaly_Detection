import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),        
])

# image_dir = "C:/Users/USER/Downloads/Garba/Uninfected/"
# dataset = ImageDataset(image_dir=image_dir, transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)