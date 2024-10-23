import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),        
])

image_dir = "/kaggle/input/para2dataset/Para_dataset_2/train_data/uninfected"
dataset = ImageDataset(image_dir=image_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)