from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images, image_names, transform=None):
        self.images = images
        self.image_names = image_names  # Store only names, not paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img_name = self.image_names[idx]  # Use image name instead of path
        if self.transform:
            image = self.transform(image)
        return image, img_name

    
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),        
])