import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import ImageDataset, transform
from train import device
# from model import model

model = ""

image_dir = "/kaggle/input/para2dataset/Para_dataset_2/train_data/parasitic"
dataset = ImageDataset(image_dir=image_dir, transform=transform)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def calculate_reconstruction_error(recon_x, x):
    # Reconstruction error (Mean Squared Error between original and reconstructed)
    return F.mse_loss(recon_x, x, reduction='none').mean([1, 2, 3])

def detect_anomalies(model, dataloader, threshold):
    model.eval()
    anomalies = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Reconstruct images
            recon_batch, _, _ = model(batch)
            
            # Calculate reconstruction error
            recon_error = calculate_reconstruction_error(recon_batch, batch)
            reconstruction_errors.extend(recon_error.cpu().numpy())
            
            # Detect anomalies (reconstruction error greater than the threshold)
            anomalies.extend(recon_error > threshold)
    
    return anomalies, reconstruction_errors

# Example of setting a threshold for anomaly detection
threshold = 0.01 

# Detect anomalies
anomalies, errors = detect_anomalies(model, test_dataloader, threshold)

# Print the number of detected anomalies
print(f"Number of anomalies detected: {sum(anomalies)}")
