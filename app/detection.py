import torch
import torch.nn as nn
import torch.nn.functional as F
# from train import device
from preprocess import ImageDataset, transform


image_dir = "C:/Users/USER/Downloads/Garba/Uninfected/"
dataset = ImageDataset(image_dir=image_dir, transform=transform)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_reconstruction_error(recon_x, x):
    # Reconstruction error (Mean Squared Error between original and reconstructed)
    return F.mse_loss(recon_x, x, reduction='none').mean([1, 2, 3])

def detect_anomalies(model, test_dataloader, threshold):
    model.eval()
    anomalies = []
    reconstruction_errors = []
    results = []
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            batch, image_paths = batch_data  # Assuming batch contains (images, paths)
            batch = batch.to(device)
            
            # Reconstruct images
            recon_batch, _, _ = model(batch)
            
            # Calculate reconstruction error
            recon_error = calculate_reconstruction_error(recon_batch, batch)
            reconstruction_errors.extend(recon_error.cpu().numpy())
            
            # Detect anomalies (reconstruction error greater than the threshold)
            anomaly_status = recon_error > threshold
            
            # Store the results in dict form
            for img_path, is_anomaly in zip(image_paths, anomaly_status):
                results.append({"image": img_path, "is_anomaly": bool(is_anomaly)})
    
    return results, reconstruction_errors

