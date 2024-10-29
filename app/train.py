
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvVAE
from preprocess import dataloader

def loss_function(recon_x, x, mu, logvar):
    # (batch_size, 3, 64, 64)
    # Reconstruction loss (Binary Cross Entropy for images)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  # No need to flatten
    
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =ConvVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_load in dataloader:
        batch, image = batch_load
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
