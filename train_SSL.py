import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import the network and loss from models.py
# Ensure info_nce_loss accepts (features, batch_size, temperature)
from models import SimCLR_Radio, info_nce_loss

# Import the dataset and simulator from simulation.py
from simulation import MultiViewRadioDataset

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64      
EPOCHS = 50          
LR = 1e-3            
TEMPERATURE = 0.1    
SAVE_PATH = "simclr_radio_encoder.pth"

def train():
    # 1. Initialize Dataset and DataLoader
    # Adjust n_samples based on your available memory/time
    train_dataset = MultiViewRadioDataset(n_samples=10000, size=128)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True 
    )

    # 2. Initialize Model
    model = SimCLR_Radio().to(DEVICE)
    
    # 3. Setup Optimizers
    # We split optimizers to allow the linear probe to monitor 
    # progress without affecting the encoder weights
    contrastive_optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.projector.parameters()), 
        lr=LR
    )
    probe_optimizer = optim.Adam(model.linear_probe.parameters(), lr=LR)
    
    # 4. Define Loss Functions
    criterion_probe = nn.CrossEntropyLoss()

    print(f"Starting SimCLR training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_contrastive_loss = 0
        epoch_probe_loss = 0

        for v1, v2, labels in train_loader:
            v1, v2, labels = v1.to(DEVICE), v2.to(DEVICE), labels.to(DEVICE)

            # --- STEP 1: Contrastive Update (Self-Supervised) ---
            # Forward pass through the encoder + projector
            z1 = model(v1)
            z2 = model(v2)
            
            # Combine views for the InfoNCE loss
            # We pass TEMPERATURE here to ensure the scaling is correct
            loss_c = info_nce_loss(torch.cat([z1, z2], dim=0), BATCH_SIZE, temperature=TEMPERATURE)
            
            contrastive_optimizer.zero_grad()
            loss_c.backward()
            contrastive_optimizer.step()

            # --- STEP 2: Linear Probe Update (Supervised Monitor) ---
            # We use 'detach' to ensure the gradient only flows through the linear layer,
            # keeping the encoder purely self-supervised.
            with torch.no_grad():
                h = model(v1, return_embedding=True)
            
            outputs = model.linear_probe(h.detach())
            loss_p = criterion_probe(outputs, labels)

            probe_optimizer.zero_grad()
            loss_p.backward()
            probe_optimizer.step()

            epoch_contrastive_loss += loss_c.item()
            epoch_probe_loss += loss_p.item()

        # Average losses for the epoch
        avg_c_loss = epoch_contrastive_loss / len(train_loader)
        avg_p_loss = epoch_probe_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Contrastive Loss: {avg_c_loss:.4f} | Probe Loss: {avg_p_loss:.4f}")

        # 5. Save Checkpoint
        # Usually, we only want to save the encoder for downstream RFI tasks
        torch.save(model.encoder.state_dict(), SAVE_PATH)

    print(f"Training finished. Encoder saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()