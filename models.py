import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# class SimCLR_Radio(nn.Module):
#     def __init__(self, base_model="resnet18", out_dim=128):
#         super(SimCLR_Radio, self).__init__()
        
#         # 1. The Encoder (Backbone)
#         # We modify the first layer to accept 1-channel (grayscale) radio maps
#         self.encoder = models.resnet18(weights=None)
#         self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
#         # Remove the final classification layer to get the feature vector 'h'
#         num_ftrs = self.encoder.fc.in_features
#         self.encoder.fc = nn.Identity()

#         # 2. The Projection Head 'g(h)'
#         # This is where the contrastive loss is calculated
#         self.projector = nn.Sequential(
#             nn.Linear(num_ftrs, num_ftrs),
#             nn.ReLU(),
#             nn.Linear(num_ftrs, out_dim)
#         )

#     def forward(self, x):
#         h = self.encoder(x)    # Features
#         z = self.projector(h)  # Projection for contrastive loss
#         return h, z
class SimCLR_Radio(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128, num_classes=2):
        super(SimCLR_Radio, self).__init__()
        
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection Head (Used only during Pretraining)
        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim)
        )

        # Linear Probe (Used to monitor accuracy during training)
        # We wrap this in a separate optimizer in the train script
        self.linear_probe = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, return_embedding=False):
        h = self.encoder(x)
        if return_embedding:
            return h # Return features for the Linear Probe
        z = self.projector(h)
        return z

# --- The NT-Xent Loss (Normalized Temperature-scaled Cross Entropy) ---
def info_nce_loss(features, batch_size, temperature=0.5):
    """
    Standard SimCLR loss: makes positive pairs close and negative pairs far.
    """
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    
    # Mask out the self-similarity (diagonal)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives and negatives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    
    target = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    return F.cross_entropy(logits, target)
