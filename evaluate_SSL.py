import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# --- Imports from your project files ---
from models import SimCLR_Radio
from simulation import MultiViewRadioDataset

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
ENCODER_PATH = "simclr_radio_encoder.pth"
BATCH_SIZE = 64
N_SAMPLES = 2000  # Increased for a more robust t-SNE plot

def evaluate():
    # 1. Load the Pre-trained Encoder
    # Ensure the architecture matches your training script
    full_model = SimCLR_Radio(base_model="resnet18").to(DEVICE)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: {ENCODER_PATH} not found. Please train the model first.")
        return

    # Load the state dict
    full_model.encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder = full_model.encoder
    encoder.eval() 

    # 2. Prepare Evaluation Data
    print(f"Generating {N_SAMPLES} samples for evaluation...")
    eval_dataset = MultiViewRadioDataset(n_samples=N_SAMPLES, size=128)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    features_list = []
    labels_list = []

    print("Extracting features...")
    with torch.no_grad():
        for view1, _, labels in eval_loader:
            view1 = view1.to(DEVICE)
            
            # Extract features from the encoder
            feat = encoder(view1)
            
            # Safety check: If ResNet returns a 4D tensor (B, C, H, W), apply Global Average Pooling
            if len(feat.shape) == 4:
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = torch.flatten(feat, 1)
            
            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 3. Feature Normalization
    # Standardizing features is critical for both t-SNE and Logistic Regression
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 4. Visualize with t-SNE
    print("Running t-SNE visualization (this may take a minute)...")
    # Using 'pca' initialization for better stability and 'auto' learning rate
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    features_2d = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=labels, 
        cmap='coolwarm', 
        alpha=0.7, 
        edgecolors='w', 
        linewidth=0.5
    )
    
    # Customizing the legend/colorbar for your specific classes
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Compact', 'Jetted'])
    
    plt.title("t-SNE: SSL Learned Radio Morphologies", fontsize=15)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_name = "tsne_morphology_clusters.png"
    plt.savefig(save_name, dpi=300)
    print(f"t-SNE plot saved as {save_name}")

    # 5. Linear Probe Accuracy (Downstream Task)
    print("Training Linear Probe (Logistic Regression)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )
    
    # max_iter increased to ensure convergence on complex feature sets
    clf = LogisticRegression(max_iter=5000, solver='lbfgs').fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    print(f"\n" + "="*30)
    print(f"--- EVALUATION RESULTS ---")
    print(f"Linear Probe Accuracy: {score * 100:.2f}%")
    print(f"Total Samples Tested: {N_SAMPLES}")
    print(f"="*30)

if __name__ == "__main__":
    import os
    evaluate()