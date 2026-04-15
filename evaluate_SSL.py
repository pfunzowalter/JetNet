import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# 1. FIX IMPORTS: Pull from the correct files
from models import SimCLR_Radio  # Model architecture stays in models.py
from simulation import MultiViewRadioDataset  # Simulator moved to simulation.py

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "simclr_radio_encoder.pth"
BATCH_SIZE = 64

def evaluate():
    # 2. Load the Pre-trained Encoder
    full_model = SimCLR_Radio(base_model="resnet18").to(DEVICE)
    # Load the state dict we saved during training
    full_model.encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder = full_model.encoder
    encoder.eval() 

    # 3. Prepare Evaluation Data
    # We use 1000 samples to test how well the clusters formed
    eval_dataset = MultiViewRadioDataset(n_samples=1000, size=128)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    features_list = []
    labels_list = []

    print("Extracting features from the latent space...")
    with torch.no_grad():
        for view1, view2, labels in eval_loader:
            view1 = view1.to(DEVICE)
            # We only need one view to extract features for t-SNE
            features = encoder(view1) 
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 4. Visualize with t-SNE
    print("Running t-SNE visualization...")
    # tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    # To this:
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ticks=[0, 1], label='0: Compact, 1: Jetted')
    plt.title("t-SNE: SSL Learned Radio Morphologies (Compact vs Jetted)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("tsne_results.png")
    # plt.show()

    # 5. Linear Probe Accuracy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    print(f"\n--- Linear Probe Results ---")
    print(f"Validation Accuracy: {score * 100:.2f}%")

if __name__ == "__main__":
    evaluate()