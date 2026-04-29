#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.convolution import convolve, Gaussian2DKernel
from typing import List, Tuple, Optional

class RadioMorphologySimulator:
    """
    Radio Source Simulator for SSL Training.
    """
    def __init__(self, size: int = 128, beam_fwhm: float = 5.0, pixel_scale: float = 1.0):
        self.size = size
        self.shape = (size, size)
        self.yy, self.xx = np.mgrid[:size, :size]
        
        self.beam_pix = beam_fwhm / pixel_scale
        self.sigma_pix = self.beam_pix / (2 * np.sqrt(2 * np.log(2)))
        self.psf = Gaussian2DKernel(self.sigma_pix)

    def create_compact(self, amp: float, x: float, y: float) -> np.ndarray:
        radius = np.random.uniform(0.5, 4.0)
        g = models.Gaussian2D(amplitude=amp, x_mean=x, y_mean=y, 
                              x_stddev=radius, y_stddev=radius)
        return g(self.xx, self.yy)

    def create_jetted(self, amp: float, x: float, y: float) -> np.ndarray:
        img = np.zeros(self.shape)
        theta = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(10, 25)
        is_one_sided = np.random.choice([True, False], p=[0.3, 0.7])
        curvature = np.random.uniform(-0.04, 0.04)
        
        core = models.Gaussian2D(amplitude=amp, x_mean=x, y_mean=y, 
                                 x_stddev=1.0, y_stddev=1.0)
        img += core(self.xx, self.yy)

        steps = 50 
        directions = [1] if is_one_sided else [-1, 1]
        
        for sign in directions:
            for i in range(1, steps):
                t = i / steps
                dist = t * length * sign
                curr_theta = theta + (curvature * dist)
                px, py = x + dist * np.cos(curr_theta), y + dist * np.sin(curr_theta)
                
                width = 0.8 + (t * 3.0)
                brightness = amp * (0.87**i)
                
                if i == steps - 1 and np.random.random() > 0.6:
                    brightness = amp * 1.2
                    width = 1.5

                img += models.Gaussian2D(amplitude=brightness, x_mean=px, y_mean=py,
                                         x_stddev=width, y_stddev=width)(self.xx, self.yy)
        return img

    def observe(self, sky: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        convolved = convolve(sky, self.psf, boundary='extend')
        noise = np.random.normal(0, noise_level * np.max(convolved), self.shape)
        observed = convolved + noise
        return (observed - np.min(observed)) / (np.max(observed) + 1e-8)

class MultiViewRadioDataset(Dataset):
    def __init__(self, n_samples: int = 1000, size: int = 128):
        self.sim = RadioMorphologySimulator(size=size)
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        is_jetted = np.random.random() > 0.5
        amp = np.random.uniform(50, 100)
        x, y = 64 + np.random.uniform(-5, 5), 64 + np.random.uniform(-5, 5)

        sky = self.sim.create_jetted(amp, x, y) if is_jetted else self.sim.create_compact(amp, x, y)
        
        view1 = self.sim.observe(sky, noise_level=np.random.uniform(0.01, 0.05))
        view2 = self.sim.observe(sky, noise_level=np.random.uniform(0.01, 0.05))
        
        v1_tensor = torch.from_numpy(view1).float().unsqueeze(0)
        v2_tensor = torch.from_numpy(view2).float().unsqueeze(0)
        label = 1 if is_jetted else 0
        
        return v1_tensor, v2_tensor, label

def plot_fifteen(dataset: Dataset):
    """Visualizes 15 samples from the dataset in a 3x5 grid."""
    num_to_plot = min(15, len(dataset))
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(num_to_plot):
        # Extract only the first view and the label from the dataset tuple
        v1, v2, label = dataset[i]
        img = v1.squeeze().numpy() # Remove channel dim for imshow
        
        axes[i].imshow(img, origin='lower', cmap='gist_heat')
        axes[i].axis('off')
        type_str = "Jetted" if label == 1 else "Compact"
        axes[i].set_title(f"Sample {i+1}: {type_str}")
        
    plt.tight_layout()
    plt.savefig("samples.png")
    print("Saved visualization to samples.png")

if __name__ == "__main__":
    n_to_generate = 30000 
    dataset = MultiViewRadioDataset(n_samples=n_to_generate)
    
    # Generate visualization for the first 15
    plot_fifteen(dataset)

    all_images = []
    all_labels = []

    print(f"Generating {n_to_generate} samples for training...")
    
    for i in range(n_to_generate):
        v1, _, label = dataset[i]
        all_images.append(v1.numpy())
        all_labels.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"Progress: {i+1}/{n_to_generate}")

    np.savez_compressed(
        "radio_train_data.npz", 
        images=np.array(all_images), 
        labels=np.array(all_labels)
    )
    
    print("Dataset saved successfully to 'radio_train_data.npz'")