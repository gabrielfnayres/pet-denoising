#!/usr/bin/env python3
# Normalization Analysis for PET Images

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import mean_squared_error

def analyze_normalization():
    # Path to the PET image
    IMAGE_PATH = "/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/halfdataset/train/100_070722_3_20220707_164103_127.mat"

    # Load the data
    full_image = scipy.io.loadmat(IMAGE_PATH)['img']
    print(f"Original image shape: {full_image.shape}")
    
    # Extract low-dose and full-dose parts
    # The low-dose PET is in the first half: image[:,0:128,:]
    # The full-dose (ground truth) is in the second half: image[:,128:256,:]
    low_dose = full_image[:,0:128,:]
    full_dose = full_image[:,128:256,:]
    
    print(f"Low-dose shape: {low_dose.shape}")
    print(f"Full-dose shape: {full_dose.shape}")
    
    # Squeeze the first dimension if it's 1
    if low_dose.shape[0] == 1:
        low_dose = np.squeeze(low_dose, axis=0)
        full_dose = np.squeeze(full_dose, axis=0)
        
    print(f"After squeeze - Low-dose shape: {low_dose.shape}")
    print(f"After squeeze - Full-dose shape: {full_dose.shape}")
    
    # Calculate basic statistics for both images
    total_pixels = low_dose.shape[0] * low_dose.shape[1]

    # Basic statistics for low-dose image
    print("\nLow-dose image statistics:")
    print(f"Min value: {np.min(low_dose)}")
    print(f"Max value: {np.max(low_dose)}")
    print(f"Mean value: {np.mean(low_dose)}")
    print(f"Standard deviation: {np.std(low_dose)}")
    
    # Basic statistics for full-dose image
    print("\nFull-dose image statistics:")
    print(f"Min value: {np.min(full_dose)}")
    print(f"Max value: {np.max(full_dose)}")
    print(f"Mean value: {np.mean(full_dose)}")
    print(f"Standard deviation: {np.std(full_dose)}")

    # Apply different normalization methods to low-dose image
    min_max_norm = (low_dose - np.min(low_dose)) / (np.max(low_dose) - np.min(low_dose))
    z_norm = (low_dose - np.mean(low_dose)) / np.std(low_dose)
    one_one = 2 * ((low_dose - np.min(low_dose)) / (np.max(low_dose) - np.min(low_dose))) - 1

    # Visualize the results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # First row: original images
    axes[0, 0].imshow(low_dose, cmap='hot')
    axes[0, 0].set_title('Low-dose (Original)')
    
    axes[0, 1].imshow(min_max_norm, cmap='hot')
    axes[0, 1].set_title('Min-Max Normalization')
    
    axes[0, 2].imshow(z_norm, cmap='hot')
    axes[0, 2].set_title('Z-score Normalization')
    
    axes[0, 3].imshow(one_one, cmap='hot')
    axes[0, 3].set_title('[-1, 1] Normalization')
    
    # Second row: comparison with full dose
    axes[1, 0].imshow(full_dose, cmap='hot')
    axes[1, 0].set_title('Full-dose (Ground Truth)')
    
    # Normalize full-dose for fair comparison
    full_dose_norm = (full_dose - np.min(full_dose)) / (np.max(full_dose) - np.min(full_dose))
    axes[1, 1].imshow(full_dose_norm, cmap='hot')
    axes[1, 1].set_title('Full-dose (Normalized)')
    
    # Difference maps
    diff_original = np.abs(full_dose_norm - min_max_norm)
    axes[1, 2].imshow(diff_original, cmap='hot')
    axes[1, 2].set_title('Difference Map')
    
    # SSIM or another metric visualization
    axes[1, 3].imshow(np.abs(full_dose - low_dose), cmap='hot')
    axes[1, 3].set_title('Original Difference')
    plt.tight_layout()
    plt.savefig('normalization_comparison.png')
    plt.show()

    # Normalize full-dose image for fair comparison
    full_dose_min_max = (full_dose - np.min(full_dose)) / (np.max(full_dose) - np.min(full_dose))
    full_dose_z = (full_dose - np.mean(full_dose)) / np.std(full_dose)
    full_dose_one_one = 2 * ((full_dose - np.min(full_dose)) / (np.max(full_dose) - np.min(full_dose))) - 1
    
    # Calculate MSE between normalized low-dose and normalized full-dose
    original_mse = mean_squared_error(low_dose.flatten(), full_dose.flatten())
    min_max_mse = mean_squared_error(min_max_norm.flatten(), full_dose_min_max.flatten())
    z_norm_mse = mean_squared_error(z_norm.flatten(), full_dose_z.flatten())
    one_one_mse = mean_squared_error(one_one.flatten(), full_dose_one_one.flatten())

    print("\nMSE comparison between normalized low-dose and normalized full-dose images:")
    print(f"Original (unnormalized): {original_mse:.6f}")
    print(f"Min-Max Normalization: {min_max_mse:.6f}")
    print(f"Z-score Normalization: {z_norm_mse:.6f}")
    print(f"[-1, 1] Normalization: {one_one_mse:.6f}")

    methods = ["Original", "Min-Max", "Z-score", "[-1, 1]"]
    mse_values = [original_mse, min_max_mse, z_norm_mse, one_one_mse]
    best_method_index = mse_values.index(min(mse_values))
    print(f"\nBest normalization method: {methods[best_method_index]} with MSE of {min(mse_values):.6f}")
    
    # Calculate additional metrics for the best normalization method
    print(f"\nPercentage improvement of {methods[best_method_index]} over original: {((original_mse - min(mse_values))/original_mse)*100:.2f}%")

if __name__ == "__main__":
    analyze_normalization()
