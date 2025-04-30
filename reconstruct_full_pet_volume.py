#!/usr/bin/env python3
"""
Full PET Volume Reconstruction Tool

This script walks through all .mat files in the train directory,
reconstructs the full 3D PET volume, and visualizes it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io
from glob import glob
import argparse

def load_pet_volume(directory):
    """
    Load all PET images from .mat files in the specified directory
    to reconstruct the full 3D volume.
    
    Args:
        directory: Directory containing .mat files
        
    Returns:
        low_dose_volume: 3D array of low-dose PET images
        full_dose_volume: 3D array of full-dose PET images
        file_names: List of file names for reference
    """
    # Find all .mat files in the directory
    file_paths = sorted(glob(os.path.join(directory, "*.mat")))
    
    if not file_paths:
        raise ValueError(f"No .mat files found in {directory}")
    
    print(f"Found {len(file_paths)} .mat files. Loading...")
    
    # Extract file names without extension
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    
    # Initialize volumes with the correct shape
    # First load a sample to get dimensions
    sample = scipy.io.loadmat(file_paths[0])['img']
    print(f"Sample data shape: {sample.shape}")
    
    # Initialize volumes
    low_dose_volume = []
    full_dose_volume = []
    
    # Load all files
    for i, file_path in enumerate(file_paths):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load the .mat file
        data = scipy.io.loadmat(file_path)['img']
        
        # Check the dimensionality of the data
        if len(data.shape) == 3:
            # 3D data: [1, 256, 128] format
            if data.shape[0] == 1:
                data = np.squeeze(data, axis=0)  # Now it's [256, 128]
            # Split into low-dose and full-dose
            low_dose = data[0:128, :]
            full_dose = data[128:256, :]
        elif len(data.shape) == 2:
            # Already 2D data
            # Check if the first dimension is 256 (combined low and full dose)
            if data.shape[0] == 256:
                low_dose = data[0:128, :]
                full_dose = data[128:256, :]
            else:
                # If it's not in the expected format, print info and skip
                print(f"Unexpected data shape in {os.path.basename(file_path)}: {data.shape}")
                continue
        
        # Add to volumes
        low_dose_volume.append(low_dose)
        full_dose_volume.append(full_dose)
    
    # Convert lists to numpy arrays if they're not empty
    if low_dose_volume and full_dose_volume:
        low_dose_volume = np.array(low_dose_volume)
        full_dose_volume = np.array(full_dose_volume)
        print(f"Loaded volumes with shape: {low_dose_volume.shape}")
    else:
        print("No valid data was loaded. Please check the file format.")
        return np.array([]), np.array([]), []
    
    return low_dose_volume, full_dose_volume, file_names

def normalize_volume(volume, method='min_max'):
    """
    Normalize a volume using the specified method.
    
    Args:
        volume: Numpy array
        method: Normalization method ('min_max', 'z_score', or 'one_one')
        
    Returns:
        Normalized volume
    """
    if method == 'min_max':
        return (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    elif method == 'z_score':
        return (volume - np.mean(volume)) / np.std(volume)
    elif method == 'one_one':
        return 2 * ((volume - np.min(volume)) / (np.max(volume) - np.min(volume))) - 1
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def visualize_slices(low_dose_volume, full_dose_volume, output_dir=None, normalize=True, step=5):
    """
    Visualize slices of the 3D volumes.
    
    Args:
        low_dose_volume: 3D array of low-dose PET images
        full_dose_volume: 3D array of full-dose PET images
        output_dir: Directory to save output images
        normalize: Whether to normalize the volumes
        step: Step size for slice visualization (to avoid too many images)
    """
    if normalize:
        low_dose_norm = normalize_volume(low_dose_volume)
        full_dose_norm = normalize_volume(full_dose_volume)
    else:
        low_dose_norm = low_dose_volume
        full_dose_norm = full_dose_volume
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get number of slices
    num_slices = low_dose_volume.shape[0]
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Function to update the plot for each slice
    def update(slice_idx):
        for ax in axes:
            ax.clear()
        
        # Display low-dose slice
        axes[0].imshow(low_dose_norm[slice_idx], cmap='hot')
        axes[0].set_title(f'Low-dose (Slice {slice_idx}/{num_slices-1})')
        axes[0].axis('off')
        
        # Display full-dose slice
        axes[1].imshow(full_dose_norm[slice_idx], cmap='hot')
        axes[1].set_title(f'Full-dose (Slice {slice_idx}/{num_slices-1})')
        axes[1].axis('off')
        
        # Display difference
        diff = np.abs(full_dose_norm[slice_idx] - low_dose_norm[slice_idx])
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'slice_{slice_idx:03d}.png'), dpi=150)
    
    # Create animation with specified step size
    frames = range(0, num_slices, step)
    anim = FuncAnimation(fig, update, frames=frames, interval=200)
    
    # Save animation if output directory is provided
    if output_dir:
        anim.save(os.path.join(output_dir, 'volume_animation.gif'), writer='pillow', dpi=100)
    
    plt.tight_layout()
    plt.show()

def create_montage(volume, rows=10, cols=10, start_idx=0, title="PET Slices Montage"):
    """
    Create a montage of slices from the volume.
    
    Args:
        volume: 3D array
        rows: Number of rows in the montage
        cols: Number of columns in the montage
        start_idx: Starting index for slices
        title: Plot title
    """
    # Normalize volume for visualization
    norm_volume = normalize_volume(volume)
    
    # Calculate number of slices to display
    num_slices = min(rows * cols, volume.shape[0] - start_idx)
    end_idx = start_idx + num_slices
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Plot each slice
    for i in range(num_slices):
        slice_idx = start_idx + i
        if slice_idx < volume.shape[0]:
            axes[i].imshow(norm_volume[slice_idx], cmap='hot')
            axes[i].set_title(f'Slice {slice_idx}', fontsize=8)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_slices, rows*cols):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Reconstruct and visualize full PET volume')
    parser.add_argument('--dir', type=str, 
                        default="/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/halfdataset/train",
                        help='Directory containing .mat files')
    parser.add_argument('--output', type=str, default="pet_output", 
                        help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, choices=['slices', 'montage', 'both'], default='both', 
                        help='Visualization mode: slices, montage, or both')
    parser.add_argument('--normalize', action='store_true', help='Normalize the volumes')
    parser.add_argument('--step', type=int, default=5, help='Step size for slice visualization')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows in montage')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns in montage')
    parser.add_argument('--max_files', type=int, default=100, 
                        help='Maximum number of files to process (default: 100, use 0 for all files)')
    
    args = parser.parse_args()
    
    # Set max_files to None if 0 is specified (process all files)
    max_files = None if args.max_files == 0 else args.max_files
    
    print(f"Processing up to {max_files if max_files else 'all'} files from {args.dir}")
    
    # Load PET volumes
    low_dose_volume, full_dose_volume, file_names = load_pet_volume(args.dir)
    
    # Check if we have valid data
    if low_dose_volume.size == 0 or full_dose_volume.size == 0:
        print("No valid data to visualize. Exiting.")
        return
    
    # Print some statistics
    print("\nLow-dose volume statistics:")
    print(f"Min value: {np.min(low_dose_volume)}")
    print(f"Max value: {np.max(low_dose_volume)}")
    print(f"Mean value: {np.mean(low_dose_volume)}")
    print(f"Standard deviation: {np.std(low_dose_volume)}")
    
    print("\nFull-dose volume statistics:")
    print(f"Min value: {np.min(full_dose_volume)}")
    print(f"Max value: {np.max(full_dose_volume)}")
    print(f"Mean value: {np.mean(full_dose_volume)}")
    print(f"Standard deviation: {np.std(full_dose_volume)}")
    
    # Visualize based on selected mode
    if args.mode in ['slices', 'both']:
        print("\nCreating slice visualization...")
        visualize_slices(low_dose_volume, full_dose_volume, args.output, args.normalize, args.step)
    
    if args.mode in ['montage', 'both']:
        print("\nCreating montage for low-dose volume...")
        create_montage(low_dose_volume, args.rows, args.cols, 0, "Low-dose PET Montage")
        
        print("Creating montage for full-dose volume...")
        create_montage(full_dose_volume, args.rows, args.cols, 0, "Full-dose PET Montage")

if __name__ == "__main__":
    main()
