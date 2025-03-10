import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from natsort import natsorted

# Configuration
config = {
    'input_dir': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/data/pet_38_aligned/imagesTs_full_2d',  # Directory containing input low-dose PET images
    'output_dir': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/results/inference',  # Directory containing synthesized high-dose PET images
    'plot_dir': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/results/plots',  # Directory to save plots
}

def load_mat_file(filepath):
    """Load .mat file and return the image data"""
    data = scipy.io.loadmat(filepath)
    image = data['image'] if 'image' in data else data['data']
    return image

def plot_comparison(input_image, output_image, filename, save_path):
    """Plot input and output images side by side"""
    plt.figure(figsize=(12, 6))
    
    # Plot input image
    plt.subplot(121)
    plt.imshow(input_image, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.title('Low-dose Input')
    plt.axis('off')
    
    # Plot output image
    plt.subplot(122)
    plt.imshow(output_image, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.title('Synthesized High-dose')
    plt.axis('off')
    
    plt.suptitle(f'PET Image Synthesis - {filename}')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create plot directory if it doesn't exist
    os.makedirs(config['plot_dir'], exist_ok=True)
    
    # Get list of input files
    input_files = natsorted(
        [f for f in os.listdir(config['input_dir']) if f.endswith('.mat')],
        key=lambda y: y.lower()
    )
    
    # Get list of output files
    output_files = natsorted(
        [f for f in os.listdir(config['output_dir']) if f.endswith('.mat')],
        key=lambda y: y.lower()
    )
    
    print(f"Found {len(input_files)} input files and {len(output_files)} output files")
    
    # Plot each pair of images
    for input_file, output_file in zip(input_files, output_files):
        print(f"Processing {input_file}...")
        
        # Load input and output images
        input_image = load_mat_file(os.path.join(config['input_dir'], input_file))
        output_image = load_mat_file(os.path.join(config['output_dir'], output_file))
        
        # Create plot filename
        plot_filename = f"comparison_{os.path.splitext(input_file)[0]}.png"
        plot_path = os.path.join(config['plot_dir'], plot_filename)
        
        # Plot and save comparison
        plot_comparison(input_image, output_image, input_file, plot_path)
    
    print(f"Plots saved in {config['plot_dir']}")

if __name__ == "__main__":
    main()
