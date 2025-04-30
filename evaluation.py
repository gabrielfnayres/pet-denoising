import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import scipy.io
from natsort import natsorted
import pandas as pd
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from cm.karras_diffusion import KarrasDenoiser, karras_sample
from cm.script_util import create_ema_and_scales_fn
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    ResizeWithPadOrCropd,
    ToTensord,
    ScaleIntensityd
)
from Network.Diffusion_model_Unet_2d import UNetModel

# Configuration
config = {
'test_data_dir': r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\dataset\test_mat\\",
'results_dir': r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\results\inference",
'checkpoint':  r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\checkpoints_newdataset_normalized\consistency_model_best.pt",
'batch_size': 1,
'num_steps': 3,
}

def to_numpy(tensor_or_array):
    """Safely convert a tensor or array to a numpy array"""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    else:
        return np.array(tensor_or_array)

def calculate_metrics(pred, target):
    """Calculate comprehensive image quality metrics"""
    # Convert PyTorch tensors to numpy arrays if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    elif isinstance(pred, np.ndarray) and pred.ndim > 2:
        pred = np.squeeze(pred)
        
    if isinstance(target, torch.Tensor):
        target = target.squeeze().cpu().numpy()
    elif isinstance(target, np.ndarray) and target.ndim > 2:
        target = np.squeeze(target)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - target))
    
    # Mean Squared Error
    mse = np.mean((pred - target) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Peak Signal to Noise Ratio
    data_range = np.max([np.max(pred) - np.min(pred), np.max(target) - np.min(target)])
    try:
        psnr = peak_signal_noise_ratio(target, pred, data_range=data_range)
    except:
        psnr = 0
    
    # Structural Similarity Index
    try:
        ssim = structural_similarity(pred, target, data_range=data_range)
    except:
        ssim = 0
    
    # Normalized Root Mean Squared Error
    nrmse = rmse / (np.max(target) - np.min(target))
    
    return {
        'PSNR': psnr,
        'SSIM': ssim
    }

class EvaluationDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = natsorted(glob.glob(os.path.join(data_dir, "*.mat")), key=lambda y: y.lower())
        print(f"Found {len(self.file_list)} files for evaluation")
        
        self.transforms = Compose([
            ScaleIntensityd(keys=["image", "label"], minv=-1, maxv=1.0),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(96, 192),
                constant_values=-1,
            ),
            ToTensord(keys=["image", "label"]),
        ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = scipy.io.loadmat(file_path)
        
        image_data = data['image'] if 'image' in data else data['data']
        label_data = data['label'] if 'label' in data else data['target']
        
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        if label_data.ndim == 2:
            label_data = label_data[np.newaxis, ...]
        
        data_dict = {
            'image': image_data.astype(np.float32),
            'label': label_data.astype(np.float32),
            'filename': os.path.basename(file_path)
        }
        
        transformed_data = self.transforms(data_dict)
        return transformed_data

def setup_model(checkpoint_path, device):
    """Initialize and load the model"""
    # Model parameters
    num_channels = 128
    attention_resolutions = "16,8"
    channel_mult = (1, 2, 3, 4)
    num_heads = [4, 4, 8, 16]
    window_size = [[4,4], [4,4], [4,4], [4,4]]
    num_res_blocks = [2, 2, 2, 2]
    patch_width = 64
    patch_size = (patch_width, patch_width)
    sample_kernel = ([2,2],[2,2],[2,2]),
    
    attention_ds = [patch_size[0]//int(res) for res in attention_resolutions.split(",")]

    model = UNetModel(
        img_size=patch_size,
        image_size=patch_width,
        in_channels=2,
        model_channels=num_channels,
        out_channels=1,
        dims=2,
        num_res_blocks=num_res_blocks[0],
        attention_resolutions=tuple(attention_ds),
        dropout=0.,
        sample_kernel=sample_kernel,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False
    ).to(device)

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    consistency = KarrasDenoiser(
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l1"
    )
    
    return model, consistency

def evaluate_model(model, consistency, dataset, device, num_steps=3, save_dir=None):
    """Evaluate model performance on the dataset"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    metrics_list = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            data = dataset[idx]
            low_dose = data['image'].unsqueeze(0).to(device)
            high_dose = data['label'].unsqueeze(0).to(device)
            filename = data['filename']
            
            # Generate prediction
            steps = np.round(np.linspace(1.0, 150.0, num=num_steps))
            prediction = karras_sample(
                consistency,
                model,
                shape=low_dose.shape,
                condition=low_dose,
                sampler="multistep",
                steps=151,
                ts=steps,
                device=device
            )
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, high_dose)
            metrics['filename'] = filename
            metrics_list.append(metrics)
            
            if save_dir:
                # Save the results as images
                base_name = os.path.splitext(filename)[0]
                
                # Convert tensors to numpy arrays and scale to 0-255 for saving as images
                low_dose_np = low_dose.squeeze().cpu().numpy()
                high_dose_np = high_dose.squeeze().cpu().numpy()
                prediction_np = prediction.squeeze().cpu().numpy()
                
                # Normalize to 0-1 range if needed
                def normalize_for_display(img):
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        return (img - img_min) / (img_max - img_min)
                    return img
                
                low_dose_np = normalize_for_display(low_dose_np)
                high_dose_np = normalize_for_display(high_dose_np)
                prediction_np = normalize_for_display(prediction_np)
                
                # Save individual images
                plt.imsave(os.path.join(save_dir, f"input_{base_name}.png"), low_dose_np, cmap='gray')
                plt.imsave(os.path.join(save_dir, f"ground_truth_{base_name}.png"), high_dose_np, cmap='gray')
                plt.imsave(os.path.join(save_dir, f"synthesized_{base_name}.png"), prediction_np, cmap='gray')
                
                # Create and save a comparison visualization
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.imshow(low_dose_np, cmap='gray')
                plt.title('Low-dose PET')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(prediction_np, cmap='gray')
                plt.title(f'Synthesized PET\nPSNR: {metrics["PSNR"]:.2f}, SSIM: {metrics["SSIM"]:.4f}')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(high_dose_np, cmap='gray')
                plt.title('Ground Truth (Full-dose)')
                plt.axis('off')
                
                plt.tight_layout()
                vis_dir = os.path.join(save_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f'comparison_{base_name}.png'))
                plt.close()
                
                # Save the results as .mat file for further analysis
                result_dict = {
                    'low_dose': low_dose.cpu().numpy(),
                    'high_dose': high_dose.cpu().numpy(),
                    'prediction': prediction.cpu().numpy(),
                    'metrics': metrics
                }
                save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_results.mat")
                scipy.io.savemat(save_path, result_dict)
    
    # Aggregate results
    results_df = pd.DataFrame(metrics_list)
    mean_metrics = results_df.mean(numeric_only=True)
    std_metrics = results_df.std(numeric_only=True)
    
    print("\nEvaluation Results:")
    print("Mean Metrics:")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    if save_dir:
        # Save detailed results to CSV
        results_df.to_csv(os.path.join(save_dir, 'detailed_metrics.csv'), index=False)
        
        # Save summary statistics
        summary = pd.DataFrame({
            'Metric': mean_metrics.index,
            'Mean': mean_metrics.values,
            'Std': std_metrics.values
        })
        summary.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)
    
    return results_df

def main():
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, consistency = setup_model(config['checkpoint'], device)
    
    # Create dataset
    dataset = EvaluationDataset(config['test_data_dir'])
    
    # Evaluate model
    results_df = evaluate_model(
        model=model,
        consistency=consistency,
        dataset=dataset,
        device=device,
        num_steps=config['num_steps'],
        save_dir=config['results_dir']
    )
    
    # Save metrics to CSV
    if results_df is not None:
        metrics_file = os.path.join(config['results_dir'], 'metrics.csv')
        results_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")

def evaluate_synthesized_images(results_dir, output_csv=None, visualize=False):
    """
    Evaluate synthesized PET images against ground truth using PSNR and SSIM metrics
    
    Args:
        results_dir: Directory containing input_ and synthesized_ images
        output_csv: Path to save metrics as CSV file (optional)
        visualize: Whether to generate visualization plots
    
    Returns:
        DataFrame with evaluation metrics
    """
    print(f"Evaluating synthesized images in {results_dir}")
    
    # Get all input image files
    input_files = natsorted(glob.glob(os.path.join(results_dir, "input_*.png")), key=lambda y: y.lower())
    
    # Extract base filenames to match with synthesized images
    pattern = re.compile(r'input_(.*)\.png')
    
    metrics_list = []
    
    # Determine naming pattern for synthesized images
    synthesized_pattern = "synthesized_{}.png"  # Default pattern
    
    # Check for alternative naming patterns if needed
    alt_patterns = ["synth_{}.png", "pred_{}.png", "output_{}.png"]
    
    for input_file in tqdm(input_files, desc="Processing images"):
        match = pattern.search(os.path.basename(input_file))
        if not match:
            continue
            
        base_name = match.group(1)
        
        # Try to find the synthesized image with different possible naming patterns
        synthesized_file = None
        for pattern_template in [synthesized_pattern] + alt_patterns:
            synth_name = pattern_template.format(base_name)
            potential_file = os.path.join(results_dir, synth_name)
            if os.path.exists(potential_file):
                synthesized_file = potential_file
                break
        
        if synthesized_file is None:
            print(f"Warning: No synthesized image found for {os.path.basename(input_file)}")
            continue
        
        # Load images
        try:
            # For PNG images
            input_img = np.array(Image.open(input_file).convert('L')) / 255.0
            synth_img = np.array(Image.open(synthesized_file).convert('L')) / 255.0
            
            # According to the memory, the low-dose PET is in the first half and full-dose in the second half
            # Extract ground truth from the input image (if applicable)
            if input_img.shape[1] == 256 and input_img.shape[0] == 128:  # Check if image has both low and full dose
                low_dose = input_img[:, :128]  # First half is low-dose
                ground_truth = input_img[:, 128:]  # Second half is full-dose (ground truth)
            else:
                # If the input image doesn't contain the ground truth, try to find it separately
                ground_truth_file = os.path.join(results_dir, f"ground_truth_{base_name}.png")
                if os.path.exists(ground_truth_file):
                    ground_truth = np.array(Image.open(ground_truth_file).convert('L')) / 255.0
                else:
                    print(f"Warning: No ground truth found for {os.path.basename(input_file)}")
                    continue
            
            # Calculate metrics
            metrics = calculate_metrics(synth_img, ground_truth)
            metrics['filename'] = os.path.basename(input_file)
            metrics_list.append(metrics)
            
            if visualize and len(metrics_list) % 10 == 0:  # Visualize every 10th image
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.imshow(low_dose, cmap='gray')
                plt.title('Low-dose PET')
                plt.subplot(132)
                plt.imshow(synth_img, cmap='gray')
                plt.title(f'Synthesized PET\nPSNR: {metrics["PSNR"]:.2f}, SSIM: {metrics["SSIM"]:.4f}')
                plt.subplot(133)
                plt.imshow(ground_truth, cmap='gray')
                plt.title('Ground Truth (Full-dose)')
                plt.tight_layout()
                
                # Save visualization
                vis_dir = os.path.join(results_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f'vis_{base_name}.png'))
                plt.close()
                
        except Exception as e:
            print(f"Error processing {os.path.basename(input_file)}: {e}")
    
    # Compile results
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        
        # Calculate average metrics
        avg_metrics = {
            'PSNR': df['PSNR'].mean(),
            'SSIM': df['SSIM'].mean()
        }
        
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_metrics['PSNR']:.4f}")
        print(f"SSIM: {avg_metrics['SSIM']:.4f}")
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")
        
        return df
    else:
        print("No metrics calculated. Check if synthesized images exist.")
        return None


def evaluate_mat_files(mat_dir, output_csv=None, visualize=False, checkpoint_path='checkpoints/best_model.pth'):
    """
    Evaluate .mat files by generating synthesized images from low-dose PET and comparing with ground truth
    
    Args:
        mat_dir: Directory containing .mat files
        output_csv: Path to save metrics as CSV file (optional)
        visualize: Whether to generate visualization plots
        checkpoint_path: Path to model checkpoint
    
    Returns:
        DataFrame with evaluation metrics
    """
    print(f"Evaluating .mat files in {mat_dir}")
    
    # Set device
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Load model
    model, consistency = setup_model(checkpoint_path, device)
    model.eval()
    
    # Get all .mat files
    mat_files = natsorted(glob.glob(os.path.join(mat_dir, "*.mat")), key=lambda y: y.lower())
    print(f"Found {len(mat_files)} .mat files")
    
    metrics_list = []
    
    # Create directory for visualizations if needed
    if visualize:
        vis_dir = os.path.join(os.path.dirname(mat_dir), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Process just one file first to debug
    debug_mode = False
    if debug_mode:
        mat_files = mat_files[:1]
    
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        try:
            # Load .mat file
            data = scipy.io.loadmat(mat_file)
            
            # Extract data based on expected structure
            if 'img' in data:
                image_data = data['img']
            elif 'image' in data:
                image_data = data['image']
            else:
                print(f"Warning: Expected data structure not found in {os.path.basename(mat_file)}")
                continue
            
            # Print debug info about the loaded data
            print(f"Image data shape: {image_data.shape}")
                
            if image_data.shape[1] == 256 and image_data.shape[2] == 128:
                # According to memory, low-dose is first half, full-dose is second half
                low_dose = image_data[:, 0:128, :]  # Low-dose PET
                ground_truth = image_data[:, 128:256, :]  # Full-dose PET (ground truth)
                
                # Make sure the data is in the right format (C, H, W)
                if low_dose.shape[0] != 1 and low_dose.shape[0] != 3:
                    # If the channel dimension is not at the front, rearrange
                    low_dose = np.transpose(low_dose, (0, 2, 1))
                    ground_truth = np.transpose(ground_truth, (0, 2, 1))
                
                # Normalize data to range [-1, 1] which is common for neural networks
                low_dose = (low_dose - low_dose.min()) / (low_dose.max() - low_dose.min()) * 2 - 1
                ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) * 2 - 1
                
                # Print shapes after normalization
                print(f"Low-dose shape: {low_dose.shape}, Ground truth shape: {ground_truth.shape}")
                
                # Convert to tensor and add batch dimension if needed
                if isinstance(low_dose, np.ndarray):
                    low_dose_tensor = torch.from_numpy(low_dose).float()
                    if low_dose_tensor.dim() == 3:  # If it's already 3D (C, H, W)
                        low_dose_tensor = low_dose_tensor.unsqueeze(0)  # Add batch dimension
                else:
                    low_dose_tensor = low_dose
                
                if isinstance(ground_truth, np.ndarray):
                    ground_truth_tensor = torch.from_numpy(ground_truth).float()
                    if ground_truth_tensor.dim() == 3:  # If it's already 3D (C, H, W)
                        ground_truth_tensor = ground_truth_tensor.unsqueeze(0)  # Add batch dimension
                else:
                    ground_truth_tensor = ground_truth
                
                # Move tensors to device
                low_dose_tensor = low_dose_tensor.to(device)
                ground_truth_tensor = ground_truth_tensor.to(device)
                
                # Print tensor shapes
                print(f"Low-dose tensor shape: {low_dose_tensor.shape}, Ground truth tensor shape: {ground_truth_tensor.shape}")
                
                # The model expects 2 channels, but we only have 1 channel
                # Create a 2-channel input by duplicating the low_dose
                if low_dose_tensor.shape[1] == 1:
                    # Option 1: Duplicate the channel
                    model_input = torch.cat([low_dose_tensor, low_dose_tensor], dim=1)
                    # Option 2: Add a zero channel
                    # zero_channel = torch.zeros_like(low_dose_tensor)
                    # model_input = torch.cat([low_dose_tensor, zero_channel], dim=1)
                else:
                    # If it already has the right number of channels, use as is
                    model_input = low_dose_tensor
                
                # Print model input shape
                print(f"Model input shape: {model_input.shape}")
                
                # Generate synthesized image
                try:
                    with torch.no_grad():
                        # Create a dummy timestep tensor (0 for inference)
                        timesteps = torch.zeros(model_input.shape[0], dtype=torch.long, device=device)
                        print(f"Timesteps shape: {timesteps.shape}, device: {timesteps.device}")
                        print(f"Model input device: {model_input.device}")
                        
                        # Forward pass through the model
                        synthesized_tensor = model(model_input, timesteps)
                        print(f"Successfully ran model forward pass")
                    
                    # Print debug info about the output
                    print(f"Model output type: {type(synthesized_tensor)}")
                    if hasattr(synthesized_tensor, 'shape'):
                        print(f"Model output shape: {synthesized_tensor.shape}")
                    
                    # Convert to numpy for metrics calculation using our helper function
                    synthesized = to_numpy(synthesized_tensor)
                    
                    # Remove batch dimension if present
                    if synthesized.ndim > 3:
                        synthesized = synthesized[0]
                except Exception as e:
                    print(f"Error during model inference: {e}")
                    continue  # Skip this file and move to the next one
                
                # Calculate metrics
                # Make sure both arrays are numpy arrays for metric calculation
                # Use our helper function to safely convert to numpy
                synthesized = to_numpy(synthesized)
                ground_truth = to_numpy(ground_truth)
                    
                metrics = calculate_metrics(synthesized, ground_truth)
                metrics['filename'] = os.path.basename(mat_file)
                metrics_list.append(metrics)
                
                # Generate visualization if requested
                if visualize and len(metrics_list) % 50 == 0:  # Visualize every 50th image to avoid too many plots
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Normalize for display
                    low_dose_display = (low_dose.squeeze() + 1) / 2
                    synthesized_display = (synthesized.squeeze() + 1) / 2
                    ground_truth_display = (ground_truth.squeeze() + 1) / 2
                    
                    axes[0].imshow(low_dose_display, cmap='gray')
                    axes[0].set_title('Low-dose PET')
                    axes[0].axis('off')
                    
                    axes[1].imshow(synthesized_display, cmap='gray')
                    axes[1].set_title(f'Synthesized PET\nPSNR: {metrics["PSNR"]:.2f}, SSIM: {metrics["SSIM"]:.4f}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(ground_truth_display, cmap='gray')
                    axes[2].set_title('Ground Truth (Full-dose PET)')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"{os.path.basename(mat_file).split('.')[0]}_comparison.png"))
                    plt.close()
            else:
                print(f"Warning: Unexpected image shape in {os.path.basename(mat_file)}: {image_data.shape}")
                
        except Exception as e:
            print(f"Error processing {os.path.basename(mat_file)}: {e}")
    
    # Compile results
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        
        # Calculate average metrics
        avg_metrics = {
            'PSNR': df['PSNR'].mean(),
            'SSIM': df['SSIM'].mean()
        }
        
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_metrics['PSNR']:.4f}")
        print(f"SSIM: {avg_metrics['SSIM']:.4f}")
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")
        
        return df
    else:
        print("No metrics calculated. Check if .mat files exist and have the correct structure.")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PET image synthesis")
    parser.add_argument("--mode", type=str, default="evaluate_mat", choices=["inference", "evaluate", "evaluate_synthesized", "evaluate_mat"], 
                        help="Mode of operation: inference, evaluate, evaluate_synthesized, or evaluate_mat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_newdataset_normalized/consistency_model_best.pt", 
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset/test_mat", 
                        help="Directory containing test data")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--output_csv", type=str, default="metrics.csv", 
                        help="Path to save metrics as CSV file")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualization plots")
    parser.add_argument("--num_steps", type=int, default=3,
                        help="Number of sampling steps")
    
    args = parser.parse_args()
    
    if args.mode == "inference":
        # Run inference
        main(args.checkpoint, args.data_dir, args.results_dir)
    elif args.mode == "evaluate":
        # Evaluate model on test dataset
        main(args.checkpoint, args.data_dir, args.results_dir, evaluate=True)
    elif args.mode == "evaluate_synthesized":
        # Evaluate synthesized images
        evaluate_synthesized_images(args.results_dir, args.output_csv, args.visualize)
    elif args.mode == "evaluate_mat":
        # Evaluate .mat files directly
        evaluate_mat_files(args.data_dir, args.output_csv, args.visualize, args.checkpoint)
