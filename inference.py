import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import scipy.io
from natsort import natsorted
import pandas as pd
from skimage.metrics import structural_similarity
from cm.karras_diffusion import KarrasDenoiser, karras_sample
from cm.script_util import create_ema_and_scales_fn
import matplotlib.pyplot as plt

# Configuration
config = {
    'input_dir': r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\dataset\train_mat\\",  # Directory containing input low-dose PET images
    'output_dir': r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\results\inference",  # Directory to save synthesized high-dose PET images
'checkpoint': r"D:\Users\UFPB\gabriel ayres\New folder\pet-denoising\checkpoints_newdataset_normalized\consistency_model_best.pt",  # Path to model checkpoint

'batch_size': 1,  # Batch size for inference
'num_steps': 3,  # Number of diffusion steps
'eval_mode': False,  # Enable evaluation mode if ground truth is available
}
from monai.transforms import (
Compose,
ResizeWithPadOrCropd,
    ToTensord,
    ScaleIntensityd
)
from monai.inferers import SlidingWindowInferer
from cm.karras_diffusion import karras_sample

# Network import - using the same model architecture as training
from Network.Diffusion_model_Unet_2d import UNetModel

def calculate_metrics(pred, target):
    """Calculate various image quality metrics"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - target))
    
    # Mean Squared Error
    mse = np.mean((pred - target) ** 2)
    
    # Peak Signal to Noise Ratio
    max_pixel = max(np.max(pred), np.max(target))
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
    # Structural Similarity Index
    data_range = np.max([np.max(pred) - np.min(pred), np.max(target) - np.min(target)])
    ssim = structural_similarity(pred, target, data_range=data_range)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }

class InferenceDataset:
    def __init__(self, imgs_path, eval_mode=False):
        self.imgs_path = imgs_path
        self.eval_mode = eval_mode
        files_list = []
        for img_path in imgs_path:
            file_list = natsorted(glob.glob(os.path.join(img_path, "*.mat")), key=lambda y: y.lower())     
            files_list += file_list
        self.data = []
        print(f"Found {len(files_list)} files")
        for img_path in files_list:
            class_name = os.path.basename(img_path)  # Correctly extracts the filename
            self.data.append([img_path, class_name])

        self.transforms = Compose([
            # Data normalization: -1 to 1
            ScaleIntensityd(keys=["image"], minv=-1, maxv=1.0),
            # Pad or crop all images to a uniform size
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=(96, 192),
                constant_values=-1,
            ),
            ToTensord(keys=["image"]),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        cao = scipy.io.loadmat(img_path)
        
        # Convert data to numpy array and ensure it's 2D
        image_data = cao['img'][:, :128, :]
        label_data = cao['img'][:, 128:, :]       
        # Add channel dimension if not present (shape should be [C,H,W])
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        if label_data is not None and label_data.ndim == 2:
            label_data = label_data[np.newaxis, ...]
        
        # Create the data dictionary with numpy array
        data_dict = {
            'image': image_data.astype(np.float32),
        }
        if label_data is not None:
            data_dict['label'] = label_data.astype(np.float32)
        
        transformed_data = self.transforms(data_dict)
        result = {'image': transformed_data['image'], 'filename': class_name}
        if 'label' in transformed_data:
            result['label'] = transformed_data['label']
        return result

def setup_model(checkpoint_path, device):
    # Model parameters (same as training)
    num_channels = 128
    attention_resolutions = "16,8"
    channel_mult = (1, 2, 3, 4)
    num_heads = [4, 4, 8, 16]
    window_size = [[4,4], [4,4], [4,4], [4,4]]
    num_res_blocks = [2, 2, 2, 2]
    patch_width = 64
    patch_size = (patch_width, patch_width)
    sample_kernel = ([2,2],[2,2],[2,2]),  # Trailing comma makes it a tuple of tuples
    
    # Calculate attention resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(patch_size[0]//int(res))

    # Initialize model
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

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        print("Loading from training checkpoint...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Loading from model-only checkpoint...")
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Initialize consistency model
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

def diffusion_sampling(low_dose, model, consistency, device, num_steps=3):
    """Perform diffusion sampling with Karras scheduler"""
    steps = np.round(np.linspace(1.0, 150.0, num=num_steps))
    sampled_images = karras_sample(
        consistency,
        model,
        x_start=low_dose,
        shape=low_dose.shape,
        condition=low_dose,
        sampler="multistep",
        steps=151,
        ts=steps,
        device=device
    )
    return sampled_images

def main():
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # Set up device
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Initialize model and consistency
    model, consistency = setup_model(config['checkpoint'], device)

    # Set up data loader
    dataset = InferenceDataset([config['input_dir']], eval_mode=config['eval_mode'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    # Set up sliding window inferer
    inferer = SlidingWindowInferer(
        roi_size=(64, 64),
        sw_batch_size=40,
        overlap=0.75,
        mode='constant',
        cval=-1,
        sw_device=device,
        device=device
    )

    # Run inference
    print("Starting inference...")
    metrics_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            low_dose = batch['image'].to(device)
            filenames = batch['filename']
            
            # Generate high-dose images
            high_dose_samples = inferer(low_dose, lambda x: diffusion_sampling(x, model, consistency,device, config['num_steps']))
            
            # Calculate metrics if in evaluation mode
            if config['eval_mode'] and 'label' in batch:
                high_dose_true = batch['label'].to(device)
                for i in range(len(filenames)):
                    metrics = calculate_metrics(high_dose_samples[i, 0], high_dose_true[i, 0])
                    metrics['filename'] = filenames[i]
                    metrics_list.append(metrics)
            
            # Save results
            for i, filename in enumerate(filenames):
                # Save .mat file
                output_path = os.path.join(config['output_dir'], f"synthesized_{filename}")
                save_dict = {
                    'image': high_dose_samples[i, 0].cpu().numpy(),
                    'low_dose_input': low_dose[i, 0].cpu().numpy()
                }
                if config['eval_mode'] and 'label' in batch:
                    save_dict['high_dose_true'] = batch['label'][i, 0].cpu().numpy()
                scipy.io.savemat(output_path, save_dict)
                
                # Save PNG files
                base_name = os.path.splitext(filename)[0]
                
                # Save synthesized high-dose image
                plt.figure(figsize=(8, 8))
                plt.imshow(high_dose_samples[i, 0].cpu().numpy(), cmap='hot')
                plt.colorbar(label='Intensity')
                plt.axis('off')
                plt.title('Synthesized High-dose PET')
                plt.savefig(os.path.join(config['output_dir'], f"synthesized_{base_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save input low-dose image
                plt.figure(figsize=(8, 8))
                plt.imshow(low_dose[i, 0].cpu().numpy(), cmap='hot')
                plt.colorbar(label='Intensity')
                plt.axis('off')
                plt.title('Input Low-dose PET')
                plt.savefig(os.path.join(config['output_dir'], f"input_{base_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # If in eval mode, save ground truth high-dose image
                if config['eval_mode'] and 'label' in batch:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(batch['label'][i, 0].cpu().numpy(), cmap='hot')
                    plt.colorbar(label='Intensity')
                    plt.axis('off')
                    plt.title('Ground Truth High-dose PET')
                    plt.savefig(os.path.join(config['output_dir'], f"ground_truth_{base_name}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Save metrics if in evaluation mode
    if config['eval_mode'] and metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        print("\nEvaluation Metrics:")
        print(f"Mean MAE: {metrics_df['MAE'].mean():.4f} ± {metrics_df['MAE'].std():.4f}")
        print(f"Mean MSE: {metrics_df['MSE'].mean():.4f} ± {metrics_df['MSE'].std():.4f}")
        print(f"Mean PSNR: {metrics_df['PSNR'].mean():.2f} ± {metrics_df['PSNR'].std():.2f}")
        print(f"Mean SSIM: {metrics_df['SSIM'].mean():.4f} ± {metrics_df['SSIM'].std():.4f}")
        
        # Save metrics to CSV
        metrics_path = os.path.join(config['output_dir'], 'evaluation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nDetailed metrics saved to {metrics_path}")

    print(f"Inference completed. Results saved in {config['output_dir']}")

if __name__ == "__main__":
    main()
