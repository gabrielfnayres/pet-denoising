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
    'test_data_dir': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/data/pet_38_aligned/imagesTs_full_2d',
    'results_dir': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/results_normalized/evaluation',
    'checkpoint': '/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/checkpoints_normalized/consistency_model_checkpoint.pt',
    'batch_size': 1,
    'num_steps': 3,
}

def calculate_metrics(pred, target):
    """Calculate comprehensive image quality metrics"""
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    
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
                # Save the results
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
    
    # Create dataset
    dataset = EvaluationDataset(config['test_data_dir'])
    
    # Setup model
    model, consistency = setup_model(config['checkpoint'], device)
    
    # Run evaluation
    results = evaluate_model(
        model,
        consistency,
        dataset,
        device,
        num_steps=config['num_steps'],
        save_dir=config['results_dir']
    )

if __name__ == "__main__":
    main()