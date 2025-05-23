
# 2D-Medical-Consistency-Model
**This is the repository for the paper published in Medical Physics: "[Full-dose Whole-body PET Synthesis from Low-dose PET Using High-efficiency Denoising Diffusion Probabilistic Model: PET Consistency Model](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17068)".**

Consistency Model is one of the super fast Denoising Diffusion Probability Models (DDPMs), which only use 2-timestep to generate the target image, while the DDPMs usually require 50- to 1000-timesteps. This is particular useful for: 1) Three-dimensional Medical image synthesis, 2) Image translation instead image creation like traditional DDPMs do.

The codes were created based on [image-guided diffusion](https://github.com/openai/guided-diffusion), [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet), and [Monai](https://monai.io/)

Notice: Due to the data restriction, we can only provide MATLAB file (so no patient information) with over-smoothed PET images. The data we show just to demonstrate how the user should organize their data. The dicom or nii file processing are also included in the Jupyter notebook.

# Required packages

The requires packages are in test_env.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\test_env.yaml
```

# How to organize your data
The data organization example is shown in folder "data/pet_38_aligned". Or you can see the below screenshots:
![image](https://github.com/shaoyanpan/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/assets/89927506/a2fdf7af-25be-47d7-8b49-7bc7c2c2468f)
MATLAB files: every matlab file can contain a dict has image and label together. So you see you only need two folders: imagesTr_full_2d for training, imagesTs_full_2d for testing. You can change the name but please make sure also change the reading dir in the jupyter notebook.

![image](https://github.com/shaoyanpan/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/assets/89927506/a7bf529f-3e4e-4e58-b0fe-3a87fb5ecbe9)
Nii files: one nii file can only contain either image or label. So in this case, you need imagesTr and labelsTr for training, imagesTs and labelsTs for testing, and imagesVal and labelsVal for validation

# Usage

There are two ways to use this model:

1. Using the Jupyter notebook `Consistency_Low_Dose_Denoising_main.ipynb` for interactive experimentation
2. Using the Python scripts for training and inference

## Using the Python Scripts

### Training
To train the model, use `train_consistency.py`. The script includes progress bars and automatic checkpoint saving:

```bash
python train_consistency.py
```

### Inference
To generate high-dose PET images from low-dose inputs, use `inference.py`:

```bash
python inference.py \
    --input_dir /path/to/low/dose/images \
    --output_dir /path/to/save/results \
    --checkpoint /path/to/model/checkpoint.pt \
    --batch_size 1 \
    --num_steps 3
```

Arguments:
- `--input_dir`: Directory containing input low-dose PET images (in .mat format)
- `--output_dir`: Directory to save synthesized high-dose images
- `--checkpoint`: Path to the trained model checkpoint
- `--batch_size`: Batch size for inference (default: 1)
- `--num_steps`: Number of diffusion steps (default: 3)
- `--eval`: Enable evaluation mode if ground truth is available

The output will be saved as .mat files containing:
- Synthesized high-dose image
- Original low-dose input
- Ground truth high-dose image (if in evaluation mode)

When running in evaluation mode (--eval flag), the script will calculate and output:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

These metrics will be displayed in the console and saved to a CSV file in the output directory.

## Interactive Usage
For interactive experimentation, you can use the Jupyter notebook. Here are some examples:

**Create Consistency-diffusion**
```
from cm.resample import UniformSampler
from cm.karras_diffusion import KarrasDenoiser,karras_sample
consistency = KarrasDenoiser(        
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l1")

schedule_sampler = UniformSampler(consistency)
```

**Create network for input image with size of 64x64 (Notice this is because we apply the 64x64 patch-based training and inference for our 96x196 low-dose PET images**
```
from Diffusion_model_transformer import *

num_channels=128
attention_resolutions="16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4],[4,4],[4,4],[4,4]]
num_res_blocks = [2,2,2,2]
sample_kernel=([2,2],[2,2],[2,2]),

attention_ds = []
for res in attention_resolutions.split(","):
    # Careful for the image_size//int(res), only use for CNN
    attention_ds.append(image_size//int(res))
class_cond = False
use_scale_shift_norm = True

Consistency_network = SwinVITModel(
        image_size=img_size,
        in_channels=2,
        model_channels=num_channels,
        out_channels=1,
        dims=2,
        sample_kernel = sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size = window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)

# Don't forget the ema model. You must have this to run the code no matter you use ema or not.
Consistency_network_ema = copy.deepcopy(Consistency_network)
```

**Train the consistency model (you don't have to use the ema as in our .ipynb**
```
# Create fake examples, just for you to run the code
img_size = (96,192) # Adjust this for the size of your image input
condition = torch.randn([1,1,96,192]) #batch, channel, height, width
target = torch.randn([1,1,96,192]) #batch, channel, height, width

all_loss = consistency.consistency_losses(Consistency_network,
            target,
            condition,
            num_scales,
            target_model=Consistency_network_ema)
loss = (all_loss["loss"] * weights).mean()
```

**generate new synthetic images**
```
# Create fake examples
Low_dose = torch.randn([1,1,96,192]) #batch, channel, height, width
img_size = (96,192) # Adjust this for the size of your image input

# Set up the step# for your inference
consistency_num = 3
steps = np.round(np.linspace(1.0, 150.0, num=consistency_num))
def diffusion_sampling(Low_dose,A_to_B_model):
    sampled_images = karras_sample(
                        consistency,
                        A_to_B_model,
                        shape=Low_dose.shape,
                        condition=Low_dose,
                        sampler="multistep",
                        steps = 151,
                        ts = steps,
                        device = device)
    return sampled_images

# Patch-based inference parameter
overlap = 0.75
mode ='constant'
back_ground_intensity = -1
Inference_patch_number_each_time = 40
from monai.inferers import SlidingWindowInferer
inferer = SlidingWindowInferer(img_size, Inference_patch_number_each_time, overlap=overlap,
                               mode =mode ,cval = back_ground_intensity, sw_device=device,device = device)

# 
High_dose_samples = inferer(Low_dose,diffusion_sampling,Consistency_network)  
```


# Visual examples
![Picture1](https://github.com/shaoyanpan/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/assets/89927506/15e56941-d7c6-4eab-994a-04e2d1d4d1df)

