import time
import torch
from torch.utils.data import Dataset
import glob
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd
)
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
import copy
from cm.karras_diffusion import KarrasDenoiser,karras_sample
from cm.script_util import (
    create_ema_and_scales_fn,
)

from cm.fp16_util import (
    MixedPrecisionTrainer,
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)

# Here are the dataloader hyper-parameters, including the batch size,
# image full size, patch size, image spacing, and color channel (usually 1 for medical images)
BATCH_SIZE_TRAIN = 20*1
BATCH_SIZE_TEST = 1
img_full_size = (96,192)
patch_width = 64
patch_size = (patch_width,patch_width)
spacing = (1,1)
patch_num = 2
channels = 1
# Set up device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Here are the dataloader class if your data are .mat files which contains both image and label in a single file. For nii.gz files, see the next block.
# Please see the comment below for your data pre-processing.
# load image-> add channel dimension to the image -> intensity normalization 
# -> padding or crop the boundary to ensure all images have same size -> extract random patches
class CustomDataset(Dataset):
    def __init__(self,imgs_path,labels_path=None, train_flag = True):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.train_flag = train_flag
        files_list = []
        labels_list = []
        for img_path in imgs_path:
            file_list = natsorted(glob.glob(img_path + "*mat"), key=lambda y: y.lower())     
            files_list += file_list
        self.data = []
        self.label = []
        print(f"Found {len(files_list)} files")
        for img_path in files_list:
            class_name = img_path.split("/")[-1]
            self.data.append([img_path, class_name])

        # Now we start to see what data preprocessing we need  
        self.train_transforms = Compose(
                [
                    # Add a new dimension to channel. Must have.
  #                  EnsureChannelFirstd(keys=["image","label"]),
                    
                    # Data normalization: -1 to 1
                    ScaleIntensityd(keys=["image","label"], minv=-1, maxv=1.0),
                    
                    # Pad or crop all images to a uniform size
                    ResizeWithPadOrCropd(
                          keys=["image","label"],
                          spatial_size = img_full_size,
                          constant_values = -1,
                    ),
                    
                    # Randomly extract several patches from the input image
                    RandSpatialCropSamplesd(keys=["image","label"],
                          roi_size = patch_size,
                          num_samples = patch_num,
                          random_size=False,
                          ),
                    ToTensord(keys=["image","label"]),
                ]
            )
        self.test_transforms = Compose(
                [
#                    EnsureChannelFirstd(keys=["image","label"]),
#                     ScaleIntensityd(keys=["image","label"], minv=-1, maxv=1.0),
                    ResizeWithPadOrCropd(
                          keys=["image","label"],
                          spatial_size = img_full_size,
                          constant_values = -1,
                    ),
                    ToTensord(keys=["image","label"]),
                ]
            ) 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        cao = scipy.io.loadmat(img_path)
        
        image_data = cao['img'][:, :128, :]
        label_data = cao['img'][:, 128:, :]
        
        # Add channel dimension if not present (shape should be [C,H,W])
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        if label_data.ndim == 2:
            label_data = label_data[np.newaxis, ...]
        
        # Create the data dictionary with numpy arrays
        data_dict = {
            'image': image_data.astype(np.float32),
            'label': label_data.astype(np.float32)
        }
        
        if not self.train_flag:
            transformed_data = self.test_transforms(data_dict)   
            img_tensor = transformed_data['image']
            label_tensor = transformed_data['label']
        else:
            transformed_data = self.train_transforms(data_dict)   
            img_tensor = torch.stack([d['image'] for d in transformed_data])
            label_tensor = torch.stack([d['label'] for d in transformed_data])
        
        return {'image': img_tensor, 'label': label_tensor}

# Here enter your network parameters:
# num_channels means the initial channels in each block,128 here.
# Length of the channel_mult means the layer#, 4 here.
# channel_mult means the multipliers of the channels (in this case, 128,256,384,512 for the first to the fourth block),
# attention_resulution means we use the transformer blocks in the third to the fourth block
# number of heads, window size in each transformer block
# 
num_channels=128
attention_resolutions="16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4],[4,4],[4,4],[4,4]]
num_res_blocks = [2,2,2,2]
sample_kernel=([2,2],[2,2],[2,2]),

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(patch_width//int(res))
class_cond = False
use_scale_shift_norm = True

# from Network.Diffusion_model_transformer import *
# Consistency_network = SwinVITModel(
#         image_size=patch_size,
#         in_channels=2,
#         model_channels=num_channels,
#         out_channels=1,
#         dims=2,
#         sample_kernel = sample_kernel,
#         num_res_blocks=num_res_blocks,
#         attention_resolutions=tuple(attention_ds),
#         dropout=0,
#         channel_mult=channel_mult,
#         num_classes=None,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=num_heads,
#         window_size = window_size,
#         num_head_channels=64,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=False,
#         use_new_attention_order=False,
#     ).to(device)

# # Don't forget the target model. You must have this to run the code no matter you use ema or not.
# Consistency_network_target = copy.deepcopy(Consistency_network)

## The Unet we provide here if you want to use it
from Network.Diffusion_model_Unet_2d import *
Consistency_network = UNetModel(
        img_size = patch_size,
        image_size=patch_width,
        in_channels=2,
        model_channels=num_channels,
        out_channels=1,
        dims = 2,
        num_res_blocks=num_res_blocks[0],
        attention_resolutions=tuple(attention_ds),
        dropout=0.,
        sample_kernel=sample_kernel,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=True,
        use_new_attention_order=False,
    ).to(device)

# Don't forget the target model. You must have this to run the code no matter you use ema or not.
Consistency_network_target = copy.deepcopy(Consistency_network)

# Setup the hyper-parameters of the consistency model
consistency = KarrasDenoiser(        
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l1")

# Consistency start and end steps in the ema optimization process

start_scales = 2
end_scales = 150
metric = torch.nn.L1Loss()
ema_scale_fn = create_ema_and_scales_fn(
    target_ema_mode='adaptive',
    start_ema=0.95,
    scale_mode='progressive',
    start_scales=start_scales,
    end_scales=end_scales,
    total_steps=800000,
    distill_steps_per_iter=5000,
)

ema_rate  = "0.9999,0.99994,0.9999432189950708"
real_ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
mp_trainer = MixedPrecisionTrainer(
            model=Consistency_network,
            use_fp16=True,
            fp16_scale_growth=False,
        )
Consistency_model_optimizer = torch.optim.RAdam(mp_trainer.master_params, lr=1e-5,weight_decay = 1e-4)



# Here we Consistency_network the training process
def train(Consistency_network, Consistency_network_target, Consistency_model_optimizer, data_loader1, consistency, global_step):
    
    #1: set the network and its ema network to training mode
    Consistency_network.train()
    Consistency_network_target.requires_grad_(False)
    Consistency_network_target.train()
    total_samples = len(data_loader1.dataset)
    A_to_B_loss_sum = []
    total_time = 0
    aa = time.time()
    
    progress_bar = tqdm(data_loader1, desc=f'Training', position=1, leave=False)
    for i, data in enumerate(progress_bar):
        
        #2: Loop the whole dataset, condition (low dose PET, input) and target (high dose PET, target)
        condition = data['image'].view(-1,1,patch_width,patch_width).to(device)
        target = data['label'].view(-1,1,patch_width,patch_width).to(device)
        
        #3: EMA optimizations setup
        A_to_B_param_groups_and_shapes = get_param_groups_and_shapes(
            Consistency_network.named_parameters()
        )
        A_to_B_master_params = make_master_params(A_to_B_param_groups_and_shapes)

        ema_params = [
            copy.deepcopy(A_to_B_master_params)
            for _ in range(len(real_ema_rate))
        ]
        
        ema, num_scales = ema_scale_fn(global_step)
        
                
        #4: Optimize the Consistency_network to correct perform the consistency process
        mp_trainer.zero_grad()
        # Use torch.amp.autocast with device type
        with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            all_loss = consistency.consistency_losses(Consistency_network,
                        target,
                        condition,
                        num_scales,
                        target_model=Consistency_network_target)
            
            A_to_B_loss = all_loss['loss'].mean()
            A_to_B_loss_sum.append(A_to_B_loss.detach().cpu().numpy())
            
        mp_trainer.backward(A_to_B_loss)
        mp_trainer.optimize(Consistency_model_optimizer)

        #5: _update_ema()
        for rate, params in zip(real_ema_rate, ema_params):
            for targ, src in zip(params, A_to_B_master_params):
                targ.detach().mul_(0.99).add_(src, alpha=1 - 0.99)
        
        target_model_param_groups_and_shapes = get_param_groups_and_shapes(
            Consistency_network_target.named_parameters()
        )
        target_model_master_params = make_master_params(
            target_model_param_groups_and_shapes
        )
        
        target_ema, scales = ema_scale_fn(global_step)
        with torch.no_grad():
            for targ, src in zip(target_model_master_params, A_to_B_master_params):
                targ.detach().mul_(target_ema).add_(src, alpha=1 - target_ema)

            master_params_to_model_params(
                target_model_param_groups_and_shapes,
                target_model_master_params,
            )

        total_time += time.time()-aa
        
        
        #6: Update progress bar with current loss
        global_step += 1
        progress_bar.set_postfix({'Loss': f'{np.nanmean(A_to_B_loss_sum):.7f}', 'Time': f'{time.time()-aa:.2f}s'})

        
    average_loss = np.nanmean(A_to_B_loss_sum) 
    print('Averaged loss is: '+ str(average_loss))
    return global_step


# Use the window sliding method to denoise the whole PET image. Must used it.
# For example, if your whole image is 96x192, and our window size is 64x64, so the function will automatically sliding down
# the whole image with a certain overlapping ratio

# The window size (img_size) is shown in the "Build the data loader using the monai library" section.
# img_size: the size of sliding window
# img_num: the number of sliding window in each process, only related to your gpu memory, it will still run through the whole volume
# overlap: the overlapping ratio

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
inferer = SlidingWindowInferer(patch_size, Inference_patch_number_each_time, overlap=overlap,
                               mode =mode ,cval = back_ground_intensity, sw_device=device,device = device)

def evaluate(Consistency_network, epoch, checkpoint_dir, data_loader1, best_loss, global_step):
    Consistency_network.eval()
    prediction = []
    true = []
    img = []
    loss_all = []
    aa = time.time()
    with torch.no_grad():
        progress_bar = tqdm(data_loader1, desc=f'Evaluating', position=1, leave=False)
        for i, data in enumerate(progress_bar):
                Low_dose = data['image'].to(device)        
                High_dose = data['label'].to(device)
                
                High_dose_samples = inferer(Low_dose, diffusion_sampling, Consistency_network)  
                loss = metric(High_dose_samples, High_dose)
                progress_bar.set_postfix({'MSE': f'{loss:.7f}'})
                img.append(Low_dose.cpu().numpy())
                true.append(High_dose.cpu().numpy())
                prediction.append(High_dose_samples.cpu().numpy())    
                loss_all.append(loss.cpu().numpy())
        
        avg_loss = np.mean(loss_all)
        print(f'Evaluation completed in {time.time()-aa:.2f}s with average loss: {avg_loss:.7f}')
        return avg_loss

# Enter your data folder
training_set1 = CustomDataset(['/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/dataset/train_mat/'], train_flag=True)
testing_set1 = CustomDataset(['/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/dataset/test_mat/'],train_flag=False)

# Enter your data reader parameters
params = {'batch_size': BATCH_SIZE_TRAIN,
          'shuffle': True,
          'pin_memory': True,
          'drop_last': False}
train_loader1 = torch.utils.data.DataLoader(training_set1, **params)

params = {'batch_size': 320,
          'shuffle': False,
          'pin_memory': True,
          'drop_last': False}
test_loader1 = torch.utils.data.DataLoader(testing_set1, **params)
# Enter your total number of epoch
N_EPOCHS = 500

# Set up checkpoint directory and paths
checkpoint_dir = "/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/checkpoints_newdataset_normalized"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, 'consistency_model_checkpoint.pt')
best_model_path = os.path.join(checkpoint_dir, 'consistency_model_best.pt')

# Initialize training state
best_loss = float('inf')
global_step = 0
train_loss_history, test_loss_history = [], []

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    Consistency_network.load_state_dict(checkpoint['model_state_dict'])
    Consistency_model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    best_loss = checkpoint['best_loss']
    train_loss_history = checkpoint['train_loss_history']
    test_loss_history = checkpoint['test_loss_history']
    print(f'Resuming from global step {global_step} with best loss {best_loss:.7f}')

Consistency_network_model_ema = copy.deepcopy(Consistency_network)
# Create progress bar for epochs
epoch_pbar = tqdm(range(0, N_EPOCHS), desc='Training Progress', position=0, leave=True)

for epoch in epoch_pbar:
    start_time = time.time()
    
    # Training phase
    global_step = train(Consistency_network, Consistency_network_model_ema, Consistency_model_optimizer,
                       train_loader1, consistency, global_step)
    
    # Evaluation phase
    if epoch % 1 == 0:
        average_loss = evaluate(Consistency_network, epoch, checkpoint_dir, test_loader1, best_loss, global_step)
        test_loss_history.append(average_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': Consistency_network.state_dict(),
            'optimizer_state_dict': Consistency_model_optimizer.state_dict(),
            'global_step': global_step,
            'best_loss': best_loss,
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if improved
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(Consistency_network.state_dict(), best_model_path)
            epoch_pbar.set_postfix({'Best Loss': f'{best_loss:.7f}'})
    
    # Update epoch progress bar
    epoch_time = time.time() - start_time
    epoch_pbar.set_postfix({'Loss': f'{average_loss:.7f}', 'Time/epoch': f'{epoch_time:.2f}s'})

print('\nTraining completed!')