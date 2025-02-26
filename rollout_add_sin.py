from calendar import c
import torch
from PIL import Image
import pdb
from models.visual_world_model import VWorldModel
import torchvision.transforms.functional as TF
from models.dino import DinoV2Encoder
from einops import rearrange
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import slice_trajdict_with_t
import numpy as np


traj = torch.load("data/reach/rollout.pth")
model_ckpt = torch.load("/home/yuxinchen/dino_wm/outputs/2025-02-18/18-59-30/checkpoints/model_21.pth")
norm_params = torch.load("/home/yuxinchen/dino_wm/norm_params_reach.pth")


save_image(traj['init_obs'][0] / 255, "init_obs.png", nrow=1, normalize=True, value_range=(0, 1))
# save_image(traj['init_obs'][0] / 255, "init_obs_disturbed.png", nrow=1, normalize=True, value_range=(0, 1))


init_obs = TF.resize(traj['init_obs'], (224, 224)).unsqueeze(1).repeat(1, 3, 1, 1, 1).cuda() / 255 # shape [10, 3, 3, 224, 224], dtype torch.float32
action = traj['action'][:,:,:3].numpy()

# Define sinusoidal noise parameters
freq = 2 * np.pi / 15  # One full cycle over 48 steps
amplitude =  .3 # Adjust as needed

# Generate sinusoidal noise
timesteps = np.arange(15)
sin_noise = amplitude * np.sin(freq * timesteps)  # Shape [15]
sin_noise = sin_noise[None, :, None]  # Reshape to [1, 15, 1] for broadcasting

# Add noise to actions
action = action + np.concatenate((sin_noise, np.zeros((1, 15, 2))), axis=2)

action = torch.tensor(action).to(torch.float32)

proprio = traj['state'].unsqueeze(1).repeat(1, 3, 1).to(torch.float32) # shape [10, 3, 24], dtype torch.float32

action = (action - norm_params['action_mean']) / norm_params['action_std']
proprio = (proprio - norm_params['proprio_mean']) / norm_params['proprio_std']

action = F.pad(action, (0, 0, 10, 40-action.shape[1]-10))
action = rearrange(action, "b (h f) d -> b h (f d)", f=5) # shape [10, 8, 15], dtype torch.float32


action = action.cuda()
proprio = proprio.cuda()

model = VWorldModel(
    image_size=224,
    num_hist=3,
    num_pred=1,
    encoder=DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens"),
    proprio_encoder=model_ckpt["proprio_encoder"],
    action_encoder=model_ckpt["action_encoder"],
    decoder=model_ckpt["decoder"],
    predictor=model_ckpt["predictor"],
    proprio_dim=10,
    action_dim=10,
    concat_dim=1,
    num_action_repeat=1,
    num_proprio_repeat=1,
).cuda()


z_obses, z = model.rollout(obs_0={"visual": init_obs, "proprio": proprio}, act=action)
z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)

visuals = model.decode_obs(z_obses)[0]["visual"].cpu()
save_image(visuals.reshape(-1, 3, 224, 224), "rollouts_add_big_sin.png", nrow=visuals.shape[1], normalize=True, value_range=(-1, 1))

