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
import matplotlib.pyplot as plt


traj = torch.load("data/reach/rollout.pth")
model_ckpt = torch.load("/home/yuxinchen/dino_wm/outputs/2025-02-18/18-59-30/checkpoints/model_21.pth")
norm_params = torch.load("/home/yuxinchen/dino_wm/norm_params_reach.pth")
traj_disturbed = torch.load("data/reach/rollout_disturbed.pth")
img_erased = plt.imread("init_erase_box.png")

img_erased = torch.tensor(img_erased).permute((2, 0, 1)).unsqueeze(0)

# pdb.set_trace()

# save_image(traj['init_obs'][0] / 255, "init_obs_disturbed.png", nrow=1, normalize=True, value_range=(0, 1))


init_obs = TF.resize(img_erased, (224, 224)).unsqueeze(1).repeat(10, 3, 1, 1, 1).cuda() # shape [10, 3, 3, 224, 224], dtype torch.float32
action = traj['action'][:,:,:3]


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
save_image(visuals.reshape(-1, 3, 224, 224), "rollout_erase_box.png", nrow=visuals.shape[1], normalize=True, value_range=(-1, 1))

# visuals = (model.decode_obs(z_obses)[0]["visual"].cpu() * 255).to(torch.int)

# grid = save_image(visuals[0].cpu(), "rollout.png", nrows=len(visuals[0]), normalize=True, value_range=(-1, 1))
# Convert to uint8
# data = visuals.byte()  # Ensure data is uint8 [0,255]

# # Rearrange into a gridc

# rows = []
# for i in range(10):  # Iterate over rollouts
#     row = torch.cat([data[i, j] for j in range(9)], dim=2)  # Concatenate 9 frames horizontally
#     rows.append(row)

# # Stack all rows vertically
# final_image = torch.cat(rows, dim=1)  # Shape: (10*224, 9*224, 3)

# # Convert to NumPy
# final_image = final_image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

# # Save using PIL
# image = Image.fromarray(final_image)
# image.save("rollouts_grid.png")
