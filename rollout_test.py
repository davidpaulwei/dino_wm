import torch
from PIL import Image
import pdb

from wandb import init
from models.visual_world_model import VWorldModel
import torchvision.transforms.functional as TF
from models.dino import DinoV2Encoder
from einops import rearrange
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import slice_trajdict_with_t


# action = torch.load("/home/yuxinchen/dino_wm/data/reach/actions.pth")[:1, :40]
# proprio = torch.load("/home/yuxinchen/dino_wm/data/reach/states.pth")[:1, :15:5]
# init_obs = torch.load("/home/yuxinchen/dino_wm/data/reach/obses/episode_000.pth")[:15:5].permute(0, 3, 1, 2)
# init_obs = TF.resize(init_obs, (224, 224)).unsqueeze(0).cuda() / 255
# init_obs = init_obs.to(torch.float32)





model_ckpt = torch.load("/home/yuxinchen/dino_wm/outputs/2025-02-18/18-59-30/checkpoints/model_21.pth")
# norm_params = torch.load("/home/yuxinchen/dino_wm/norm_params_reach.pth")



# init_obs = TF.resize(traj['init_obs'], (224, 224)).unsqueeze(1).repeat(1, 3, 1, 1, 1).cuda() / 255 # shape [10, 3, 3, 224, 224], dtype torch.float32

# action = (action - norm_params['action_mean']) / norm_params['action_std']
# proprio = (proprio - norm_params['proprio_mean']) / norm_params['proprio_std']

# action = rearrange(action, "b (h f) d -> b h (f d)", f=5) # shape [10, 8, 15], dtype torch.float32


# action = action.to(torch.float32).cuda()
# proprio = proprio.to(torch.float32).cuda()

obs_0 = torch.load("/home/yuxinchen/dino_wm/obs_0.pth")
actions = torch.load("/home/yuxinchen/dino_wm/actions.pth")


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


        # utils.save_image(
        #     imgs,
        #     img_name,
        #     nrow=num_columns,
        #     normalize=True,
        #     value_range=(-1, 1),
        # )

# z_obses, z = model.rollout(obs_0={"visual": init_obs, "proprio": proprio}, act=action)
# z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)

z_obses, z = model.rollout(obs_0=obs_0, act=actions)
z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)


visuals = model.decode_obs(z_obses)[0]["visual"].cpu()

pdb.set_trace()

save_image(visuals[0], "rollouts_grid_test.png", nrow=len(visuals), normalize=True, value_range=(-1, 1))

# # grid = save_image(visuals[0].cpu(), "rollout.png", nrows=len(visuals[0]), normalize=True, value_range=(-1, 1))
# # Convert to uint8
# data = visuals.byte()  # Ensure data is uint8 [0,255]

# # Rearrange into a grid
# rows = []
# for i in range(1):  # Iterate over rollouts
#     row = torch.cat([data[i, j] for j in range(9)], dim=2)  # Concatenate 9 frames horizontally
#     rows.append(row)

# # Stack all rows vertically
# final_image = torch.cat(rows, dim=1)  # Shape: (10*224, 9*224, 3)

# # Convert to NumPy
# final_image = final_image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

# # Save using PIL
# image = Image.fromarray(final_image)
# image.save("rollouts_grid_test.png")