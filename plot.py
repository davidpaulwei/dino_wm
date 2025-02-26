import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch

traj = torch.load("data/reach/rollout.pth")
actions = traj['action'][:,:,:3].numpy()


# Define sinusoidal noise parameters
freq = 2 * np.pi / 15  # One full cycle over 48 steps
amplitude = .05 # Adjust as needed

# Generate sinusoidal noise
timesteps = np.arange(15)
sin_noise = amplitude * np.sin(freq * timesteps)  # Shape [15]
sin_noise = sin_noise[None, :, None]  # Reshape to [1, 15, 1] for broadcasting

# Add noise to actions
actions = actions + np.concatenate((sin_noise, np.zeros((1, 15, 2))), axis=2)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory as a separate line
for i in range(actions.shape[0]):  # Loop over trajectories
    ax.plot(actions[i, :, 0], actions[i, :, 1], actions[i, :, 2], label=f'Traj {i+1}')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Action Trajectories')

# Show legend
ax.legend()

# Display the plot
# Ensure equal scaling for all axes
x_limits = [actions[:, :, 0].min(), actions[:, :, 0].max()]
y_limits = [actions[:, :, 1].min(), actions[:, :, 1].max()]
z_limits = [actions[:, :, 2].min(), actions[:, :, 2].max()]
max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]) / 2

mid_x = (x_limits[1] + x_limits[0]) / 2
mid_y = (y_limits[1] + y_limits[0]) / 2
mid_z = (z_limits[1] + z_limits[0]) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.view_init(elev=45, azim=10) 

plt.savefig("small_sin_action.png")


