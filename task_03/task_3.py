import numpy as np
import matplotlib.pyplot as plt

# Load the processed floorplans.
all_u = np.load('test_floorplans.npy')
# Load the building IDs.
with open('test_floorplans_ids.txt', 'r') as f:
    building_ids = f.read().splitlines()
N = all_u.shape[0]

# Set the grid dimensions (1 row x 4 columns).
rows, cols = 1, 4

# Determine global min and max for consistent colormap mapping.
vmin = all_u.min()
vmax = all_u.max()

# Create a figure with subplots (adjust figsize to reduce white space).
fig, axs = plt.subplots(rows, cols, figsize=(20, 8))
axs = axs.flatten()

# Plot each floorplan on the grid.
for i in range(rows * cols):
    ax = axs[i]
    if i < N:
        im = ax.imshow(all_u[i], cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Building {building_ids[i]}', fontsize=16)
    else:
        ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout with a smaller bottom margin.
plt.tight_layout()

# Add a horizontal colorbar just below the plots with reduced height.
cbar_ax = fig.add_axes([0.01, 0.08, 0.98, 0.08])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Temperature')

# Increase tick label size.
cbar.ax.tick_params(labelsize=14)

# Increase colorbar label size.
cbar.ax.xaxis.label.set_size(16)

plt.show()

# Optionally, save the figure.
fig.savefig('floor_plans_simulation.png', dpi=300)