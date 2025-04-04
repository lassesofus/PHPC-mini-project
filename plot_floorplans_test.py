import numpy as np
import matplotlib.pyplot as plt

# Load the processed floorplans.
all_u = np.load('test_floorplans.npy')
N = all_u.shape[0]

# Set the grid dimensions (3 rows x 4 columns).
rows, cols = 3, 4

# Determine the global min and max for consistent colormap mapping.
vmin = all_u.min()
vmax = all_u.max()

# Create a figure with subplots.
fig, axs = plt.subplots(rows, cols, figsize=(20, 15))

# Flatten the axes array for easier iteration.
axs = axs.flatten()

# Plot each floorplan on the grid.
for i in range(rows * cols):
    ax = axs[i]
    if i < N:
        im = ax.imshow(all_u[i], cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Floorplan {i+1}')
    else:
        # Turn off any unused subplots.
        ax.axis('off')
    # Remove axis ticks.
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout to leave space on the right for a common colorbar.
fig.subplots_adjust(right=0.87, wspace=0.3, hspace=0.3)
cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Temperature')

plt.show()

# Optionally, save the figure.
fig.savefig('floor_plans.png', dpi=300)