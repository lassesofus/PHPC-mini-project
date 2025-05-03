from os.path import join
import sys
from simulate import load_data, jacobi, summary_stats
import matplotlib.pyplot as plt

import numpy as np


# Load and visuazlize the data
if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    # Load floor plans
    all_u0 = np.empty((len(building_ids), 514, 514))
    all_interior_mask = np.empty((len(building_ids), 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids[:4]):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    num_buildings = len(building_ids[:4])
        
    # Determine global min and max for all_u0 for consistent colormap mapping.
    vmin = all_u0.min()
    vmax = all_u0.max()

    # Create a plot with 2 rows and 4 columns.
    fig, axs = plt.subplots(2, num_buildings, figsize=(20, 10))

    for i in range(num_buildings):
        # Plot initial temperature (domain) using same vmin/vmax for consistency.
        im0 = axs[0, i].imshow(all_u0[i], cmap='hot', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Building {building_ids[i]}: Initial Temperature')
        
        # Plot interior mask.
        im1 = axs[1, i].imshow(all_interior_mask[i], cmap='gray')
        axs[1, i].set_title(f'Building {building_ids[i]}: Interior Mask')
    
    # Use tight_layout first to pack subplots.
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Reserve right 10% for the colorbar


    # Remove individual colorbars and add one big colorbar for all u0 images.
    fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.9])  # [left, bottom, width, height]
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label("Initial Temperature", fontsize=12)
    plt.show()
    fig.savefig('floor_plans.png', dpi=300)