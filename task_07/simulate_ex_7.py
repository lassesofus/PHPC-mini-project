from os.path import join
import sys
from multiprocessing import Pool
import math
from numba import jit, prange
import matplotlib.pyplot as plt

import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    SIZE = u.shape[0] - 2  # 512

    for it in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        max_delta = 0.0
        for i in prange(SIZE):
            for j in range(SIZE):
                if interior_mask[i, j]:
                    delta = abs(u[1+i, 1+j] - u_new[i, j])
                    u[1+i, 1+j] = u_new[i, j]
                    if delta > max_delta:
                        max_delta = delta
        if max_delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

def process_floorplan(task):
    # Each task is a tuple (u0, interior_mask, max_iter, atol)
    u0, interior_mask, max_iter, atol = task
    return jacobi(u0, interior_mask, max_iter, atol)

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    num_processes = int(sys.argv[1])

    # Use at most 100 floorplans for experiments.
    N = min(len(building_ids), 100)
    # N = min(len(building_ids), 10)
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    tasks = []
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
        # Prepare each task with chosen Jacobi parameters.
        tasks.append((u0, interior_mask, 20000, 1e-4))

    # Static scheduling: determine a chunk size so that tasks are evenly distributed.
    chunk_size = math.ceil(N / num_processes)
    print(f"Processing {N} floorplans on {num_processes} processes (chunk size = {chunk_size})")
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_floorplan, tasks, chunksize=1)
        results = list(results)

    all_u = np.array(results)
    print(f"All floorplans processed. Output shape: {all_u.shape}")
    
    # Print summary statistics in CSV format.
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    all_stats = np.empty((N, len(stat_keys)))
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        all_stats[i] = [stats[k] for k in stat_keys]
        stats_str = ", ".join(str(stats[k]) for k in stat_keys)
        print(f"{bid}, {stats_str}")


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, key in enumerate(stat_keys):
    axs[idx].hist(all_stats[:, idx], bins=20, edgecolor='black')
    axs[idx].set_title(f"Histogram of {key}")
    axs[idx].set_xlabel(key)
    axs[idx].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("sim_ex_7.png")
plt.show()