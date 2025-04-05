from os.path import join
import sys
from multiprocessing import Pool
import math

import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
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
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    num_processes = int(sys.argv[1])

    # Use at most 100 floorplans for experiments.
    N = min(len(building_ids), 100)
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
        results = pool.map(process_floorplan, tasks, chunk_size)

    all_u = np.array(results)
    print(f"All floorplans processed. Output shape: {all_u.shape}")
    
    # Print summary statistics in CSV format.
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        stats_str = ", ".join(str(stats[k]) for k in stat_keys)
        print(f"{bid}, {stats_str}")