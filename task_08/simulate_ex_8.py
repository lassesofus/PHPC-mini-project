from os.path import join
import sys
from numba import cuda
from time import perf_counter
import numpy as np
import os

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):

    i, j = cuda.grid(2)

    if i >= u_old.shape[0] or j >= u_old.shape[1]:
        return

    val = u_old[i, j]

    # Interior nodes live in 1..SIZE inclusive (exclude ghost cells 0 and SIZE+1)
    if 1 <= i < u_old.shape[0] - 1 and 1 <= j < u_old.shape[1] - 1:
        if interior_mask[i - 1, j - 1]:  # mask aligns with interior region
            val = 0.25 * (
                u_old[i - 1, j] + u_old[i + 1, j] +
                u_old[i, j - 1] + u_old[i, j + 1]
            )
    
    u_new[i, j] = val
        
def jacobi_cuda(u_host, interior_mask_host, max_iter):
    u_old_dev = cuda.to_device(u_host)
    u_new_dev = cuda.device_array_like(u_old_dev)
    mask_dev = cuda.to_device(interior_mask_host)

    threadsperblock = (16, 16)
    blockspergrid = ((u_old_dev.shape[0] + 15)//16,
                     (u_old_dev.shape[1] + 15)//16)

    for _ in range(max_iter):
        jacobi_kernel[blockspergrid, threadsperblock](u_old_dev, u_new_dev, mask_dev)
        u_old_dev, u_new_dev = u_new_dev, u_old_dev

    return u_old_dev.copy_to_host()



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


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    all_u = np.empty_like(all_u0)

    # --- Warmup: compile & allocate ---
    _ = jacobi_cuda(all_u0[0], all_interior_mask[0], 1)
    cuda.synchronize()

    # --- Timing the GPU run ---
    start = perf_counter()
    for i, (u0, mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi_cuda(u0, mask, MAX_ITER)
    cuda.synchronize()
    end = perf_counter()

    total_time = end - start
    print(f"GPU total time for {N} buildings: {total_time:.4f} s")
    print(f"Average per building: {total_time/N:.4f} s")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nbuilding_id, ' + ', '.join(stat_keys))
    for bid, u, mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    # Save the processed floorplans an their building ids
    np.save('test_floorplans.npy', all_u)
    with open('test_floorplans_ids.txt', 'w') as f:
        for bid in building_ids:
            f.write(f"{bid}\n")
    print(f"Saved processed floorplans to 'test_floorplans.npy' and ids to 'test_floorplans_ids.txt'")