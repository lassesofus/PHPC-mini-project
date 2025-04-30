from os.path import join
import sys
from numba import cuda
from time import perf_counter
import numpy as np

SIZE = 512  # grid without the ghost layer
THREADS = (16, 16)
BLOCKS = ((SIZE + THREADS[0] - 1) // THREADS[0],
          (SIZE + THREADS[1] - 1) // THREADS[1])


def load_data(load_dir: str, bid: str):
    """Return padded temperature field and 512×512 interior mask."""
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float32)  # ghost layer around domain
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):
    """Single Jacobi sweep over the padded domain.

    Parameters
    ----------
    u_old, u_new : 2‑D float32 device arrays of shape (SIZE+2, SIZE+2)
        Old and new temperature fields with a one‑cell ghost layer.
    interior_mask : 2‑D bool device array of shape (SIZE, SIZE)
        True for interior fluid cells (offset by −1 relative to u)."""

    i, j = cuda.grid(2)

    if i >= u_old.shape[0] or j >= u_old.shape[1]:
        return

    val = u_old[i, j]

    if 1 <= i < u_old.shape[0] - 1 and 1 <= j < u_old.shape[1] - 1:
        if interior_mask[i - 1, j - 1]:
            val = 0.25 * (
                u_old[i - 1, j] + u_old[i + 1, j] +
                u_old[i, j - 1] + u_old[i, j + 1]
            )
    
    u_new[i, j] = val


def jacobi_cuda(u_host: np.ndarray, interior_mask_host: np.ndarray, max_iter: int):
    """Run *max_iter* Jacobi iterations on the GPU and return the result on host."""

    u_old_dev = cuda.to_device(u_host.astype(np.float32))
    u_new_dev = cuda.device_array_like(u_old_dev)
    mask_dev = cuda.to_device(interior_mask_host)

    for _ in range(max_iter):
        jacobi_kernel[BLOCKS, THREADS](u_old_dev, u_new_dev, mask_dev)
        u_old_dev, u_new_dev = u_new_dev, u_old_dev  # swap device pointers

    # copy_to_host() blocks until the last kernel completes (implicit sync)
    return u_old_dev.copy_to_host()


def summary_stats(u: np.ndarray, interior_mask: np.ndarray):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = float(u_interior.mean())
    std_temp = float(u_interior.std())
    pct_above_18 = float((u_interior > 18).sum() * 100.0 / u_interior.size)
    pct_below_15 = float((u_interior < 15).sum() * 100.0 / u_interior.size)
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


if __name__ == "__main__":
    # ---- I/O ----
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    # ---- Load problem data ----
    all_u0 = np.empty((N, SIZE + 2, SIZE + 2), dtype=np.float32)
    all_interior_mask = np.empty((N, SIZE, SIZE), dtype=bool)

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000

    # ---- Warm‑up run to JIT‑compile and allocate once ----
    _ = jacobi_cuda(all_u0[0], all_interior_mask[0], 1)
    cuda.synchronize()  # make sure compilation & allocation are finished

    # ---- Timed GPU solve ----
    start = perf_counter()

    all_u = np.empty_like(all_u0)
    for i in range(N):
        all_u[i] = jacobi_cuda(all_u0[i], all_interior_mask[i], MAX_ITER)

    cuda.synchronize()  # wait for every building to finish
    end = perf_counter()

    total_time = end - start
    print(f"GPU total time for {N} buildings: {total_time:.4f} s")
    print(f"Average per building: {total_time / N:.4f} s")

    # ---- Statistics ----
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("\nbuilding_id, " + ", ".join(stat_keys))
    for bid, u, mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, mask)
        print(bid + ", " + ", ".join(str(stats[k]) for k in stat_keys))
