from os.path import join
import sys
import cupy as cp

SIZE = 512  # grid interior size


def load_data(load_dir, bid):
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_batch(u, interior_mask, max_iter, atol=1e-6):
    """
    Batched Jacobi solver with per-image convergence.
    - u:    (B, SIZE+2, SIZE+2)
    - interior_mask: (B, SIZE, SIZE) boolean
    Returns u after each image has individually converged.
    """
    B = u.shape[0]
    # All images start “active”
    active = cp.ones((B,), dtype=bool)

    for _ in range(max_iter):
        # neighbor-average for all
        u_center = u[:, 1:-1, 1:-1]          # (B, SIZE, SIZE)
        u_new = 0.25 * (
            u[:, 1:-1, :-2] +   # left
            u[:, 1:-1, 2:]  +   # right
            u[:, :-2, 1:-1] +   # up
            u[:, 2:, 1:-1]      # down
        )

        # compute per-image max change over interior
        diff = cp.abs(u_new - u_center) * interior_mask
        # shape (B,): max over SIZE×SIZE for each image
        deltas = diff.reshape(B, -1).max(axis=1)

        # figure out which images still need more iterations
        still_active = deltas > atol
        if not still_active.any():
            # everyone converged!
            break

        # update only interior points of still-active images
        # build a (B,SIZE,SIZE) mask: interior & still_active[:,None,None]
        update_mask = interior_mask & still_active[:, None, None]
        u[:, 1:-1, 1:-1] = cp.where(update_mask, u_new, u_center)

        # update our active array for next iter
        active = still_active

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp':    float(u_interior.mean()),
        'std_temp':     float(u_interior.std()),
        'pct_above_18': float((u_interior > 18).sum() / u_interior.size * 100),
        'pct_below_15': float((u_interior < 15).sum() / u_interior.size * 100),
    }


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # N = total buildings, B_size = batch size
    N      = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    B_size = int(sys.argv[2]) if len(sys.argv) >= 3 else N
    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL  = 1e-4

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))

    for start in range(0, N, B_size):
        chunk = building_ids[start:start + B_size]
        B = len(chunk)

        # load batch
        u_batch    = cp.empty((B, SIZE+2, SIZE+2), dtype=cp.float64)
        mask_batch = cp.empty((B, SIZE, SIZE),    dtype=bool)
        for i, bid in enumerate(chunk):
            u0, mask = load_data(LOAD_DIR, bid)
            u_batch[i]    = u0
            mask_batch[i] = mask

        # batched solve with per-image convergence
        u_batch = jacobi_batch(u_batch, mask_batch, MAX_ITER, ABS_TOL)

        # output stats
        for bid, u_fin, mask in zip(chunk, u_batch, mask_batch):
            stats = summary_stats(u_fin, mask)
            vals  = [stats[k] for k in stat_keys]
            print(f"{bid}," + ",".join(f"{v:.6f}" for v in vals))