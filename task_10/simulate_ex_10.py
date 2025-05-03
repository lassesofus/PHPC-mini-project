from os.path import join
import sys

import cupy as cp

def load_data(load_dir, bids):
    """Return padded domains and masks stacked along axis-0 (N, 514, 514)."""
    SIZE = 512
    u0   = cp.zeros((len(bids), SIZE + 2, SIZE + 2), dtype=cp.float32)
    mask = cp.empty((len(bids), SIZE, SIZE), dtype=cp.bool_)

    for k, bid in enumerate(bids):
        u0[k, 1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
        mask[k]           = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u0, mask


def jacobi_batched(u0, mask, max_iter=20_000, atol=1e-4):
    """
    u0   : (B, 514, 514)  padded domains (float32)
    mask : (B, 512, 512)  interior masks (bool)
    """
    u     = u0.copy()
    inner = u[:, 1:-1, 1:-1]           # view of the interior

    for _ in range(max_iter):
        # 1) 4-point average
        avg = 0.25 * (
              u[:, 1:-1, :-2] + u[:, 1:-1,  2:]
            + u[:, :-2,  1:-1] + u[:,  2:,  1:-1]
        )

        # 2) residual BEFORE overwriting
        diff = cp.abs(avg - inner) * mask          # element-wise kernel

        # 3) masked in-place update
        cp.copyto(inner, avg, where=mask)          # element-wise kernel

        # 4) convergence check (sync once per sweep)
        if diff.max().item() < atol:
            break
    return u


def summary_stats_batched(u, mask):
    """
    u    : (B, 514, 514)  full padded fields
    mask : (B, 512, 512)  interior cells == True
    returns four (B,) CuPy vectors: mean, std, %>18, %<15
    """
    interior = u[:, 1:-1, 1:-1]            # (B,512,512) view
    n_cells   = mask.sum(axis=(1, 2))      # number of interior cells per bld.

    # mean
    sum_vals  = (interior * mask).sum(axis=(1, 2))
    mean      = sum_vals / n_cells

    # std  (sqrt(E[x²] – E[x]²))
    sum_sq    = (interior**2 * mask).sum(axis=(1, 2))
    std       = cp.sqrt(sum_sq / n_cells - mean**2)

    # percentage above / below thresholds
    pct_above_18 = ((interior > 18) & mask).sum(axis=(1, 2)) / n_cells * 100
    pct_below_15 = ((interior < 15) & mask).sum(axis=(1, 2)) / n_cells * 100

    return mean, std, pct_above_18, pct_below_15


if __name__ == "__main__":
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    bids     = open(join(LOAD_DIR, "building_ids.txt")).read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bids = bids[:N]

    u0, mask = load_data(LOAD_DIR, bids)
    u  = jacobi_batched(u0, mask)

    cp.cuda.Device().synchronize()     # make sure GPU is done

    mean, std, p18, p15 = summary_stats_batched(u, mask)

    print("building_id, mean_temp, std_temp, pct_above_18, pct_below_15")
    for i, bid in enumerate(bids):
        print(f"{bid}, {mean[i].item():.4f}, {std[i].item():.4f}, "
              f"{p18[i].item():.2f}, {p15[i].item():.2f}")