import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

def sample_gaussian_ply(
    in_ply: str,
    out_ply: str,
    n_samples: int = 5_000,
    method: str = "random",        # "random" | "opacity" | "fps"
    opacity_field: str = "opacity"  # Adjust field name if necessary
):
    """
    Extract a subset of Gaussians from a PLY file (Gaussian point cloud) and save as a new PLY file.

    Parameters
    ----------
    in_ply : str
        Path to the input PLY file.
    out_ply : str
        Path for the output (sampled) PLY file.
    n_samples : int
        Number of Gaussians to keep. If the input has fewer, all are preserved.
    method : str
        Sampling method:
        - "random" : Uniform random sampling
        - "opacity": Probability proportional to the opacity (or alpha) values
        - "fps"    : Farthest-Point Sampling (maintains spatial diversity)
    opacity_field : str
        Name of the opacity column. Can be changed to "alpha" or another name as needed.
    """
    in_ply = Path(in_ply)
    out_ply = Path(out_ply)
    assert in_ply.exists(), f"{in_ply} not found"

    ply = PlyData.read(str(in_ply))
    verts = ply["vertex"].data  # structured NumPy array

    N = len(verts)
    if N <= n_samples:
        print(f"Input ({N}) <= requested ({n_samples}), copying without sampling.")
        # ply.write(str(out_ply))
        return

    # ── Coordinates (required) ────────────────────────────────────────────
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]])

    # ── Select sampling indices ─────────────────────────────────────────
    if method == "random":
        idx = np.random.choice(N, n_samples, replace=False)

    elif method == "opacity":
        if opacity_field not in verts.dtype.names:
            raise KeyError(f"Opacity field '{opacity_field}' not found in PLY")
        weights = verts[opacity_field].astype(np.float64)
        weights = np.clip(weights, 1e-8, None)  # avoid zeros
        weights /= weights.sum()
        idx = np.random.choice(N, n_samples, replace=False, p=weights)

    elif method == "fps":
        idx = np.zeros(n_samples, dtype=np.int64)
        dists = np.full(N, np.inf)  # distance to nearest selected point
        idx[0] = np.random.randint(N)
        for i in range(1, n_samples):
            dist_new = np.linalg.norm(xyz - xyz[idx[i - 1]], axis=1)
            dists = np.minimum(dists, dist_new)
            idx[i] = np.argmax(dists)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Create new PlyElement ────────────────────────────────────────────
    sampled_verts = verts[idx]
    el = PlyElement.describe(sampled_verts, "vertex")
    PlyData([el], text=ply.text, byte_order=ply.byte_order).write(str(out_ply))
    print(f"✓ {in_ply.name}: {N} → {n_samples} samples, saved to {out_ply}")


# Usage example:
# sample_gaussian_ply(
#     in_ply="input_gaussians.ply",
#     out_ply="sampled_5000.ply",
#     n_samples=5_000,
#     method="fps"
# )
