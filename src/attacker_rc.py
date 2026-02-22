# attacker_rc.py
"""
Ray-casting / stealth fabrication and removal attacks for CAD experiments.

This module performs three types of operations on the point cloud in the ego frame:
- ray_cast_fabrication: filling a rectangular area with fake points (spoof).
- stealth_fabrication_near_local: making fake points similar to local points (stealthy spoof).
- remove_points: randomly removing part of the points (removal attack).
"""

from typing import Tuple, Optional
import numpy as np


def _rng(seed: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return a numpy Generator (int seed or existing Generator)."""
    
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# region is (x0, y0, w, h) in ego frame (meters)
def ray_cast_fabrication(
    region: Tuple[float, float, float, float],
    n_points: int = 100,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Simple spoofing: fill the region uniformly with fabricated points.

    region: (x0, y0, w, h) in ego frame (meters).
    n_points: Number of fake points.
    """
    rng = _rng(rng)
    x0, y0, w, h = region

    n_points = int(max(0, n_points))
    if n_points == 0:
        return np.zeros((0, 2), dtype=float)

    xs = rng.uniform(x0, x0 + w, size=n_points)
    ys = rng.uniform(y0, y0 + h, size=n_points)
    return np.vstack([xs, ys]).T.astype(float)


def stealth_fabrication_near_local(
    local_pts: np.ndarray,
    region: Tuple[float, float, float, float],
    n_points: int = 100,
    overlap_frac: float = 0.6,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
Stealthier spoofing:

- Some of the fake points are built from existing local points (overlap).
- The rest are generated uniformly inside the area.
- Local points used for overlap are those "in front" and close
to the attack area.

overlap_frac: The fraction of fake points that are derived from local points (0..1).
"""
    rng = _rng(rng)
    local_pts = np.asarray(local_pts, dtype=float)
    x0, y0, w, h = region

   # Reasonable number of points
    n_points = int(max(0, n_points))
    if n_points == 0:
        return np.zeros((0, 2), dtype=float)

  # If we don't have any points, let's go back to the simple ray_cast mode
    if local_pts.size == 0:
        return ray_cast_fabrication(region, n_points=n_points, rng=rng)

 # Clip overlap_frac to [0,1]
    overlap_frac = float(np.clip(overlap_frac, 0.0, 1.0))
    n_overlap = int(round(overlap_frac * n_points))
    n_overlap = max(0, min(n_points, n_overlap))
    n_uniform = n_points - n_overlap

   # Select local points close to the region:
   # - in front of the ego (x > 0)
   # - in the range around the region (in both x and y) with a small margin
    margin_x = max(2.0, 0.5 * w)
    margin_y = max(2.0, 0.5 * h)
    lx = local_pts[:, 0]
    ly = local_pts[:, 1]

    mask = (
        (lx > 0.0) &
        (lx >= x0 - margin_x) & (lx <= x0 + w + margin_x) &
        (ly >= y0 - margin_y) & (ly <= y0 + h + margin_y)
    )
    cand = local_pts[mask]
    if cand.shape[0] == 0:
       # If no candidate is found, we use all local_pts
        cand = local_pts

   # If n_overlap > 0, we iteratively sample cand
    if n_overlap > 0:
        idx = rng.integers(0, cand.shape[0], size=n_overlap)
        base = cand[idx]

        # Map into region (clip) and add small jitter
        bx = np.clip(base[:, 0], x0, x0 + w)
        by = np.clip(base[:, 1], y0, y0 + h)
        base_mod = np.stack([bx, by], axis=1)

       # jitter value proportional to region size
        jitter_scale = 0.05 * max(w, h)  # ~5% of the largest dimension
        base_mod = base_mod + rng.normal(scale=jitter_scale, size=base_mod.shape)

       # If jitter takes them out of the region, we clip again
        base_mod[:, 0] = np.clip(base_mod[:, 0], x0, x0 + w)
        base_mod[:, 1] = np.clip(base_mod[:, 1], y0, y0 + h)
    else:
        base_mod = np.zeros((0, 2), dtype=float)

  # Uniform section inside the region
    if n_uniform > 0:
        uni = ray_cast_fabrication(region, n_points=n_uniform, rng=rng)
        pts = np.vstack([base_mod, uni])
    else:
        pts = base_mod

    return pts.astype(float)


def remove_points(
    points: np.ndarray,
    frac_remove: float = 0.4,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Randomly drop a fraction of existing points (removal attack).

    frac_remove: The fraction of points to remove (0..1).
    
    """
    rng = _rng(rng)
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return points.copy()

    # Clip frac_remove to [0,1]
    frac_remove = float(np.clip(frac_remove, 0.0, 1.0))
    if frac_remove <= 0.0:
        return points.copy()

    n = points.shape[0]
    keep_prob = 1.0 - frac_remove

    # Maintenance mask
    keep = rng.random(size=n) < keep_prob

    if not keep.any():
      # If all are removed, we keep at least one point
        keep[rng.integers(0, n)] = True

    return points[keep].astype(float)
