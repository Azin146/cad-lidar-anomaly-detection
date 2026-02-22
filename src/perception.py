# perception.py
"""
Simple local perception model for the ego vehicle.

local_perception(ego_pose, obstacles, max_range, jitter, dropout_prob,
                 loc_error_std, range_drop, rng) -> (M, 2) array of points
in the ego frame, mimicking a 2D LiDAR scan.
"""

from typing import Optional, Union
import numpy as np


def _rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """
    Small helper: if we already have a Generator, just return it,
    otherwise create a new one from the given seed (or from entropy).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def local_perception(
    ego_pose,
    obstacles,
    max_range: float = 50.0,
    jitter: float = 0.02,
    dropout_prob: float = 0.0,
    loc_error_std: float = 0.0,
    range_drop: float = 0.0,
    rng: Optional[np.random.Generator] = None,
):
    """
    Simulate a 2D LiDAR-like scan of obstacles around ego.

    Parameters
    ----------
    ego_pose : (2,) array-like
        Ego position [x, y] in world/ego frame.
    obstacles : (N, 2) array-like
        Obstacle centers [x, y] in the same frame.
    max_range : float
        Maximum sensing range (meters).
    jitter : float
        Standard deviation of measurement noise (meters) for each beam hit.
    dropout_prob : float
        Base probability to drop each obstacle independently (0..1).
    loc_error_std : float
        Standard deviation of ego pose perturbation (meters) – simulates
        global localization error (rigid shift).
    range_drop : float
        Additional dropout that increases linearly with distance / max_range.
        E.g. 0.5 means up to 50% extra dropout for far obstacles.
    rng : np.random.Generator or None
        Random generator (for reproducibility).

    Returns
    -------
    pts : (M, 2) ndarray
        Simulated LiDAR points in the ego frame.
    """
    rng = _rng(rng)

    obstacles = np.asarray(obstacles, dtype=float)
    if obstacles.size == 0:
        return np.zeros((0, 2), dtype=float)

    ego_pose = np.asarray(ego_pose, dtype=float).reshape(1, 2)

    # --- ego localization error: rigid shift of perceived scene ---
    if loc_error_std > 0.0:
        ego_err = rng.normal(scale=loc_error_std, size=(1, 2))
    else:
        ego_err = np.zeros((1, 2), dtype=float)

    # positions relative to (possibly mis-localized) ego
    rel = obstacles - (ego_pose + ego_err)  # (N,2)
    dists = np.linalg.norm(rel, axis=1)

    # --- field-of-view / max range filter ---
    in_range = dists <= max_range
    rel = rel[in_range]
    dists = dists[in_range]
    if rel.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)

    # --- obstacle-level dropout (independent) ---
    if dropout_prob > 0.0:
        keep_mask = rng.random(size=rel.shape[0]) >= dropout_prob
        rel = rel[keep_mask]
        dists = dists[keep_mask]
        if rel.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)

    # --- extra dropout for far obstacles (range-based) ---
    if range_drop > 0.0:
        # probability grows linearly with normalized distance
        p_extra = np.clip(range_drop * (dists / max_range), 0.0, 1.0)
        keep_mask = rng.random(size=rel.shape[0]) >= p_extra
        rel = rel[keep_mask]
        dists = dists[keep_mask]
        if rel.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)

    # --- generate multiple LiDAR hits per remaining obstacle ---
    # this makes occupancy maps denser and improves IoU with GT boxes
    pts_list = []
    for p in rel:
        # number of beams hitting this obstacle (a small cluster)
        n_hits = int(rng.integers(6, 15))  # 6..14 points per obstacle
        if jitter > 0.0:
            noise = rng.normal(scale=jitter, size=(n_hits, 2))
        else:
            noise = 0.0
        cluster = p.reshape(1, 2) + noise
        pts_list.append(cluster)

    if not pts_list:
        return np.zeros((0, 2), dtype=float)

    pts = np.vstack(pts_list).astype(float)
    return pts
