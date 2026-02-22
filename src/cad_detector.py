# cad_detector.py
import numpy as np
from scipy.ndimage import label


def _to_binary_occupied(arr) -> np.ndarray:
    """
Convert an arbitrary occupancy map to a binary occupied mask.

Any value > 0 is considered occupied, to be compatible with maps that have
e.g. 0/1/2 or float.

"""
    a = np.asarray(arr)
    return (a > 0).astype(np.uint8)


def detect_inconsistency(local_map, fused_map, min_ratio=0.05, **kwargs):
    """
Simple CAD baseline:

- diff = cells occupied only in fused map
(occupied in fused AND free/empty in local)
- ratio = (# diff) / (# occupied in fused)
- suspicious if ratio >= min_ratio

returns:
suspicious : bool
ratio : float

"""
   # Convert to binary to make it compatible with different map types
    local_occ = _to_binary_occupied(local_map)
    fused_occ = _to_binary_occupied(fused_map)

    if local_occ.shape != fused_occ.shape:
        raise ValueError("local_map and fused_map must have same shape")

    diff = (fused_occ == 1) & (local_occ == 0)
    n_fake_cells = int(diff.sum())
    n_occ_fused = int(fused_occ.sum())

    if n_occ_fused == 0:
        # We have no occupied cells in fused → neither suspicious, nor ratio 0
        return False, 0.0

    ratio = n_fake_cells / float(n_occ_fused)
    suspicious = ratio >= float(min_ratio)
    return bool(suspicious), float(ratio)


def detect_inconsistency_enhanced(
    local_map,
    fused_map,
    min_ratio: float = 0.05,
    min_cluster_size: int = 3,
    distance_threshold: float = 0.5,
    map_resolution: float = 0.2,
):
    """
Enhanced CAD (paper-like version):

1) diff = fused_only = cells that are occupied in fused but empty in local.
2) We cluster (connected-component) on diff
with 8 connections (structure 3x3).
3) Clusters that:
- have size < min_cluster_size or
- have distance from cluster center to ego < distance_threshold
are ignored.
4) ratio = (# cells in suspicious clusters) / (# total occupied cells in fused)

If ratio >= min_ratio → the scene is considered suspicious.

Returns:
suspicious : bool
ratio : float

"""
    local_occ = _to_binary_occupied(local_map)
    fused_occ = _to_binary_occupied(fused_map)

    if local_occ.shape != fused_occ.shape:
        raise ValueError("local_map and fused_map must have same shape")

    # fused-only occupancy
    diff = (fused_occ == 1) & (local_occ == 0)
    n_occ_fused = int(fused_occ.sum())
    if n_occ_fused == 0:
        return False, 0.0

    # 8-connected components (3x3 full structure = eight-connected)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_clusters = label(diff.astype(np.uint8), structure=structure)

    h, w = local_occ.shape
    suspicious_cells = 0
    suspicious_clusters = 0

   # Center of Ego coordinates on the map (assumption: Ego is located in the center of the map)
    cx_ego = w / 2.0
    cy_ego = h / 2.0

    for cid in range(1, n_clusters + 1):
        mask = (labeled == cid)
        size = int(mask.sum())
        if size < int(min_cluster_size):
            continue

        ys, xs = np.nonzero(mask)
       # Geometric center of the cluster in map coordinates
        cx = float(xs.mean())
        cy = float(ys.mean())

       # Convert to meters based on map_resolution
        dx = (cx - cx_ego) * map_resolution
        dy = (cy - cy_ego) * map_resolution
        dist = np.sqrt(dx * dx + dy * dy)

        # We discard clusters that are too close to the ego
        if dist < float(distance_threshold):
            continue

        suspicious_clusters += 1
        suspicious_cells += size

    if suspicious_clusters == 0 or suspicious_cells == 0:
       # We have no suspicious clusters → ratio zero and not suspicious
        return False, 0.0

    ratio = suspicious_cells / float(n_occ_fused)
    suspicious = ratio >= float(min_ratio)
    return bool(suspicious), float(ratio)
