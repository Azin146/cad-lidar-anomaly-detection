# occupancy_map.py
"""
Unified occupancy map generator for CAD experiments.
- Real resolution (0.2m)
- Bounds consistent with simulation.py → (-5, 40), (-10, 10)
- Center-aligned map: (0,0) is exactly in the middle of the image
- IoU is calculated perfectly with attack_region

"""

import numpy as np

# Must be consistent with simulation.py
DEFAULT_BOUNDS = (-5.0, 40.0, -10.0, 10.0)
DEFAULT_RES = 0.2   # each pixel = 0.2 meters


def build_occupancy_map(points,
                        bounds=DEFAULT_BOUNDS,
                        resolution=DEFAULT_RES):
    """
    Convert (N,2) point cloud in ego frame to a 2D occupancy grid.

    Parameters
    ----------
    points : array of shape (N,2)
        LiDAR or fused points in ego coordinates (meters)
    bounds : (xmin, xmax, ymin, ymax)
        meter limits of the grid
    resolution : float
        meter per pixel

    Returns
    -------
    occ : uint8 array of shape (H, W)
        binary occupancy map, origin is at (0,0) ego → center of map
    """
    points = np.asarray(points, dtype=float)

    xmin, xmax, ymin, ymax = bounds

   # Map dimensions in pixels
    width_m = xmax - xmin      # in the x direction
    height_m = ymax - ymin     # in the y direction

    W = int(np.ceil(width_m / resolution))
    H = int(np.ceil(height_m / resolution))

    occ = np.zeros((H, W), dtype=np.uint8)

    if points.size == 0:
        return occ

    xs = points[:, 0]
    ys = points[:, 1]

    # Only points within the map area
    inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    xs = xs[inside]
    ys = ys[inside]

    if xs.size == 0:
        return occ

# Convert meters → pixels
# (xmin, ymin) maps to index (0, 0)
# origin="lower" in matplotlib makes v=yindex
    u = ((xs - xmin) / resolution).astype(int)
    v = ((ys - ymin) / resolution).astype(int)

   # Preventing expulsion
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    occ[v, u] = 1
    return occ
