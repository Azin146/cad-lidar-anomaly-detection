# simulation.py
"""
Scenario generator for CAD experiments (spoof / remove attacks).

Scenario format:
{
  "ego": {"id": "ego", "pose": np.array([x, y])},
  "attacker": {"id": "attacker", "pose": np.array([x, y])},
  "obstacles": np.ndarray shape (N, 2)   # in ego/world frame, meters
  "attack_region": (x0, y0, w, h),       # rectangle in ego frame (meters)
  "target": {"id": "target", "bbox": (x0, y0, w, h)},
  "bounds": (xmin, xmax, ymin, ymax),
  "meta": {...}
}
"""

from typing import Optional, Tuple, List, Union
import numpy as np

# Global map bounds (meters) in ego frame
# Must be the same as occupancy_map and compute_iou_with_gt
DEFAULT_BOUNDS = (-5.0, 40.0, -10.0, 10.0)  # xmin, xmax, ymin, ymax


def _rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """Return a numpy Generator. Accepts int seed or an existing Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def make_attack_region(
    attacker_pose: Union[np.ndarray, Tuple[float, float]],
    dist_ahead: float = 6.0,
    width: float = 4.0,
    height: float = 4.0,
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
) -> Tuple[float, float, float, float]:
    
    """
Build a rectangular attack/target region in front of the attacker.

- Place the center of the box dist_ahead meters ahead of the attacker on the x-axis
- Then clip the center so that the entire box is inside the bounds

Returns (x0, y0, w, h) in meters (bottom-left corner, ego frame).

"""
    ax, ay = np.asarray(attacker_pose, dtype=float)
    xmin, xmax, ymin, ymax = bounds

    half_w = width / 2.0
    half_h = height / 2.0

   # Initial center: dist_ahead in front of the attacker
    cx = ax + dist_ahead
    cy = ay

   # Constrain the center so that the entire box is inside the map
    cx = np.clip(cx, xmin + half_w, xmax - half_w)
    cy = np.clip(cy, ymin + half_h, ymax - half_h)

    x0 = cx - half_w
    y0 = cy - half_h
    return float(x0), float(y0), float(width), float(height)


def example_scenario(
    preset: str = "baseline",
    n_obstacles_range: Tuple[int, int] = (1, 4),
    attacker_dist_range: Tuple[float, float] = (8.0, 18.0),
    attacker_lat_range: Tuple[float, float] = (-2.0, 2.0),
    obstacle_dist_range: Tuple[float, float] = (12.0, 28.0),
    obstacle_lat_range: Tuple[float, float] = (-4.0, 4.0),
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> dict:
    """
Create a single randomized scenario (paper-like but simplified).

Presets:
- 'baseline' : default ranges
- 'near' : attacker slightly closer
- 'far' : attacker further
- 'dense' : more obstacles, wider lateral spread
- 'sparse' : fewer obstacles

NOTE:
- attack_region and target["bbox"] are generated based on attacker and within DEFAULT_BOUNDS
.
- A target obstacle is added in the center of attack_region (with some noise) so
that
the occupancy map always has some real occupancy inside the bbox and IoU
is meaningful.

"""
    rng = _rng(seed)

    # ---- presets ----
    if preset == "near":
        attacker_dist_range = (6.0, 10.0)
    elif preset == "far":
        attacker_dist_range = (15.0, 25.0)
    elif preset == "dense":
        n_obstacles_range = (3, 7)
        obstacle_lat_range = (-6.0, 6.0)
    elif preset == "sparse":
        n_obstacles_range = (1, 2)

    # ego at origin
    ego = {"id": "ego", "pose": np.array([0.0, 0.0], dtype=float)}

    # attacker pose (ahead of ego)
    a_dist = float(rng.uniform(*attacker_dist_range))
    a_lat = float(rng.uniform(*attacker_lat_range))
    attacker = {"id": "attacker", "pose": np.array([a_dist, a_lat], dtype=float)}

    # number of random background obstacles
    n_obs = int(rng.integers(n_obstacles_range[0], n_obstacles_range[1] + 1))

    if n_obs > 0:
        dists = rng.uniform(obstacle_dist_range[0], obstacle_dist_range[1], size=n_obs)
        lats = rng.uniform(obstacle_lat_range[0], obstacle_lat_range[1], size=n_obs)
        obstacles = np.vstack([dists, lats]).T.astype(float)
    else:
        obstacles = np.zeros((0, 2), dtype=float)

    # clamp to bounds
    xmin, xmax, ymin, ymax = bounds
    if obstacles.shape[0] > 0:
        obstacles[:, 0] = np.clip(obstacles[:, 0], xmin, xmax)
        obstacles[:, 1] = np.clip(obstacles[:, 1], ymin, ymax)

    # attack / target rectangle (in front of attacker and inside bounds)
    attack_region = make_attack_region(
        attacker["pose"],
        dist_ahead=6.0,
        width=4.0,
        height=4.0,
        bounds=bounds,
    )

    # --- add a dedicated "target" obstacle inside attack_region ---
    x0_t, y0_t, w_t, h_t = attack_region
    # مرکز باکس
    target_center = np.array([x0_t + w_t / 2.0, y0_t + h_t / 2.0], dtype=float)
  # Small noise (about 30 cm) for variety
    target_jitter = rng.normal(scale=0.3, size=(1, 2))
    target_obstacle = target_center.reshape(1, 2) + target_jitter

    if obstacles.size == 0:
        obstacles = target_obstacle
    else:
        obstacles = np.vstack([obstacles, target_obstacle])

    target = {"id": "target", "bbox": attack_region}

    return {
        "ego": ego,
        "attacker": attacker,
        "obstacles": obstacles,
        "attack_region": attack_region,
        "target": target,
        "bounds": bounds,
        "meta": {
            "preset": preset,
            "seed": None if isinstance(seed, np.random.Generator) else seed,
            "n_obstacles": int(n_obs) + 1,  # +1 for target obstacle
            "has_explicit_target_obstacle": True,
        },
    }


def generate_scenarios(
    n: int = 10,
    preset: str = "baseline",
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> List[dict]:
    """
    Generate a list of n scenarios with derived seeds (reproducible but different).
    """
    base_rng = _rng(seed)
    scenarios = []
    for i in range(n):
        s = int(base_rng.integers(0, 2**31 - 1))
        scen = example_scenario(preset=preset, seed=s)
        scen["meta"]["index"] = i
        scenarios.append(scen)
    return scenarios


if __name__ == "__main__":
    sc = example_scenario(preset="baseline", seed=1234)
    print("Example scenario:")
    print(" ego:", sc["ego"])
    print(" attacker:", sc["attacker"])
    print(" attack_region:", sc["attack_region"])
    print(" target bbox:", sc["target"]["bbox"])
    print(" obstacles (N={}):".format(sc["obstacles"].shape[0]))
    print(sc["obstacles"])
