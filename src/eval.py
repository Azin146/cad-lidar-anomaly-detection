import numpy as np
import matplotlib.pyplot as plt

from simulation import example_scenario
from perception import local_perception
from occupancy_map import build_occupancy_map
from attacker_rc import ray_cast_fabrication
from cad_detector import detect_inconsistency



def run_single_demo(plot=True):
    # 1) Creating a simple scenario
    scenario = example_scenario()
    ego = scenario["ego"]
    obstacles = scenario["obstacles"]

    # 2) Local ego perception (no attack)
    pts_local = local_perception(ego["pose"], obstacles)
    occ_local = build_occupancy_map(pts_local)

    # 3) Attack: Adding fake dots in front of ego
    target_region = (5.0, -2.0, 4.0, 4.0)  # (x0, y0, w, h) In ego frame
    fake_points = ray_cast_fabrication(target_region, n_points=300)

    # 4) Fusion of real and fake data (fused perception)
    if pts_local.shape[0] == 0:
        pts_fused = fake_points
    else:
        pts_fused = np.vstack([pts_local, fake_points])

    occ_fused = build_occupancy_map(pts_fused)

    # 5) CAD: Detecting inconsistencies between local and merged maps
    suspicious, ratio = detect_inconsistency(occ_local, occ_fused, min_ratio=0.05)
    print(f"Suspicious? {suspicious}, fake-cell ratio={ratio:.3f}")

    # 6) Video display (optional)
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(occ_local, origin="lower")
        axes[0].set_title("Local occupancy (ego only)")

        axes[1].imshow(occ_fused, origin="lower")
        axes[1].set_title(f"Fused occupancy (with attack)\nSuspicious={suspicious}")

        plt.tight_layout()
        plt.show()

    return {"suspicious": suspicious, "ratio": ratio}


if __name__ == "__main__":
    run_single_demo(plot=True)
