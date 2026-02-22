# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 20:38:27 2025

@author: Lenovo
"""

# demo_spyder.py
# === CAV Project Demo (improved: sensor noise + stealth attacks) ===
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ensure src is visible when running from project root
sys.path.append(os.path.join(os.getcwd(), "src"))

from simulation import example_scenario
from perception import local_perception
from occupancy_map import build_occupancy_map
from attacker_rc import ray_cast_fabrication, stealth_fabrication_near_local

# try to use enhanced CAD if available, otherwise fallback
try:
    from cad_detector import detect_inconsistency_enhanced as detect_inconsistency
    ENHANCED_CAD = True
except Exception:
    from cad_detector import detect_inconsistency
    ENHANCED_CAD = False


def single_run(params, plot=True, rng=None):
    """
    Run one demo under given params.
    params: dict with keys (example defaults below)
        - sensor_jitter, sensor_dropout, loc_error, range_drop
        - attack: bool
        - attack_n_fake, attack_overlap_frac, attack_edge_spread, attack_fake_jitter
    """
    if rng is None:
        rng = np.random.default_rng()

    scenario = example_scenario()
    ego = scenario["ego"]
    obstacles = scenario["obstacles"]

    # 1) local perception (with sensor imperfections)
    pts_local = local_perception(
        ego["pose"],
        obstacles,
        max_range=50.0,
        jitter=params.get("sensor_jitter", 0.0),
        dropout_prob=params.get("sensor_dropout", 0.0),
        loc_error_std=params.get("loc_error", 0.0),
        range_drop=params.get("range_drop", 0.0),
        rng=rng,
    )
    occ_local = build_occupancy_map(pts_local)

    # 2) attacker: stealth or uniform
    attack_region = params.get("attack_region", (5.0, -2.0, 4.0, 4.0))
    attacked = params.get("attack", True)
    if attacked:
        n_fake = params.get("attack_n_fake", 120)
        overlap = params.get("attack_overlap_frac", 0.5)
        edge_spread = params.get("attack_edge_spread", 0.15)
        fake_jitter = params.get("attack_fake_jitter", 0.0)

        if pts_local.shape[0] == 0:
            fake_points = ray_cast_fabrication(attack_region, n_points=n_fake)
        else:
            fake_points = stealth_fabrication_near_local(
                pts_local,
                attack_region,
                n_points=n_fake,
                overlap_frac=overlap,
                rng=rng,
                edge_spread=edge_spread,
            )
            # optional small jitter to fake points to avoid being perfectly aligned
            if fake_jitter > 0 and fake_points.shape[0] > 0:
                fake_points = fake_points + rng.normal(scale=fake_jitter, size=fake_points.shape)
    else:
        fake_points = np.zeros((0, 2))

    # 3) fuse
    if pts_local.shape[0] == 0 and fake_points.shape[0] == 0:
        pts_fused = np.zeros((0, 2))
    elif pts_local.shape[0] == 0:
        pts_fused = fake_points
    elif fake_points.shape[0] == 0:
        pts_fused = pts_local
    else:
        pts_fused = np.vstack([pts_local, fake_points])

    occ_fused = build_occupancy_map(pts_fused)

    # 4) detection (use enhanced if available). Note: min_ratio param passed if desired.
    if ENHANCED_CAD:
        suspicious, ratio = detect_inconsistency(
            occ_local,
            occ_fused,
            min_ratio=params.get("min_ratio", 0.05),
            min_cluster_size=params.get("min_cluster_size", 3),
            distance_threshold=params.get("distance_threshold", 0.5),
        )
    else:
        suspicious, ratio = detect_inconsistency(
            occ_local, occ_fused, min_ratio=params.get("min_ratio", 0.05)
        )

    # print summary
    print(
        f"attack={attacked}, n_local={pts_local.shape[0]}, n_fused={pts_fused.shape[0]}, "
        f"suspicious={suspicious}, ratio={ratio:.3f}"
    )

    # plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(occ_local, origin="lower")
        axes[0].set_title("Local occupancy (ego only)")
        axes[1].imshow(occ_fused, origin="lower")
        t = f"Fused occupancy (with attack={attacked})\nSuspicious={suspicious}, ratio={ratio:.3f}"
        axes[1].set_title(t)
        plt.tight_layout()
        plt.show()

    return {"suspicious": suspicious, "ratio": ratio, "n_local": pts_local.shape[0], "n_fused": pts_fused.shape[0]}


def run_quick_tests(plot=True):
    """
    Run a few parameterized quick scenarios to compare behaviors.
    """
    rng = np.random.default_rng(2025)

    configs = [
        {
            "name": "strong_attack",
            "attack": True,
            "attack_n_fake": 300,
            "attack_overlap_frac": 0.1,
            "attack_fake_jitter": 0.0,
            "sensor_jitter": 0.0,
            "sensor_dropout": 0.0,
            "loc_error": 0.0,
            "min_ratio": 0.05,
        },
        {
            "name": "moderate_stealth",
            "attack": True,
            "attack_n_fake": 120,
            "attack_overlap_frac": 0.5,
            "attack_fake_jitter": 0.02,
            "sensor_jitter": 0.03,
            "sensor_dropout": 0.08,
            "loc_error": 0.02,
            "min_ratio": 0.08,
        },
        {
            "name": "very_stealth",
            "attack": True,
            "attack_n_fake": 30,
            "attack_overlap_frac": 0.75,
            "attack_fake_jitter": 0.05,
            "sensor_jitter": 0.06,
            "sensor_dropout": 0.15,
            "loc_error": 0.04,
            "min_ratio": 0.08,
        },
        {
            "name": "clean",
            "attack": False,
            "attack_n_fake": 0,
            "sensor_jitter": 0.03,
            "sensor_dropout": 0.05,
            "loc_error": 0.02,
            "min_ratio": 0.05,
        },
    ]

    results = {}
    for cfg in configs:
        print("\n--- Running config:", cfg["name"])
        res = single_run(cfg, plot=plot, rng=rng)
        results[cfg["name"]] = res
    return results


if __name__ == "__main__":
    # default: run single demo interactive
    print("Running single demo (interactive). If you want multi-tests, use run_quick_tests().")
    # example interactive params (you can edit these)
    params = {
        "attack": True,
        "attack_n_fake": 80,
        "attack_overlap_frac": 0.5,
        "attack_edge_spread": 0.15,
        "attack_fake_jitter": 0.02,
        "sensor_jitter": 0.03,
        "sensor_dropout": 0.05,
        "loc_error": 0.02,
        "range_drop": 0.0,
        "min_ratio": 0.08,
        "min_cluster_size": 3,
        "distance_threshold": 0.5,
    }

    # run single interactive demo (will plot)
    single_run(params, plot=True)

    # optionally, run the quick sweep of different scenarios (uncomment to run)
    # results = run_quick_tests(plot=True)
    # print(results)

