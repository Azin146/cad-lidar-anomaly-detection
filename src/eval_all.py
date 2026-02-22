# eval_all.py
# Run N scenarios (attack/no attack), collect statistics, and save results and graphs.
import os, sys, csv, random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Make sure src is in the path (not necessary if running from root)
# sys.path.append(os.path.join(os.getcwd(), "src"))

# Importing project modules
from simulation import example_scenario
from perception import local_perception
from occupancy_map import build_occupancy_map
from attacker_rc import ray_cast_fabrication
from cad_detector import detect_inconsistency

OUT_DIR = "results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def random_target_region(rng, ego_range=(3.0, 15.0), w_range=(1.0,5.0), h_range=(1.0,5.0)):
    """
    Defines a random rectangle in front of ego
    ego_range: distance x from ego (min,max)
    w_range/h_range: Random width and height
    We also randomize the y-offset a bit (left/right)
    """
    x0 = rng.uniform(ego_range[0], ego_range[1])
    y0 = rng.uniform(-3.0, 3.0)   # Left/Right
    w = rng.uniform(w_range[0], w_range[1])
    h = rng.uniform(h_range[0], h_range[1])
    return (x0, y0, w, h)

def run_one_trial(attacked: bool, rng, n_fake_points=300, min_ratio_threshold=0.05):
    """
    Running a scenario:
      - If attacked=True: fake points are generated
      - Returns the size of the ratio and the suspicious output.
    """
    scenario = example_scenario()
    ego = scenario['ego']
    obstacles = scenario['obstacles']

    # Local perception and local map
    pts_local = local_perception(ego['pose'], obstacles)
    occ_local = build_occupancy_map(pts_local)

    if attacked:
        target_region = random_target_region(rng)
        fake_pts = ray_cast_fabrication(target_region, n_points=n_fake_points)
        if pts_local.shape[0] == 0:
            pts_fused = fake_pts
        else:
            pts_fused = np.vstack([pts_local, fake_pts])
    else:
        # No attack, fused = local (similar to a real multi-partner scenario where everyone is normal)
        pts_fused = pts_local

    occ_fused = build_occupancy_map(pts_fused)
    suspicious, ratio = detect_inconsistency(occ_local, occ_fused, min_ratio=min_ratio_threshold)

    return {
        "attacked": attacked,
        "suspicious": bool(suspicious),
        "ratio": float(ratio),
        "n_local": int(pts_local.shape[0]),
        "n_fused": int(pts_fused.shape[0])
    }

def eval_experiments(n_attack=100, n_clean=100, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rows = []
    # Attacked performance
    for i in range(n_attack):
        r = run_one_trial(attacked=True, rng=np_rng, n_fake_points=300)
        r["trial_id"] = f"attack_{i+1}"
        rows.append(r)

    # Execution without attack
    for i in range(n_clean):
        r = run_one_trial(attacked=False, rng=np_rng, n_fake_points=0)
        r["trial_id"] = f"clean_{i+1}"
        rows.append(r)

    # Save CSV
    fieldnames = ["trial_id","attacked","suspicious","ratio","n_local","n_fused"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in fieldnames})

    print(f"[+] Saved results to {CSV_PATH}")
    return rows

def compute_metrics(rows):
    # Convert to numpy arrays
    attacked = np.array([r["attacked"] for r in rows])
    suspicious = np.array([r["suspicious"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])

    # TPR = fraction of attacked runs that were detected
    attacked_idx = attacked == True
    clean_idx = attacked == False
    tpr = suspicious[attacked_idx].sum() / max(1, attacked_idx.sum())
    fpr = suspicious[clean_idx].sum() / max(1, clean_idx.sum())

    mean_ratio_attack = ratio[attacked_idx].mean() if attacked_idx.sum()>0 else 0.0
    mean_ratio_clean  = ratio[clean_idx].mean() if clean_idx.sum()>0 else 0.0

    metrics = {
        "TPR": float(tpr),
        "FPR": float(fpr),
        "mean_ratio_attack": float(mean_ratio_attack),
        "mean_ratio_clean": float(mean_ratio_clean),
        "n_attack": int(attacked_idx.sum()),
        "n_clean": int(clean_idx.sum())
    }
    return metrics

def plot_distributions(rows):
    attacked = np.array([r["attacked"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    ratio_attack = ratio[attacked==True]
    ratio_clean = ratio[attacked==False]

    plt.figure(figsize=(6,4))
    plt.hist(ratio_attack, bins=25, alpha=0.7, label="attack")
    plt.hist(ratio_clean, bins=25, alpha=0.7, label="clean")
    plt.xlabel("fake-cell ratio")
    plt.ylabel("count")
    plt.legend()
    p1 = os.path.join(FIG_DIR, "ratio_hist.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[+] saved {p1}")

def main():
    rows = eval_experiments(n_attack=100, n_clean=100, seed=2025)
    metrics = compute_metrics(rows)
    print("=== Metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    plot_distributions(rows)
    # Save summary metrics
    with open(os.path.join(LOG_DIR, "summary.txt"), "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("[+] Done. Figures in:", FIG_DIR)

if __name__ == "__main__":
    main()
