# eval_advanced.py
# ------------------------------------------------------------
# End-to-end evaluation for CAD under spoof/remove attacks.
# Produces ROC/PR (overall & per-group), confusion, TPR/FPR bars,
# IoU histogram, score histogram, and a threshold sweep CSV.
# ------------------------------------------------------------

import os
import sys
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ---------- ensure local imports ----------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulation import example_scenario
from perception import local_perception
from occupancy_map import build_occupancy_map, DEFAULT_BOUNDS, DEFAULT_RES
from attacker_rc import (
    ray_cast_fabrication,
    stealth_fabrication_near_local,
    remove_points,
)

try:
    from cad_detector import detect_inconsistency_enhanced as detect_inconsistency
    ENHANCED_CAD = True
except Exception:
    from cad_detector import detect_inconsistency
    ENHANCED_CAD = False

# ---------- I/O ----------
OUT_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures_advanced")
LOG_DIR = os.path.join(OUT_DIR, "logs_advanced")
SAMP_DIR = os.path.join(FIG_DIR, "samples")

for d in (OUT_DIR, FIG_DIR, LOG_DIR, SAMP_DIR):
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(
    LOG_DIR, f"adv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


# ---------- helpers ----------
def compute_iou_with_gt(
    occ_mask,
    gt_bbox,
    bounds=DEFAULT_BOUNDS,
    map_resolution: float = DEFAULT_RES,
) -> float:
    """
    IoU between occupancy mask (0/1) and a GT rectangle in ego frame (meters).

Important: This function must be exactly aligned with build_occupancy_map:
   - The same bounds (xmin, xmax, ymin, ymax)
    - Same resolution
    - Convert (x,y) → (col,row) same as u,v formula
    """
    if gt_bbox is None:
        return 0.0

    occ_mask = np.asarray(occ_mask)
    h, w = occ_mask.shape  # H,W از occupancy-map

    xmin, xmax, ymin, ymax = bounds

    x0, y0, ww, hh = gt_bbox
    x1 = x0 + ww
    y1 = y0 + hh

    # meters -> pixel indices (like build_occupancy_map)
    u0 = int(np.floor((x0 - xmin) / map_resolution))
    v0 = int(np.floor((y0 - ymin) / map_resolution))
    u1 = int(np.ceil((x1 - xmin) / map_resolution))
    v1 = int(np.ceil((y1 - ymin) / map_resolution))

    # clip به داخل تصویر
    u0 = max(0, min(w - 1, u0))
    u1 = max(0, min(w - 1, u1))
    v0 = max(0, min(h - 1, v0))
    v1 = max(0, min(h - 1, v1))

    if u1 <= u0 or v1 <= v0:
        return 0.0

    gt_mask = np.zeros_like(occ_mask, dtype=bool)
 # Note: row = v, column = u
    gt_mask[v0:v1 + 1, u0:u1 + 1] = True

    occ_bool = (occ_mask == 1)

    inter = np.logical_and(occ_bool, gt_mask).sum()
    union = np.logical_or(occ_bool, gt_mask).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def generate_fake_points_for_config(local_pts, region, params, rng):
    """
    Decide which fabrication model to use (ray-cast or stealth) and
    return a cloud of fake points in the attack region.
    """
    n_fake = int(params.get("attack_n_fake", 80))
    overlap = float(params.get("attack_overlap_frac", 0.6))
    fake_jitter = float(params.get("attack_fake_jitter", 0.03))

    if n_fake <= 0:
        return np.zeros((0, 2), dtype=float)

    if local_pts is None or len(local_pts) == 0:
        fake = ray_cast_fabrication(region, n_points=n_fake, rng=rng)
    else:
        fake = stealth_fabrication_near_local(
            local_pts,
            region,
            n_points=n_fake,
            overlap_frac=overlap,
            rng=rng,
        )

    if fake_jitter > 0 and fake.shape[0] > 0:
        fake = fake + rng.normal(scale=fake_jitter, size=fake.shape)
    return fake


def save_sample_figure(occ_local, occ_fused, cfg_name, trial_idx, outdir=SAMP_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(occ_local, origin="lower")
    axes[0].set_title("Local")
    axes[1].imshow(occ_fused, origin="lower")
    axes[1].set_title(f"Fused ({cfg_name})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fname = os.path.join(outdir, f"sample_{cfg_name}_{trial_idx}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ---------- single trial ----------
def run_trial(params, rng, map_resolution: float = DEFAULT_RES):
    """
    Simulate one frame under the given config (clean or attacked).
    Returns a dict with label, CAD score, IoU, etc.
    """
  # For each scenario we generate a separate seed from the rng
    scen_seed = int(rng.integers(0, 2**31 - 1))
    preset = params.get("preset", "baseline")
    scenario = example_scenario(preset=preset, seed=scen_seed)

    ego = scenario["ego"]
    obstacles = scenario["obstacles"]
    target = scenario.get("target", None)
    region = scenario["attack_region"]  # Target/Attack area in ego frame

    # --- local sensing (no attack; baseline) ---
    pts_local = local_perception(
        ego["pose"],
        obstacles,
        max_range=50.0,
        jitter=params.get("jitter", 0.02),
        dropout_prob=params.get("dropout", 0.05),
        loc_error_std=params.get("loc_error", 0.015),
        range_drop=params.get("range_drop", 0.0),
        rng=rng,
    )

    attacked = bool(params.get("attacked", False))

   # --- build pts_fused ---
    if attacked:
      # Sometimes remove-only to make detection harder
        remove_only_prob = float(params.get("remove_only_prob", 0.25))
        remove_only = (
            params.get("attack_type", "spoof") == "spoof"
            and rng.random() < remove_only_prob
        )

        if remove_only:
            frac = float(params.get("remove_frac_hard", 0.55))
            pts_fused = remove_points(pts_local, frac_remove=frac, rng=rng)
        else:
            fake_pts = generate_fake_points_for_config(pts_local, region, params, rng)
            if params.get("attack_type", "spoof") == "remove":
                frac = float(params.get("remove_frac", 0.4))
                base = remove_points(pts_local, frac_remove=frac, rng=rng)
                pts_fused = fake_pts if base.shape[0] == 0 else np.vstack([base, fake_pts])
            else:
                pts_fused = (
                    fake_pts if pts_local.shape[0] == 0 else np.vstack([pts_local, fake_pts])
                )
    else:
        # clean but with real mismatch between local and fused
        pts_fused = np.array(pts_local, copy=True)

        # Additional dropout on fused
        cdrop = float(params.get("clean_dropout", 0.18))
        if pts_fused.shape[0] > 0 and cdrop > 0:
            keep_mask = rng.random(len(pts_fused)) >= cdrop
            pts_fused = pts_fused[keep_mask]

        # jitter on fused
        cj = float(params.get("clean_jitter", 0.06))
        if pts_fused.shape[0] > 0 and cj > 0:
            pts_fused = pts_fused + rng.normal(scale=cj, size=pts_fused.shape)

       # Several spurious points near the region
        n_spur = int(params.get("clean_spurious", rng.integers(4, 10)))
        if n_spur > 0:
            x0, y0, w, h = region
            xs = rng.uniform(x0, x0 + w, size=n_spur)
            ys = rng.uniform(y0 - h / 2.0, y0 + h / 2.0, size=n_spur)
            sp = np.vstack([xs, ys]).T
            pts_fused = sp if pts_fused.shape[0] == 0 else np.vstack([pts_fused, sp])

    # --- registration / sync drift ---
    drift_std = float(params.get("sync_offset", 0.3))
    if pts_fused.shape[0] > 0 and drift_std > 0:
        pts_fused = pts_fused + rng.normal(scale=drift_std, size=(1, 2))

   # --- Creating occupancy maps after drift ---
    occ_local = build_occupancy_map(pts_local)
    occ_fused = build_occupancy_map(pts_fused)

    # --- CAD score ---
    suspicious, ratio = detect_inconsistency(
        occ_local,
        occ_fused,
        min_ratio=params.get("min_ratio", 0.12),
        min_cluster_size=params.get("min_cluster_size", 6),
        distance_threshold=params.get("distance_threshold", 0.55),
    )

    # --- IoU with GT region (same attack_region) ---
    iou = 0.0
    if target is not None:
        iou = compute_iou_with_gt(
            occ_fused, target.get("bbox", None), bounds=DEFAULT_BOUNDS, map_resolution=map_resolution
        )

    return {
        "attacked": attacked,
        "suspicious": bool(suspicious),
        "ratio": float(ratio),
        "iou": float(iou),
        "group": params.get("group", params.get("config_name", "NA")),
        "config_name": params.get("config_name", "NA"),
        "n_local": int(pts_local.shape[0]),
        "n_fused": int(pts_fused.shape[0]),
        "occ_local": occ_local,
        "occ_fused": occ_fused,
    }


# ---------- experiment sweep ----------
def sweep_experiments(
    n_per_setting: int = 100, seed: int = 2025, save_samples: bool = True
):
    rng = np.random.default_rng(seed)
    rows = []

    def make_group(name, base):
        a = dict(base)
        a.update({"group": name, "config_name": name + "-Attack", "attacked": True})
        c = dict(base)
        c.update({"group": name, "config_name": name + "-Clean", "attacked": False})
        return [a, c]

    groups = []

    # Spoof_RC: RC-lidar based spoofing, relatively easier to detect
    groups += make_group(
        "Spoof_RC",
        {
            "preset": "baseline",
            "attack_type": "spoof",
            "attack_n_fake": rng.integers(70, 95),
            "attack_overlap_frac": 0.6,
            "attack_fake_jitter": 0.04,
            "jitter": 0.02,
            "dropout": 0.06,
            "loc_error": 0.015,
            "min_ratio": 0.13,
            "min_cluster_size": 6,
            "distance_threshold": 0.55,
            "sync_offset": 0.30,
            "remove_frac_hard": 0.60,
            "remove_only_prob": 0.25,
        },
    )

    # Spoof_Adv: More advanced attack (less fake and more overlap)
    groups += make_group(
        "Spoof_Adv",
        {
            "preset": "near",
            "attack_type": "spoof",
            "attack_n_fake": rng.integers(55, 80),
            "attack_overlap_frac": 0.75,
            "attack_fake_jitter": 0.04,
            "jitter": 0.02,
            "dropout": 0.07,
            "loc_error": 0.015,
            "min_ratio": 0.13,
            "min_cluster_size": 6,
            "distance_threshold": 0.58,
            "sync_offset": 0.32,
            "remove_frac_hard": 0.65,
            "remove_only_prob": 0.35,
        },
    )

    # Remove_RC: Remove points, TPR expectation too high
    groups += make_group(
        "Remove_RC",
        {
            "preset": "dense",
            "attack_type": "remove",
            "remove_frac": 0.18,
            "attack_n_fake": 15,
            "jitter": 0.01,
            "dropout": 0.02,
            "loc_error": 0.012,
            "min_ratio": 0.05,
            "min_cluster_size": 5,
            "distance_threshold": 0.70,
            "sync_offset": 0.30,
            "remove_only_prob": 0.0,
        },
    )

    for cfg in groups:
        for i in range(n_per_setting):
            params = dict(cfg)
            res = run_trial(params, rng)

            rows.append(
                {
                    "group": res["group"],
                    "config_name": res["config_name"],
                    "trial_idx": i,
                    "attacked": res["attacked"],
                    "suspicious": res["suspicious"],
                    "ratio": res["ratio"],
                    "iou": res["iou"],
                    "n_local": res["n_local"],
                    "n_fused": res["n_fused"],
                    "attack_type": params.get("attack_type", "spoof"),
                    "attack_n_fake": params.get("attack_n_fake", 0),
                    "attack_overlap_frac": params.get("attack_overlap_frac", 0.0),
                    "remove_frac": params.get("remove_frac", 0.0),
                    "jitter": params.get("jitter", 0.0),
                    "dropout": params.get("dropout", 0.0),
                    "loc_error": params.get("loc_error", 0.0),
                    "min_ratio": params.get("min_ratio", 0.12),
                    "min_cluster_size": params.get("min_cluster_size", 6),
                    "distance_threshold": params.get("distance_threshold", 0.55),
                    "sync_offset": params.get("sync_offset", 0.30),
                }
            )

            if save_samples and i < 3:
                tag = (
                    "TP"
                    if (res["attacked"] and res["suspicious"])
                    else "FN"
                    if (res["attacked"] and not res["suspicious"])
                    else "FP"
                    if ((not res["attacked"]) and res["suspicious"])
                    else "TN"
                )
                save_sample_figure(
                    res["occ_local"],
                    res["occ_fused"],
                    f"{res['config_name']}_{tag}",
                    i,
                )

    # CSV log of all trials
    order = [
        "group",
        "config_name",
        "trial_idx",
        "attacked",
        "attack_type",
        "attack_n_fake",
        "attack_overlap_frac",
        "remove_frac",
        "jitter",
        "dropout",
        "loc_error",
        "min_ratio",
        "min_cluster_size",
        "distance_threshold",
        "sync_offset",
        "n_local",
        "n_fused",
        "ratio",
        "iou",
        "suspicious",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in order})

    print(f"[+] Saved CSV -> {CSV_PATH}")
    return rows


# ---------- analysis & plots ----------
def _best_threshold_by_f1(scores, y_true):
    best_th, best_f1 = 0.0, -1.0
    for th in np.linspace(0, 1, 101):
        preds = (scores >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th), float(best_f1)


def analyze_and_plot(rows):
    y_true = np.array([1 if r["attacked"] else 0 for r in rows])
    scores = np.array([r["ratio"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    # overall ROC/PR
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (overall)")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"AUC_PR={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (overall)")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "pr_curve.png"), dpi=150)
    plt.close()

    # histogram attack vs clean
    attacked_scores = scores[y_true == 1]
    clean_scores = scores[y_true == 0]
    plt.figure(figsize=(6, 4))
    plt.hist(attacked_scores, bins=35, alpha=0.7, label="attack")
    plt.hist(clean_scores, bins=35, alpha=0.7, label="clean")
    plt.xlabel("fake-cell ratio")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "ratio_hist_advanced.png"), dpi=150)
    plt.close()

    # per-group ROC
    uniq = sorted(list(set(groups)))
    plt.figure(figsize=(6, 5))
    for g in uniq:
        m = groups == g
        f, t, _ = roc_curve(y_true[m], scores[m])
        plt.plot(f, t, label=f"{g} (AUC={auc(f, t):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC per group")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "roc_per_config.png"), dpi=150)
    plt.close()

    # per-group PR
    plt.figure(figsize=(6, 5))
    for g in uniq:
        m = groups == g
        p, r, _ = precision_recall_curve(y_true[m], scores[m])
        plt.plot(r, p, label=f"{g} (AUC={auc(r, p):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR per group")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "pr_per_config.png"), dpi=150)
    plt.close()

    # best threshold by global F1
    best_th, best_f1 = _best_threshold_by_f1(scores, y_true)
    preds = (scores >= best_th).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).reshape(2, 2)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Clean", "Attack"])
    plt.yticks([0, 1], ["Pred Clean", "Pred Attack"])
    plt.title(f"Confusion @ best F1 (th={best_th:.2f}, F1={best_f1:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_bestF1.png"), dpi=150)
    plt.close()

    # TPR/FPR per group at this threshold
    tprs, fprs = [], []
    for g in uniq:
        m = groups == g
        yt = y_true[m]
        pr = (scores[m] >= best_th).astype(int)
        cm_g = confusion_matrix(yt, pr, labels=[0, 1]).reshape(2, 2)
        _tn, _fp, _fn, _tp = cm_g.ravel()
        tprs.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)
        fprs.append(_fp / (_fp + _tn) if (_fp + _tn) > 0 else 0.0)

    x = np.arange(len(uniq))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, tprs, width=0.4, label="TPR")
    plt.bar(x + 0.2, fprs, width=0.4, label="FPR")
    plt.xticks(x, uniq, rotation=15)
    plt.ylim(0, 1)
    plt.title("TPR/FPR per group @ best F1 threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tpr_fpr_per_config.png"), dpi=150)
    plt.close()

    # IoU histogram
    ious = np.array([r["iou"] for r in rows if "iou" in r])
    if np.isfinite(ious).sum() > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(ious, bins=30, alpha=0.8)
        plt.xlabel("IoU w.r.t. GT bbox")
        plt.ylabel("count")
        plt.title("IoU histogram")
        plt.savefig(os.path.join(FIG_DIR, "iou_hist.png"), dpi=150)
        plt.close()

    # threshold sweep table
    sweep_rows = []
    for th in np.linspace(0, 1, 101):
        pr = (scores >= th).astype(int)
        cm_th = confusion_matrix(y_true, pr, labels=[0, 1]).reshape(2, 2)
        _tn, _fp, _fn, _tp = cm_th.ravel()
        prec = precision_score(y_true, pr, zero_division=0)
        rec = recall_score(y_true, pr, zero_division=0)
        f1v = f1_score(y_true, pr, zero_division=0)
        acc = accuracy_score(y_true, pr)
        sweep_rows.append(
            {
                "threshold": float(th),
                "TPR": float(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0),
                "FPR": float(_fp / (_fp + _tn) if (_fp + _tn) > 0 else 0.0),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1v),
                "accuracy": float(acc),
            }
        )

    sweep_csv = os.path.join(LOG_DIR, "threshold_sweep.csv")
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # summary
    with open(os.path.join(LOG_DIR, "summary_advanced.txt"), "w") as f:
        f.write(f"roc_auc: {roc_auc}\n")
        f.write(f"pr_auc: {pr_auc}\n")
        f.write(f"best_threshold_f1: {best_th}\n")
        f.write(f"n_examples: {len(rows)}\n")
        f.write(f"n_attacked: {int(y_true.sum())}\n")
        f.write(f"n_clean: {int(len(y_true) - y_true.sum())}\n")

    print("[+] Saved plots & summary")


# ---------- main ----------
def main_quick(
    n_per_setting: int = 100, seed: int = 2025, save_samples: bool = True
):
    rows = sweep_experiments(
        n_per_setting=n_per_setting, seed=seed, save_samples=save_samples
    )
    y = np.array([1 if r["attacked"] else 0 for r in rows])
    s = np.array([r["ratio"] for r in rows])
    print("[sanity] mean ratio attack / clean:", s[y == 1].mean(), "/", s[y == 0].mean())
    analyze_and_plot(rows)
    print("Done.")


if __name__ == "__main__":
    main_quick()
