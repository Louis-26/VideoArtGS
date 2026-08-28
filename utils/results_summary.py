#!/usr/bin/env python3
"""
Aggregate per-scene result.csv files into a summary table.

Each scene's metrics live at:
    outputs/{dataset}/{subset}/{scene}/final/train/ours_{iteration}/result.csv

This script scans all such files, aggregates the 4 core metrics (mean ± std), including axis/position/joint state error,
and CD loss for whole/movable/static parts

Examples
--------
# Table 2 (VideoArtGS-20, no joint-type split):
python results_summary.py --dataset videoartgs --subset sapien

# Same, plus the State column (needs joint_state_error in result.csv --
# produced by the current eval.py, or backfilled via utils/backfill_state_error.py):
python results_summary.py --dataset videoartgs --subset sapien --with-state

# Table 1 (Video2Articulation-S, split revolute/prismatic, with State column):
python results_summary.py --dataset v2a --subset sapien --split-joint --with-state

# Restrict to a subset of scenes you actually trained:
python results_summary.py --dataset v2a --subset sapien --split-joint --with-state \
    --scenes 10143_joint_0_bg_view_1 101886_joint_0_bg_view_0
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd


# csv field -> (display label, unit note). Order here = column order in output.
METRIC_FIELDS = [
    ("angle", "Axis (deg)"),
    ("distance", "Position (cm)"),
    ("joint_state_error", "State (deg/cm)"),  # deg for revolute, cm for prismatic
    ("CD_whole", "CD-w (cm)"),
    ("CD_dynamic", "CD-m (cm)"),
    ("CD_static", "CD-s (cm)"),
]

# Reference values from the paper (Ours row): (mean, std), for comparison.
PAPER_REFERENCE = {
    "videoartgs": {  # Table 2
        "angle": (0.34, 0.80), "distance": (0.10, 0.10),
        "CD_whole": (0.09, 0.09), "CD_dynamic": (0.26, 0.61), "CD_static": (0.24, 0.58),
    },
    "v2a": {  # Table 1 — revolute / prismatic reported separately in paper
        "revolute":  {"angle": (0.32, 0.44), "distance": (0.42, 0.75),
                      "joint_state_error": (1.15, 2.29),
                      "CD_whole": (0.29, 0.24), "CD_dynamic": (0.40, 0.32),
                      "CD_static": (1.11, 2.11)},
        "prismatic": {"angle": (0.35, 0.45), "joint_state_error": (1.03, 2.46),
                      "CD_whole": (0.29, 0.24), "CD_dynamic": (0.40, 0.32),
                      "CD_static": (1.11, 2.11)},
    },
}


def find_result_csvs(base, dataset, subset, iteration, output_dir="outputs", scenes=None):
    """Return list of (scene_name, csv_path) for all scenes with a result.csv."""
    root = os.path.join(base, f"{output_dir}", dataset, subset)
    pattern = os.path.join(root, "*", "final", "train", f"ours_{iteration}", "result.csv")
    paths = sorted(glob.glob(pattern))

    found = []
    for p in paths:
        # .../{scene}/final/train/ours_{iter}/result.csv  -> scene is 5 levels up
        scene = p.split(os.sep)[-5]
        if scenes is not None and scene not in scenes:
            continue
        found.append((scene, p))
    return found


def read_joint_type(base, dataset, subset, scene):
    """Read joint type from gt/mobility_v2.json. hinge->revolute, slider->prismatic."""
    gt_json = os.path.join(base, "data", dataset, subset, scene, "gt", "mobility_v2.json")
    try:
        with open(gt_json) as f:
            data = json.load(f)
        j = data[0]["joint"]  # "hinge" or "slider"
        if j == "hinge":
            return "revolute"
        if j == "slider":
            return "prismatic"
        return j
    except Exception:
        return "unknown"


def load_all(base, dataset, subset, iteration, scenes, need_joint_type, output_dir="outputs"):
    """Load every scene's metrics into a DataFrame."""
    entries = find_result_csvs(base, dataset, subset, iteration, output_dir, scenes)
    if not entries:
        raise FileNotFoundError(
            f"No result.csv found under {output_dir}/{dataset}/{subset}/*/final/train/"
            f"ours_{iteration}/result.csv"
        )

    records = []
    for scene, path in entries:
        series = pd.read_csv(path, index_col="Metric")["Value"]
        rec = {"scene": scene}
        for field, _ in METRIC_FIELDS:
            rec[field] = series.get(field, np.nan)
        if need_joint_type:
            rec["joint_type"] = read_joint_type(base, dataset, subset, scene)
        records.append(rec)

    return pd.DataFrame(records), len(entries)


def aggregate(df, fields):
    """Return dict field -> (mean, std, count) over non-NaN values."""
    out = {}
    for f in fields:
        vals = df[f].dropna()
        if len(vals) == 0:
            out[f] = (np.nan, np.nan, 0)
        else:
            out[f] = (vals.mean(), vals.std(), len(vals))
    return out


def fmt(x, nd=3):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{nd}f}"


def markdown_table(group_stats, fields, paper_ref=None):
    """
    Transposed (horizontal) Markdown table:
      columns = metrics (Axis, Position, ...), rows = method (Ours, Paper).

    group_stats: dict group_name -> (stats_dict, n_scenes)
    """
    labels = dict(METRIC_FIELDS)
    groups = list(group_stats.keys())

    # --- header row: Method | metric1 | metric2 | ... ---
    header = "| Method | " + " | ".join(labels.get(f, f) for f in fields) + " |"
    sep = "|" + "---|" * (len(fields) + 1)
    lines = [header, sep]

    # --- one row per group (e.g. "Ours (reproduced)") ---
    for g in groups:
        stats, cnt = group_stats[g]
        row = f"| {g} (n={cnt}) | "
        cells = []
        for field in fields:
            mean, std, c = stats.get(field, (np.nan, np.nan, 0))
            if c == 0 or (isinstance(mean, float) and np.isnan(mean)):
                cells.append("—")
            else:
                cell = f"{fmt(mean)} ± {fmt(std)}"
                if c != cnt:  # fewer scenes have this metric (e.g. State-P)
                    cell += f" (n={c})"
                cells.append(cell)
        row += " | ".join(cells) + " |"
        lines.append(row)

    # --- paper reference row (if provided) ---
    if paper_ref is not None:
        row = "| Ours (paper) | "
        cells = []
        for field in fields:
            pv = paper_ref.get(field) if isinstance(paper_ref, dict) else None
            if pv is None:
                cells.append("—")
            elif isinstance(pv, (tuple, list)):   # (mean, std)
                cells.append(f"{fmt(pv[0])} ± {fmt(pv[1])}")
            else:                                  # bare mean (back-compat)
                cells.append(fmt(pv))
        row += " | ".join(cells) + " |"
        lines.append(row)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-scene result.csv into a Markdown summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base", default=".",
                    help="Repo root (where outputs/ and data/ live).")
    ap.add_argument("--dataset", required=True, choices=["videoartgs", "v2a"],
                    help="Dataset name (videoartgs=Table 2, v2a=Table 1).")
    ap.add_argument("--subset", default="sapien", help="Subset name.")
    ap.add_argument("--output_dir", default="outputs", help="Output directory.")
    ap.add_argument("--iteration", default=20000, type=int,
                    help="Iteration number in ours_{iteration}.")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Optional explicit scene list; default = all found.")
    ap.add_argument("--split-joint", action="store_true",
                    help="Split aggregation by joint type (revolute/prismatic).")
    ap.add_argument("--with-state", action="store_true",
                    help="Include the joint_state_error (State) metric.")
    ap.add_argument("--no-paper", action="store_true",
                    help="Do not show paper reference column.")
    ap.add_argument("--save-per-scene", default=None,
                    help="Optional path to save per-scene CSV.")
    ap.add_argument("--save-markdown", default=None,
                    help="Optional path to save the Markdown table.")
    args = ap.parse_args()

    # decide which metric fields to report
    fields = [f for f, _ in METRIC_FIELDS]
    if not args.with_state:
        fields = [f for f in fields if f != "joint_state_error"]

    df, n = load_all(
        args.base, args.dataset, args.subset, args.iteration,
        set(args.scenes) if args.scenes else None,
        need_joint_type=args.split_joint, output_dir=args.output_dir
    )

    print(f"Found {n} result.csv files for {args.dataset}/{args.subset}\n")

    # drop fields absent from every result.csv (e.g. State-R/P on v2a, whose
    # eval writes only the single-joint joint_state_error)
    fields = [f for f in fields if not df[f].isna().all()]

    # per-scene dump
    show_cols = ["scene"] + (["joint_type"] if args.split_joint else []) + fields
    print("=== Per-scene metrics ===")
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    if args.save_per_scene:
        df[show_cols].to_csv(args.save_per_scene, index=False)
        print(f"[saved] per-scene CSV -> {args.save_per_scene}")

    # build group stats
    if args.split_joint:
        group_stats = {}
        for jt in ["revolute", "prismatic"]:
            sub = df[df["joint_type"] == jt]
            if len(sub) == 0:
                continue
            # prismatic has no Position in the paper; drop distance for that group
            jt_fields = list(fields)
            if jt == "prismatic" and "distance" in jt_fields:
                jt_fields = [f for f in jt_fields if f != "distance"]
            group_stats[jt] = (aggregate(sub, jt_fields), len(sub))
        # for split mode we print two tables (paper ref differs per group)
        md_blocks = []
        for jt, (stats, cnt) in group_stats.items():
            jt_fields = [f for f in fields
                         if not (jt == "prismatic" and f == "distance")]
            ref = None
            if not args.no_paper:
                ref = PAPER_REFERENCE.get("v2a", {}).get(jt)
            block = f"### {jt.capitalize()} Joint Estimation (n={cnt})\n\n" + markdown_table(
                {"Ours (reproduced)": (stats, cnt)}, jt_fields, paper_ref=ref
            )
            md_blocks.append(block)
        markdown = "\n\n".join(md_blocks)
    else:
        stats = aggregate(df, fields)
        ref = None
        if not args.no_paper:
            ref = PAPER_REFERENCE.get(args.dataset)
            if isinstance(ref, dict) and "revolute" in ref:
                ref = None  # don't show split ref in non-split mode
        markdown = markdown_table({"Ours (reproduced)": (stats, n)}, fields, paper_ref=ref)

    print("=== Markdown table ===\n")
    print(markdown)
    print()

    if args.save_markdown:
        with open(args.save_markdown, "w") as f:
            f.write(markdown + "\n")
        print(f"[saved] Markdown -> {args.save_markdown}")


if __name__ == "__main__":
    main()