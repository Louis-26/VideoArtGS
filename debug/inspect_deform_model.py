#!/usr/bin/env python3
"""
Inspect the complete architecture, parameter shapes, input/output dimensions,
and memory footprint of VideoArtGS Deformation Field (DeformModel), including:
  1. HybridSeg (ProgressiveBandHashGrid, motion_grid, residual MLP, motion_mlp, ellipsoids)
  2. ArticulationModel (joint origins, directions, dual quaternion constants)
  3. TimeModel (TimePrediction / TimeInterpolation / PolynomialEmbedding)

All comments and terminal outputs are strictly in English.
"""

import os
import sys
import argparse
import torch


def find_default_checkpoint():
    """Search for the most representative deform.pth checkpoint in outputs/."""
    candidates = [
        "outputs/videoartgs/realscan/cab_1r_1p/final/deform/iteration_best/deform.pth",
        "outputs/videoartgs/sapien/1280/init/deform/iteration_1/deform.pth",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback search across outputs directory
    for root, _, files in os.walk("outputs"):
        if "deform.pth" in files:
            return os.path.join(root, "deform.pth")

    return None


def format_shape(shape):
    """Format tensor shape as [dim0, dim1, ...] string."""
    return str(list(shape))


def format_shape_calc(shape):
    """Format dimension multiplication expression."""
    if len(shape) == 0:
        return "1 (scalar)"
    return " * ".join(map(str, shape)) + f" = {numel_calc(shape):,}"


def numel_calc(shape):
    res = 1
    for s in shape:
        res *= s
    return res


def infer_io_description(param_name, shape):
    """Infer input/output dimensions from parameter name and shape."""
    if len(shape) == 2 and "weight" in param_name:
        # PyTorch Linear weight: [out_features, in_features]
        return f"In: {shape[1]:<4} ➔  Out: {shape[0]:<4}"
    elif len(shape) == 1 and "bias" in param_name:
        return f"Bias       ➔  Out: {shape[0]:<4}"
    elif len(shape) == 1:
        return f"1D Vector  ➔  Len: {shape[0]:<4}"
    elif len(shape) == 2:
        return f"2D Matrix  ➔  [{shape[0]}, {shape[1]}]"
    else:
        return f"Tensor     ➔  {shape}"


def build_module_tree(state_dict):
    """Build a nested hierarchical dictionary tree from state_dict keys."""
    tree = {"_size": 0, "_children": {}}

    for key, tensor in state_dict.items():
        parts = key.split(".")
        curr = tree
        for part in parts[:-1]:
            curr = curr["_children"].setdefault(part, {"_size": 0, "_children": {}})

        leaf_name = parts[-1]
        num_elements = tensor.numel()
        shape = list(tensor.shape)

        curr["_children"][leaf_name] = {
            "_size": num_elements,
            "_shape": shape,
            "_shape_calc": format_shape_calc(shape),
            "_io_str": infer_io_description(key, shape),
            "_tensor": tensor,
        }

    def calc_sizes(node):
        total = 0
        if "_children" in node:
            for child in node["_children"].values():
                total += calc_sizes(child)
            node["_size"] = total
        else:
            total = node["_size"]
        return total

    calc_sizes(tree)
    return tree


def print_module_tree(node, name="root", prefix=""):
    """Recursively display the module architecture tree with visual box characters."""
    connector = "└── " if name != "root" else ""

    if "_shape_calc" in node:
        # Leaf tensor node
        print(
            f"{prefix}{connector}{name:<16} "
            f"| [ {node['_io_str']:<22} ] "
            f"| 📐 {node['_shape_calc']}"
        )
    else:
        # Intermediate module node
        size_str = f"{node['_size']:,}"
        mem_mb = node["_size"] * 4 / (1024 ** 2)
        print(
            f"{prefix}{connector}{name} "
            f"[📦 Module Parameters: {size_str} ({mem_mb:.2f} MB)]"
        )

        children = node.get("_children", {})
        items = list(children.items())
        for idx, (child_name, child_node) in enumerate(items):
            is_last = idx == len(items) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_module_tree(child_node, child_name, new_prefix)


def inspect_checkpoint(checkpoint_path):
    print("=" * 115)
    print(f" 🔍 VideoArtGS Deformation Field Inspector")
    print(f" Target Checkpoint: {checkpoint_path}")
    print("=" * 115)

    if not os.path.isfile(checkpoint_path):
        print(f"[Error] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Load state dict on CPU (portable, does not require CUDA)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 1. Print Hierarchical Architecture Tree
    print("\n" + "=" * 115)
    print(" 🌳 1. Hierarchical Architecture Data Flow Tree")
    print("=" * 115)
    tree = build_module_tree(state_dict)
    print_module_tree(tree, "DeformModel (VideoArtGS)")

    # 2. Detailed Parameter Table
    print("\n" + "=" * 115)
    print(" 📊 2. Complete Parameter List & Value Probe")
    print("=" * 115)
    print(
        f"{'Parameter Name':<50} "
        f"| {'Shape':<18} "
        f"| {'Count':<12} "
        f"| {'Values Probe (Min / Mean / Max / Sample)'}"
    )
    print("-" * 115)

    category_counts = {
        "Hash Grids (tinycudann)": 0,
        "Segmentation MLPs (Dynamic & Static)": 0,
        "Ellipsoid Geometric Priors": 0,
        "Articulation & Time Models": 0,
        "Others / Buffers": 0,
    }

    total_params = 0

    for key, tensor in state_dict.items():
        numel = tensor.numel()
        total_params += numel
        shape_str = format_shape(tensor.shape)

        # Categorize
        if "grid.encoding.params" in key:
            category_counts["Hash Grids (tinycudann)"] += numel
        elif "mlp" in key and "motion_mlp" not in key and "time_model" not in key:
            category_counts["Segmentation MLPs (Dynamic & Static)"] += numel
        elif "motion_mlp" in key:
            category_counts["Segmentation MLPs (Dynamic & Static)"] += numel
        elif any(k in key for k in ["center", "logscale", "rot"]):
            category_counts["Ellipsoid Geometric Priors"] += numel
        elif "art_model" in key:
            category_counts["Articulation & Time Models"] += numel
        else:
            category_counts["Others / Buffers"] += numel

        # Numerical probe
        if tensor.is_floating_point():
            f_tensor = tensor.float()
            val_min = f_tensor.min().item()
            val_max = f_tensor.max().item()
            val_mean = f_tensor.mean().item()
            probe_str = f"min={val_min:+.4f}, mean={val_mean:+.4f}, max={val_max:+.4f}"
        else:
            probe_str = f"int tensor (shape: {shape_str})"

        print(
            f"{key:<50} "
            f"| {shape_str:<18} "
            f"| {numel:<12,} "
            f"| {probe_str}"
        )

    print("-" * 115)

    # 3. Categorical Parameter Summary
    print("\n" + "=" * 115)
    print(" 📦 3. Subsystem Parameter Breakdown & Memory Footprint")
    print("=" * 115)
    for cat_name, count in category_counts.items():
        mem_mb = count * 4 / (1024 ** 2)
        pct = (count / total_params) * 100 if total_params > 0 else 0
        print(f"  • {cat_name:<42} : {count:>12,} params ({mem_mb:>7.2f} MB)  [{pct:>6.2f}%]")

    total_mem_mb = total_params * 4 / (1024 ** 2)
    print("-" * 115)
    print(f"  ✨ Total Trainable Parameters in DeformModel  : {total_params:>12,} params")
    print(f"  💾 Total Estimated Float32 Memory Footprint    : {total_mem_mb:>12.2f} MB")
    print("=" * 115)
    print("\n[Done] Model inspection complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect VideoArtGS Deformation Field Architecture")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to deform.pth checkpoint file (default: auto-detected)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_default_checkpoint()
        if checkpoint_path is None:
            print("[Error] No default deform.pth found. Please specify via --checkpoint <path>.")
            sys.exit(1)

    inspect_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()

