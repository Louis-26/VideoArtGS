"""
We integrate Part Articulate Transformer (PAT, from Particulate) into the VideoArtGS
pipeline to predict articulation parameters for the canonical Gaussians.

Pipeline (aligned with the official Particulate inference, RuiningLi/particulate):
  canonical point_cloud.ply (world frame; stored normals, or Open3D-estimated when absent)
    -> rotate to PAT's +Z-up training frame (--pat_up_dir, identity by default)
    -> normalize to the [-0.5, 0.5]^3 bounding box (longest side = 1)
    -> PartField (model_objaverse.ckpt): 40k encoder points -> triplane
       -> 448-dim per-point features queried at the decoder points
    -> PAT (pat_model.pt): part segmentation + per-part joint parameters
    -> denormalize joint origins back to the world frame
    -> match PAT parts to the GT joint slots by centroid distance and
       override each slot's direction/origin
    -> zero-shot DeformModel initialization
"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import copy
import json
from argparse import ArgumentParser

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment

from scene import DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state

from particulate.models import PAT_B
from particulate.articulation_utils import plucker_to_axis_point
from particulate.partfield_utils import get_partfield_model, obtain_partfield_feats

# Number of points fed to the PartField encoder (official Particulate infer.py value).
NUM_ENC_POINTS = 40000

# Rotation matrices copied verbatim from the official Particulate infer.py
# (predict_mesh): points are mapped into PAT's +Z-up training frame via xyz @ R.T,
# where the key names which world axis currently points up.
UP_DIR_ROTATIONS = {
    "X":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
    "-X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
    "Y":  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "-Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
    "Z":  np.eye(3, dtype=np.float32),
    "-Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
}


class PAT_Initializer:
    def __init__(self, args, dataset_args, opt_args):
        self.args = args
        self.dataset_args = dataset_args
        self.opt_args = opt_args

        # 1. Load canonical point cloud (world frame) together with its normals.
        ply_path = os.path.join(self.args.source_path, "point_cloud.ply")
        print(f"\n[PAT] Step 1/4: Loading canonical point cloud from: {ply_path}")
        xyz_world, normals_world = self.load_ply_xyz_normals(ply_path)
        N = xyz_world.shape[0]

        # Rotate into PAT's +Z-up training frame (identity for --pat_up_dir Z);
        # predicted directions/origins are rotated back in bridge_pat_to_original.
        self.up_rot = UP_DIR_ROTATIONS[self.args.pat_up_dir]
        xyz_rot = xyz_world @ self.up_rot.T
        normals_rot = normals_world @ self.up_rot.T

        # PAT/PartField are trained on objects normalized to the [-0.5, 0.5]^3
        # bounding box; joint origins predicted in this frame are mapped back below.
        bbmin, bbmax = xyz_rot.min(axis=0), xyz_rot.max(axis=0)
        self.norm_center = (bbmin + bbmax) / 2
        self.norm_scale = float((bbmax - bbmin).max())
        xyz_norm = (xyz_rot - self.norm_center) / self.norm_scale

        num_dec = min(N, self.args.pat_num_points)
        print(f"[PAT] Step 2/4: Sampling {min(N, NUM_ENC_POINTS)} encoder / {num_dec} decoder points out of {N}...")
        enc_idx = np.random.choice(N, min(N, NUM_ENC_POINTS), replace=False)
        dec_idx = np.random.choice(N, num_dec, replace=False)

        pat_results = self.run_pat_inference(
            xyz_norm[enc_idx], xyz_norm[dec_idx], normals_rot[dec_idx])

        # 2. Blend PAT predictions into the original joint_infos.
        print("[PAT] Step 3/4: Bridging PAT physical priors to VideoArtGS architecture...")
        orig_json_path = os.path.join(self.args.source_path, "joint_infos.json")
        with open(orig_json_path, "r") as f:
            orig_joint_infos = json.load(f)
        print(f"[PAT] 🌲Successfully loaded {orig_json_path}, containing {len(orig_joint_infos)} slots")

        joint_infos = self.bridge_pat_to_original(
            orig_joint_infos, pat_results, xyz_world[dec_idx])

        self.save_segmentation_ply(xyz_world[dec_idx], pat_results['part_ids'])

        # 3. Feed the dataset into the model after aligning the data structure.
        dataset_args.joint_types = [j['joint_type'] for j in joint_infos]
        dataset_args.num_slots = len(joint_infos)
        dataset_args.joint_info_path = orig_json_path

        self.args.num_slots = len(joint_infos)
        self.args.joint_info_path = orig_json_path

        # 4. Initialize DeformModel & inject priors natively.
        print("[PAT] Step 4/4: Injecting priors via native DeformModel interface...")
        self.deform = DeformModel(dataset_args)
        self.deform.init_from_joint_info(joint_infos, init_joint_info=True, init_center=True)
        self.deform.train_setting(self.opt_args)

        # 5. Save the zero-shot initialized deform.pth.
        save_path = self.args.model_path
        print(f"\n[SUCCESS] Pipeline bridge complete! Saving zero-shot weights to: {save_path}")
        self.deform.save_weights(save_path, iteration=1)

    def load_ply_xyz_normals(self, ply_path):
        data = PlyData.read(ply_path).elements[0].data
        xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

        # PAT (use_normals=True) was trained with true mesh normals; PLYs written by
        # some pipelines (e.g. data_tools/process_v2a.py) carry no or all-zero normals.
        normals = None
        if {'nx', 'ny', 'nz'} <= set(data.dtype.names):
            normals = np.stack([data['nx'], data['ny'], data['nz']], axis=1).astype(np.float32)
            zero_frac = float((np.linalg.norm(normals, axis=1) <= 1e-6).mean())
            if zero_frac > 0.5:
                print(f"[PAT] ⚠️ {zero_frac:.0%} of the stored normals are zero; discarding them.")
                normals = None
            elif zero_frac > 0.01:
                print(f"[PAT] ⚠️ {zero_frac:.1%} of the stored normals are zero (substituted with [0,0,1]).")
        else:
            print(f"[PAT] ⚠️ {os.path.basename(ply_path)} has no nx/ny/nz fields.")

        if normals is None:
            print(f"[PAT] ⚠️ Estimating normals with Open3D on {len(xyz)} points (expect reduced PAT quality).")
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
            pcd.orient_normals_consistent_tangent_plane(16)
            normals = np.asarray(pcd.normals, dtype=np.float32)

        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm > 1e-6,
                           normals / np.maximum(norm, 1e-8),
                           np.array([0.0, 0.0, 1.0], dtype=np.float32))
        return xyz, normals

    def run_pat_inference(self, xyz_enc, xyz_dec, normals_dec):
        # Exact configuration of the released checkpoint (configs/particulate-B.yaml).
        pat_model = PAT_B(
            input_dim=448,
            dropout=0.1,
            use_normals=True,
            max_parts=16,
            use_part_id_embedding=True,
            use_raw_coords=True,
            use_point_features_for_motion_decoding=False,
            num_mask_hypotheses=1,
            motion_representation='per_point_closest',
        ).cuda()

        ckpt_path = os.path.join(ROOT, "particulate/model_ckpt/pat_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"PAT checkpoint missing at {ckpt_path}. Please download it.")
        pat_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        pat_model.eval()
        print(f"[PAT] Loaded PAT weights from {ckpt_path} (strict)")

        xyz_enc_t = torch.from_numpy(xyz_enc).float().cuda().unsqueeze(0)
        xyz_dec_t = torch.from_numpy(xyz_dec).float().cuda().unsqueeze(0)
        normals_t = torch.from_numpy(normals_dec).float().cuda().unsqueeze(0)

        print("[PAT] Extracting PartField features (448-dim) for the decoder points...")
        partfield_model = get_partfield_model(device="cuda")
        feats = obtain_partfield_feats(partfield_model, xyz_enc_t, xyz_dec_t).float()
        assert feats.shape[-1] == 448, f"Unexpected PartField feature dim: {feats.shape}"
        del partfield_model
        torch.cuda.empty_cache()

        print("[PAT] Executing 3D articulation inference...")
        with torch.no_grad():
            results = pat_model.infer(
                xyz=xyz_dec_t, feats=feats, normals=normals_t,
                min_part_confidence=0.0)[0]
        return results

    def bridge_pat_to_original(self, orig_joint_infos, pat_results, xyz_dec_world):
        """
        Keep slot count / joint types / segmentation init from the original
        joint_infos and override direction & origin with PAT predictions.
        GT slots and PAT parts live in different index spaces, so each moving
        GT joint is matched to the PAT part with the closest centroid.
        """
        part_ids = pat_results['part_ids']                    # (N_dec,)
        plucker = pat_results['revolute_plucker']             # (num_parts, 6)
        prismatic_axis = pat_results['prismatic_axis']        # (num_parts, 3)
        is_rev = pat_results['is_part_revolute']
        is_pris = pat_results['is_part_prismatic']

        updated_joint_infos = copy.deepcopy(orig_joint_infos)
        moving_joints = [j for j in updated_joint_infos if j['joint_type'] in ('r', 'p')]

        # Fragment parts are too small for a trustworthy centroid/axis.
        unique_parts = np.unique(part_ids)
        min_pts = max(16, int(0.001 * len(part_ids)))
        dropped = [int(pid) for pid in unique_parts if (part_ids == pid).sum() < min_pts]
        if dropped:
            print(f"[PAT Bridge] Dropping fragment part(s) {dropped} (fewer than {min_pts} points)")
        unique_parts = np.array([pid for pid in unique_parts if pid not in dropped])
        if len(unique_parts) == 0 or len(moving_joints) == 0:
            return updated_joint_infos

        centroids = np.stack([xyz_dec_world[part_ids == pid].mean(axis=0) for pid in unique_parts])
        print(f"[PAT Bridge] PAT predicted {len(unique_parts)} parts; centroids (world):")
        for pid, c in zip(unique_parts, centroids):
            n_pts = int((part_ids == pid).sum())
            print(f"  part {pid}: {n_pts} pts, centroid [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}], "
                  f"revolute={bool(is_rev[pid])}, prismatic={bool(is_pris[pid])}")

        # One-to-one assignment between GT moving joints and PAT parts. The raw
        # cost is centroid distance; type-incompatible pairs (an 'r' joint with a
        # non-revolute part, a 'p' joint with a non-prismatic part — the static
        # base part is neither) get a large finite penalty so they are only chosen
        # when no compatible part is left, and are then rejected below.
        joint_centers = np.array([j['center'] for j in moving_joints], dtype=np.float32)
        dist = np.linalg.norm(joint_centers[:, None] - centroids[None], axis=-1)
        compatible = np.stack([
            np.array([bool(is_rev[pid]) if j['joint_type'] == 'r' else bool(is_pris[pid])
                      for pid in unique_parts])
            for j in moving_joints])
        BIG = 1e6
        rows, cols = linear_sum_assignment(dist + np.where(compatible, 0.0, BIG))

        max_dist = self.args.pat_match_dist_ratio * self.norm_scale
        for k, c in zip(rows, cols):
            joint, pid = moving_joints[k], unique_parts[c]

            if not compatible[k, c]:
                print(f"[PAT Bridge] ⚠️ GT joint {k} ('{joint['joint_type']}') has no type-compatible PAT part left; keeping original init")
                continue
            if dist[k, c] > max_dist:
                print(f"[PAT Bridge] ⚠️ GT joint {k} ('{joint['joint_type']}'): nearest compatible PAT part {pid} "
                      f"is too far (dist {dist[k, c]:.3f} > {max_dist:.3f}); keeping original init")
                continue

            old_dir = np.asarray(joint['direction'], dtype=np.float32)
            if joint['joint_type'] == 'r':
                axis, point = plucker_to_axis_point(plucker[pid])
                # Undo normalization, then undo the up-dir rotation, back to world frame.
                joint['direction'] = (self.up_rot.T @ axis).tolist()
                joint['origin'] = (self.up_rot.T @ (point * self.norm_scale + self.norm_center)).tolist()
            else:  # 'p': only the direction matters (origin is unused for prismatic joints)
                d = prismatic_axis[pid]
                joint['direction'] = (self.up_rot.T @ (d / (np.linalg.norm(d) + 1e-8))).tolist()

            new_dir = np.asarray(joint['direction'], dtype=np.float32)
            denom = (np.linalg.norm(old_dir) * np.linalg.norm(new_dir) + 1e-8)
            angle = np.degrees(np.arccos(np.clip(abs(float(old_dir @ new_dir)) / denom, 0.0, 1.0)))
            print(f"[PAT Bridge] 🚀 GT joint {k} ('{joint['joint_type']}') <- PAT part {pid} "
                  f"(centroid dist {dist[k, c]:.3f}, angle vs. original init {angle:.1f}°)")

        if len(rows) < len(moving_joints):
            unmatched = set(range(len(moving_joints))) - set(rows.tolist())
            print(f"[PAT Bridge] ⚠️ {len(unmatched)} moving joint(s) {sorted(unmatched)} had no PAT part; keeping original init")

        return updated_joint_infos

    def save_segmentation_ply(self, xyz, part_ids, filename="pat_segmentation.ply"):
        """Dump a color-coded segmentation point cloud for visual inspection."""
        os.makedirs(self.args.model_path, exist_ok=True)
        out_path = os.path.join(self.args.model_path, filename)
        palette = np.random.RandomState(0).randint(0, 255, (int(part_ids.max()) + 1, 3), dtype=np.uint8)
        colors = palette[part_ids.astype(np.int64)]
        vertex = np.zeros(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertex['red'], vertex['green'], vertex['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(out_path)
        print(f"[PAT] Segmentation preview saved to: {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="PAT-Driven Zero-Shot Deformation Initialization")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pat_num_points', type=int, default=65536,
                        help="Points fed to the PAT decoder (official default is 102400)")
    parser.add_argument('--pat_up_dir', type=str, default='Z', choices=list(UP_DIR_ROTATIONS),
                        help="Which world axis points up; input is rotated to PAT's +Z-up training frame")
    parser.add_argument('--pat_match_dist_ratio', type=float, default=0.3,
                        help="Max joint-center-to-part-centroid distance for an accepted match, as a fraction of the object's longest bbox side")

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    safe_state(args.quiet)
    seed_everything(args.seed)

    initializer = PAT_Initializer(
        args=args,
        dataset_args=lp.extract(args),
        opt_args=op.extract(args)
    )
