"""
We integrate Part Articulate Transformer (PAT, from Particulate) into the VideoArtGS
pipeline to predict articulation parameters for the canonical Gaussians.

Pipeline (aligned with the official Particulate inference, RuiningLi/particulate):
  canonical point_cloud.ply (world frame; stored normals, or Open3D-estimated when absent)
    -> rotate to PAT's +Z-up training frame (--pat_up_dir, identity by default)
    -> normalize to the [-0.5, 0.5]^3 bounding box (longest side = 1)
    -> PartField (model_objaverse.ckpt): 40k encoder points -> triplane
       -> 448-dim per-point features queried at the decoder points
    -> optional extra per-point inputs (VideoArtGS extension, PAT/pat_extra_feats.py):
       track_geo (TAPIP3D trajectories from filtered.npz), track_tapip (TAPIP3D hidden
       state), vggt (VGGT tokens); which ones are used is read from the checkpoint's
       sidecar <ckpt>.json written by PAT/PAT_finetune.py (override with --extra_feats)
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
import tqdm
from plyfile import PlyData, PlyElement
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment

from scene import DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.pointnet2_utils import farthest_point_sample, index_points

from particulate.models import PAT_B
from particulate.articulation_utils import plucker_to_axis_point
from particulate.partfield_utils import get_partfield_model, obtain_partfield_feats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pat_extra_feats import NormFrame, compute_scene_extra_feats, extra_feat_dims, parse_extra_names

# Exact configuration of the released checkpoint (configs/particulate-B.yaml).
DEFAULT_MODEL_KWARGS = dict(
    input_dim=448, dropout=0.1, use_normals=True, max_parts=16,
    use_part_id_embedding=True, use_raw_coords=True,
    use_point_features_for_motion_decoding=False,
    num_mask_hypotheses=1, motion_representation='per_point_closest',
)

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
    def __init__(self, args, dataset_args, opt_args, pat_model_path):
        self.args = args
        self.dataset_args = dataset_args
        self.opt_args = opt_args
        self.pat_model_path = pat_model_path

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
        self.frame = NormFrame(self.up_rot, self.norm_center, self.norm_scale)

        # Extra input modalities: configuration comes from the checkpoint sidecar json.
        self.sidecar = self.load_sidecar(os.path.join(ROOT, self.pat_model_path))
        if self.args.extra_feats == "auto":
            self.extra_names = list(self.sidecar.get("extra_feats", []))
        else:
            self.extra_names = parse_extra_names(self.args.extra_feats)
        if self.sidecar and self.sidecar.get("pat_up_dir", self.args.pat_up_dir) != self.args.pat_up_dir:
            print(f"[PAT] ⚠️ checkpoint was fine-tuned with --pat_up_dir {self.sidecar['pat_up_dir']} "
                  f"but inference uses {self.args.pat_up_dir}")
        extra_all = {}
        if self.extra_names:
            print(f"[PAT] Extra input modalities: {extra_feat_dims(self.extra_names)}")
            extra_all, _ = compute_scene_extra_feats(
                self.args.source_path, xyz_world, self.frame, self.extra_names,
                k_frames=self.sidecar.get("k_frames", 8), knn_k=self.sidecar.get("knn_k", 4),
                valid_thr=self.sidecar.get("valid_thr", 0.02))
            if "track_geo" in extra_all:
                print(f"[PAT]   points with a nearby track: {float(extra_all['track_geo'][:, -1].mean()):.2%}")

        num_dec = min(N, self.args.pat_num_points)
        print(f"[PAT] Step 2/4: Sampling {min(N, NUM_ENC_POINTS)} encoder / {num_dec} decoder points out of {N}...")
        enc_idx = np.random.choice(N, min(N, NUM_ENC_POINTS), replace=False)
        dec_idx = np.random.choice(N, num_dec, replace=False)
        extra_dec = {n: torch.from_numpy(v[dec_idx]).float().cuda().unsqueeze(0) for n, v in extra_all.items()}

        pat_results = self.run_pat_inference(
            xyz_norm[enc_idx], xyz_norm[dec_idx], normals_rot[dec_idx], extra_dec)

        # 2. Blend PAT predictions into the original joint_infos.
        print("[PAT] Step 3/4: Bridging PAT physical priors to VideoArtGS architecture...")
        orig_json_path = os.path.join(self.args.source_path, "joint_infos.json")
        with open(orig_json_path, "r") as f:
            orig_joint_infos = json.load(f)
        print(f"[PAT] 🌲Successfully loaded {orig_json_path}, containing {len(orig_joint_infos)} slots")

        bridge = self.args.bridge if self.args.bridge != "auto" else self.sidecar.get("bridge", "legacy")
        print(f"[PAT] Bridge mode: {bridge}")
        if bridge == "pat_vlm":
            joint_infos = self.bridge_pat_vlm(orig_joint_infos, pat_results, xyz_world[dec_idx])
        else:
            joint_infos = self.bridge_pat_to_original(
                orig_joint_infos, pat_results, xyz_world[dec_idx])
        os.makedirs(self.args.model_path, exist_ok=True)
        with open(os.path.join(self.args.model_path, "joint_infos_pat.json"), "w") as f:
            json.dump(joint_infos, f, indent=4)

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

        # 5. Fit the deformation field to the observed 3D tracks, exactly as
        #    init_deform.py does for the non-PAT pipeline.
        #
        #    PAT only supplies the *initial* joint parameters. The baseline
        #    pipeline then optimises the deformation field for 10k iterations
        #    against filtered.npz (track_loss_o2o + track_loss_c2o) and hands
        #    stage 3 a fitted field; that fitting is what takes the joint_infos
        #    axes from ~1.9 deg to the 0.34 deg the baseline reports. Saving at
        #    iteration 1 instead skipped it, so stage 3 started from an unfitted
        #    field no matter how accurate PAT's axes were -- which is why every
        #    PAT run so far collapsed on the same scenes.
        self.fit_deform_to_tracks()

        save_path = self.args.model_path
        print(f"\n[SUCCESS] Pipeline bridge complete! Saving weights to: {save_path}")
        self.deform.save_weights(save_path, iteration=max(1, self.args.iterations))

    def fit_deform_to_tracks(self):
        iters = int(self.args.iterations)
        if iters <= 1:
            print("[PAT] ⚠️ --iterations <= 1: skipping track fitting (stage 3 will "
                  "start from an unfitted deformation field)")
            return

        track = np.load(os.path.join(self.args.source_path, "filtered.npz"))
        track3d = torch.from_numpy(track["coords"]).float().cuda()
        vis3d = torch.from_numpy(track["visibs"]).bool().cuda()

        idx = farthest_point_sample(track3d[0:1], 512).repeat(track3d.shape[0], 1)
        track3d1 = index_points(track3d, idx)
        vis3d1 = index_points(vis3d.unsqueeze(-1), idx).squeeze(-1)

        self.deform.deform.max_window_size = len(track3d)
        self.deform.deform.window_size = len(track3d)

        print(f"[PAT] Fitting deformation field to {track3d.shape[1]} tracks x "
              f"{track3d.shape[0]} frames for {iters} iterations...")
        ema = 0.0
        pbar = tqdm.trange(iters, desc="Track fitting")
        for it in range(1, iters + 1):
            loss = self.deform.deform.track_loss_o2o(track3d, vis3d)
            loss += self.deform.deform.track_loss_c2o(track3d1, vis3d1)
            loss.backward()
            with torch.no_grad():
                ema = 0.4 * loss.item() + 0.6 * ema
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(it)
                self.deform.update(max(0, it))
            pbar.update(1)
            if it % 10 == 0:
                pbar.set_postfix({"loss": f"{ema:.6f}"})
        pbar.close()
        print(f"[PAT] Track fitting done (final EMA loss {ema:.6f})")

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

    @staticmethod
    def load_sidecar(ckpt_path):
        """PAT_finetune.py writes <ckpt>.json next to the checkpoint describing the extra inputs."""
        sidecar_path = os.path.splitext(ckpt_path)[0] + ".json"
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                sidecar = json.load(f)
            print(f"[PAT] Loaded checkpoint sidecar {sidecar_path}: extra_feats={sidecar.get('extra_feats', [])}")
            return sidecar
        return {}

    def run_pat_inference(self, xyz_enc, xyz_dec, normals_dec, extra_dec=None):
        model_kwargs = dict(self.sidecar.get("model_kwargs", DEFAULT_MODEL_KWARGS))
        pat_model = PAT_B(**model_kwargs, extra_feat_dims=extra_feat_dims(self.extra_names)).cuda()

        ckpt_path = os.path.join(ROOT, self.pat_model_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"PAT checkpoint missing at {ckpt_path}. Please download it.")
        pat_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        pat_model.eval()
        print(f"[PAT] Loaded PAT weights from {ckpt_path} (strict; extra branches: {self.extra_names or 'none'})")

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
                extra_feats=extra_dec or None,
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

    def bridge_pat_vlm(self, orig_joint_infos, pat_results, xyz_dec_world):
        """
        Keep the slot LIST of joint_infos.json (count, joint types, order: train.py / render.py
        rebuild the deformation field from that file, and its types are the VLM's), but take
        every slot's parameters from PAT:
          center   = centroid of the PAT part's points (world frame)
          dist_max = 0.2 x max point-to-centroid distance (moving), 1.0 x for the static slot
                     (same convention as data_tools/motion_analysis.py)
          direction / origin = PAT axis (revolute: plucker -> axis point; prismatic: direction)
        PAT parts are assigned to type-compatible slots greedily by size; a slot without a
        compatible PAT part keeps its motion-analysis entry (logged).
        """
        part_ids = pat_results['part_ids']
        plucker = pat_results['revolute_plucker']
        prismatic_axis = pat_results['prismatic_axis']
        is_rev, is_pris = pat_results['is_part_revolute'], pat_results['is_part_prismatic']
        N = len(part_ids)

        min_pts = max(64, int(0.002 * N))
        cands = []
        for pid in np.unique(part_ids):
            sel = part_ids == pid
            n = int(sel.sum())
            if n < min_pts:
                continue
            pts = xyz_dec_world[sel]
            c = pts.mean(axis=0)
            typ = ('r' if bool(is_rev[pid]) else '') + ('p' if bool(is_pris[pid]) else '')
            cands.append(dict(pid=int(pid), n=n, types=typ or 's', centroid=c,
                              radius=float(np.linalg.norm(pts - c, axis=1).max())))
        print(f"[PAT Bridge/vlm] {len(cands)} PAT parts (>= {min_pts} pts): " +
              ", ".join(f"part {c['pid']}:{c['n']}pts/{c['types']}" for c in cands))

        slots = copy.deepcopy(orig_joint_infos)
        assigned = {}
        for k, slot in enumerate(slots):
            t = slot['joint_type']
            if t not in ('r', 'p'):
                continue
            pool = [c for c in cands if t in c['types'] and c['pid'] not in assigned.values()]
            if not pool:
                print(f"[PAT Bridge/vlm] ⚠️ slot {k} ('{t}'): no unassigned PAT part of that type; keeping motion-analysis init")
                continue
            best = max(pool, key=lambda c: c['n'])
            assigned[k] = best['pid']
            slot['center'] = best['centroid'].tolist()
            slot['dist_max'] = float(best['radius'] * 0.2)
            old_dir = np.asarray(slot['direction'], dtype=np.float32)
            if t == 'r':
                axis, point = plucker_to_axis_point(plucker[best['pid']])
                slot['direction'] = (self.up_rot.T @ axis).tolist()
                slot['origin'] = (self.up_rot.T @ (point * self.norm_scale + self.norm_center)).tolist()
            else:
                d = prismatic_axis[best['pid']]
                slot['direction'] = (self.up_rot.T @ (d / (np.linalg.norm(d) + 1e-8))).tolist()
                slot['origin'] = [0.0, 0.0, 0.0]
            new_dir = np.asarray(slot['direction'], dtype=np.float32)
            ang = np.degrees(np.arccos(np.clip(abs(float(old_dir @ new_dir)) /
                                               (np.linalg.norm(old_dir) * np.linalg.norm(new_dir) + 1e-8), 0, 1)))
            print(f"[PAT Bridge/vlm] 🚀 slot {k} ('{t}') <- PAT part {best['pid']} ({best['n']} pts, "
                  f"dist_max {slot['dist_max']:.3f}, angle vs. motion-analysis init {ang:.1f}°)")

        # static slot: everything that is not an assigned moving part
        static_sel = ~np.isin(part_ids, list(assigned.values()))
        for k, slot in enumerate(slots):
            if slot['joint_type'] in ('r', 'p'):
                continue
            if static_sel.sum() < min_pts:
                print(f"[PAT Bridge/vlm] ⚠️ static slot {k}: too few static points; keeping motion-analysis init")
                continue
            pts = xyz_dec_world[static_sel]
            c = pts.mean(axis=0)
            slot['center'] = c.tolist()
            slot['dist_max'] = float(np.linalg.norm(pts - c, axis=1).max())
            print(f"[PAT Bridge/vlm] static slot {k}: {int(static_sel.sum())} pts, dist_max {slot['dist_max']:.3f}")
        return slots

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
    parser.add_argument('--PAT_model_pth', type=str, default="particulate/model_ckpt/pat_model.pt",
                        help="Path to the pre-trained PAT model")
    parser.add_argument('--bridge', type=str, default="auto", choices=["auto", "legacy", "pat_vlm"],
                        help="auto = checkpoint sidecar (default legacy); legacy = override joint_infos "
                             "axes only; pat_vlm = PAT centres/extents/axes for the joint_infos slot list")
    parser.add_argument('--extra_feats', type=str, default="auto",
                        help="extra per-point PAT inputs: 'auto' = read the checkpoint sidecar json, "
                             "'' = none, or a comma list of track_geo,track_tapip,vggt")
    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    safe_state(args.quiet)
    seed_everything(args.seed)

    initializer = PAT_Initializer(
        args=args,
        dataset_args=lp.extract(args),
        opt_args=op.extract(args),
        pat_model_path=args.PAT_model_pth
    )
