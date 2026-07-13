"""
We integrate Part Articulate Transformer into VideoArtGS pipeline to predict the articulation parameters from Gaussian primitives.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import torch
import numpy as np
from scipy.spatial import cKDTree 
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from plyfile import PlyData

from scene import DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state

# Import PAT architecture from the copied particulate package
from particulate.models import PAT_B

class PAT_Initializer:
    def __init__(self, args, dataset_args, opt_args):
        self.args = args
        self.dataset_args = dataset_args
        self.opt_args = opt_args
        
        # 1. Load Canonical Point Cloud
        ply_path = os.path.join(self.args.source_path, "point_cloud.ply")
        print(f"\n[PAT] Step 1/4: Loading canonical Gaussians from: {ply_path}")
        xyz_points = self.load_ply_xyz(ply_path)
        
        # downsampling to avoid out of memory issue
        N = xyz_points.shape[0]
        num_samples = 2048 # Transformer safe boundary
        
        if N > num_samples:
            print(f"[PAT] Step 2/4: Point cloud too large ({N}). Downsampling to {num_samples} for inference...")
            indices = np.random.choice(N, num_samples, replace=False)
            sampled_xyz = xyz_points[indices]
        else:
            print(f"[PAT] Step 2/4: Point cloud size ({N}) is safe for inference.")
            sampled_xyz = xyz_points

        print("[PAT] Executing 3D articulation inference on sampled points...")
        pat_results_sampled = self.run_pat_inference(sampled_xyz)
        pat_results = pat_results_sampled.copy()
        
        if N > num_samples:
            print(f"[PAT] Upsampling (Broadcasting) results back to the full {N} Gaussians via KDTree...")
            tree = cKDTree(sampled_xyz)
            _, nearest_idx = tree.query(xyz_points, k=1)
            
            sampled_part_ids = pat_results_sampled['part_ids']
            if torch.is_tensor(sampled_part_ids):
                sampled_part_ids = sampled_part_ids.cpu().numpy()
                
            full_part_ids = sampled_part_ids[nearest_idx]
            pat_results['part_ids'] = full_part_ids
        else:
            if torch.is_tensor(pat_results['part_ids']):
                pat_results['part_ids'] = pat_results['part_ids'].cpu().numpy()
        
        # 3. Construct In-Memory joint_infos
        print("[PAT] Step 3/4: Bridging PAT physical priors to VideoArtGS architecture...")
        joint_infos = self.construct_joint_infos(xyz_points, pat_results)
        
        import json
        # 1. 规规矩矩读取原始、干净的数据集图纸
        orig_json_path = os.path.join(self.args.source_path, "joint_infos.json")
        with open(orig_json_path, "r") as f:
            orig_joint_infos = json.load(f)
            
        print(f"[PAT] 🌲 成功加载原始数据集图纸: {orig_json_path}，包含 {len(orig_joint_infos)} 个 Slots")

        # 2. 调用修改后的融合函数，把 PAT 的预测值揉进原始图纸里，不写盘！
        joint_infos = self.bridge_pat_to_original(orig_joint_infos, pat_results)
        
        # 3. 此时内存里的配置已经和原始数据集的结构完全一致了，直接喂给模型
        dataset_args.joint_types = [j['joint_type'] for j in joint_infos]
        dataset_args.num_slots = len(joint_infos)
        dataset_args.joint_info_path = orig_json_path # 依然指向原始路径
        
        self.args.num_slots = len(joint_infos)
        self.args.joint_info_path = orig_json_path
        
        # 4. Initialize DeformModel & Inject Priors natively
        print("[PAT] Step 4/4: Injecting priors via native DeformModel interface...")
        self.deform = DeformModel(dataset_args)
        self.deform.init_from_joint_info(joint_infos, init_joint_info=True, init_center=True)
        self.deform.train_setting(self.opt_args)
        
        # 5. Save the Zero-Shot Initialized deform.pth
        save_path = self.args.model_path
        print(f"\n[SUCCESS] Pipeline bridge complete! Saving zero-shot weights to: {save_path}")
        self.deform.save_weights(save_path, iteration=1)

    def load_ply_xyz(self, ply_path):
        plydata = PlyData.read(ply_path)
        x = np.asarray(plydata.elements[0].data['x'])
        y = np.asarray(plydata.elements[0].data['y'])
        z = np.asarray(plydata.elements[0].data['z'])
        return np.stack([x, y, z], axis=1)

    def run_pat_inference(self, xyz_points):
        pat_model = PAT_B(input_dim=3, use_raw_coords=True).cuda()
        ckpt_path = "../particulate/model_ckpt/model_objaverse.ckpt"
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"PAT checkpoint missing at {ckpt_path}. Please download it.")
            
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        pat_model.load_state_dict(state_dict, strict=False)
        pat_model.eval()

        input_tensor = torch.from_numpy(xyz_points).float().cuda().unsqueeze(0)
        dummy_feats = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            results = pat_model.infer(input_tensor, dummy_feats)[0]
            
        return results

    def construct_joint_infos(self, xyz_points, pat_results):
        part_ids = pat_results['part_ids'] 
        unique_parts = np.unique(part_ids)
        
        axes = pat_results.get('revolute_plucker', None)
        origins = pat_results.get('closest_point_on_axis', None)
        
        joint_infos = []
        
        for part_idx in sorted(unique_parts):
            part_mask = (part_ids == part_idx)
            part_points = xyz_points[part_mask]
            
            center = part_points.mean(axis=0).tolist() if len(part_points) > 0 else [0.0, 0.0, 0.0]
            if len(part_points) > 0:
                dist_max = float(np.linalg.norm(part_points - np.array(center), axis=1).max())
            else:
                dist_max = 1.0

            if part_idx == 0:
                joint_info = {
                    'joint_type': 's',
                    'center': center,
                    'dist_max': dist_max,
                    'origin': [0.0, 0.0, 0.0],
                    'direction': [0.0, 0.0, 0.0]
                }
            else:
                axis_idx = part_idx - 1 
                direction = [0.0, 1.0, 0.0]
                origin = [0.0, 0.0, 0.0]
                
                if axes is not None and len(axes) > axis_idx:
                    direction = axes[axis_idx][:3].tolist()
                    if origins is not None and len(origins) > axis_idx:
                        origin = origins[axis_idx].tolist()
                    
                joint_info = {
                    'joint_type': 'r',
                    'center': center,
                    'dist_max': dist_max,
                    'origin': origin,
                    'direction': direction
                }
                
            joint_infos.append(joint_info)
            
        return joint_infos

if __name__ == "__main__":
    parser = ArgumentParser(description="PAT-Driven Zero-Shot Deformation Initialization")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser) 
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    safe_state(args.quiet)
    seed_everything(args.seed)

    initializer = PAT_Initializer(
        args=args, 
        dataset_args=lp.extract(args), 
        opt_args=op.extract(args)
    )