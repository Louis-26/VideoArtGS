import os, sys
sys.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import os
import sys
import torch
import numpy as np
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
        
        # 2. Run PAT Inference
        print("[PAT] Step 2/4: Initializing PAT_B and executing 3D articulation inference...")
        pat_results = self.run_pat_inference(xyz_points)
        
        # 3. Construct In-Memory joint_infos
        print("[PAT] Step 3/4: Bridging PAT physical priors to VideoArtGS architecture...")
        joint_infos = self.construct_joint_infos(xyz_points, pat_results)
        
        # Dynamically patch arguments required by VideoArtGS Initialization
        dataset_args.joint_types = [j['joint_type'] for j in joint_infos]
        dataset_args.num_slots = len(dataset_args.joint_types)
        
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
        """Extracts strictly the [N, 3] XYZ coordinates from the PLY file."""
        plydata = PlyData.read(ply_path)
        x = np.asarray(plydata.elements[0].data['x'])
        y = np.asarray(plydata.elements[0].data['y'])
        z = np.asarray(plydata.elements[0].data['z'])
        return np.stack([x, y, z], axis=1)

    def run_pat_inference(self, xyz_points):
        """Executes the pre-trained PAT Transformer on the unorganized point cloud."""
        # Initialize the PAT_B model
        pat_model = PAT_B(input_dim=3, use_raw_coords=True).cuda()
        ckpt_path = "particulate/model_ckpt/model_objaverse.ckpt"
        
        # Ensure the checkpoint exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"PAT checkpoint missing at {ckpt_path}. Please download it.")
            
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        pat_model.load_state_dict(state_dict, strict=False)
        pat_model.eval()

        # Format input tensor [Batch=1, N_points, 3]
        input_tensor = torch.from_numpy(xyz_points).float().cuda().unsqueeze(0)
        dummy_feats = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            # infer() returns a list of dictionaries. We take batch idx 0.
            results = pat_model.infer(input_tensor, dummy_feats)[0]
            
        return results

    def construct_joint_infos(self, xyz_points, pat_results):
        """
        Translates PAT outputs into the exact dictionary schema 
        expected by VideoArtGS's init_from_joint_info() method.
        """
        part_ids = pat_results['part_ids'] # [N]
        unique_parts = np.unique(part_ids)
        
        # Extract physical kinematics if available
        axes = pat_results.get('revolute_plucker', None)
        origins = pat_results.get('closest_point_on_axis', None)
        
        joint_infos = []
        
        # Iterate through detected parts (typically 0 is static, >0 are articulated)
        for part_idx in sorted(unique_parts):
            # 1. Mask points belonging to this specific part
            part_mask = (part_ids == part_idx)
            part_points = xyz_points[part_mask]
            
            # 2. Compute Segment Geometric Properties (Required for seg_model.center)
            center = part_points.mean(axis=0).tolist() if len(part_points) > 0 else [0.0, 0.0, 0.0]
            if len(part_points) > 0:
                dist_max = float(np.linalg.norm(part_points - np.array(center), axis=1).max())
            else:
                dist_max = 1.0

            # 3. Assign Kinematics based on part identity
            if part_idx == 0:
                # Slot 0 is universally treated as the Static Base
                joint_info = {
                    'joint_type': 's',
                    'center': center,
                    'dist_max': dist_max,
                    'origin': [0.0, 0.0, 0.0],
                    'direction': [0.0, 0.0, 0.0]
                }
            else:
                # Articulated Part (Revolute joint mapping)
                # PAT arrays are 0-indexed for active joints, meaning part_idx 1 corresponds to index 0 in the axes array
                axis_idx = part_idx - 1 
                
                # Default fallback if PAT fails to predict an axis
                direction = [0.0, 1.0, 0.0]
                origin = [0.0, 0.0, 0.0]
                
                if axes is not None and len(axes) > axis_idx:
                    # Extract the first 3 dimensions of the Plucker coordinate for the direction vector
                    direction = axes[axis_idx][:3].tolist()
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
    # Setup command line argument parser
    parser = ArgumentParser(description="PAT-Driven Zero-Shot Deformation Initialization")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser) # Kept for signature compatibility
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    safe_state(args.quiet)
    seed_everything(args.seed)

    # Execute the Zero-Shot Pipeline
    initializer = PAT_Initializer(
        args=args, 
        dataset_args=lp.extract(args), 
        opt_args=op.extract(args)
    )