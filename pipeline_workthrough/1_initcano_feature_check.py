import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../")))

from scene.gaussian_model import GaussianModel

ply_path = "/scratch4/enalisn1/ylu174/VideoArtGS/outputs/videoartgs/sapien/168/init/point_cloud/iteration_20000/point_cloud.ply"

gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply(ply_path)

# Extract core Tensor
xyz = gaussians.get_xyz              # [N, 3]
rot = gaussians.get_rotation         # [N, 4]
scale = gaussians.get_scaling        # [N, 3]
opacity = gaussians.get_opacity      # [N, 1]
pe = gaussians.get_feature           # [N, 16] (segmentation feature)
if len(pe.shape) > 2:
    pe = pe.view(pe.shape[0], -1)
print(f"semantic features: {pe}") 
real_sh = gaussians.get_sh
if len(real_sh.shape) > 2:
    real_sh = real_sh.view(real_sh.shape[0], -1)
print(f"SH color: {real_sh.shape}")  



print(f"--- Canonical Gaussian Core Architecture ---")
print(f"Total point number N = {xyz.shape[0]}")
print(f"1. Position (mu) shape:   {xyz.shape}")
print(f"2. Rotation (q) shape:    {rot.shape}")
print(f"3. Scale (s) shape:       {scale.shape}")
print(f"4. Opacity (alpha) shape: {opacity.shape}")
print(f"5. SH feature shape:      {real_sh.shape}") 
print(f"6. part feature: {pe.shape}") 

full_features = torch.cat([xyz, rot, scale, opacity, real_sh, pe], dim=-1)
print(f"\nComplete Concantenated feature matrix shape: {full_features.shape}")