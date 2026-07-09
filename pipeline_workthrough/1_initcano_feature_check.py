import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.inpert(0, os.path.abspath(os.path.join(current_dir, "../")))

from scene.gaussian_model import GaussianModel

# 你的路径没问题，直接用
ply_path = "/scratch4/enalisn1/ylu174/VideoArtGS/outputs/videoartgs/sapien/168/init/point_cloud/iteration_20000/point_cloud.ply"

gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply(ply_path)

# 提取核心参数 Tensor
xyz = gaussians.get_xyz              # [N, 3]
rot = gaussians.get_rotation         # [N, 4]
scale = gaussians.get_scaling        # [N, 3]
opacity = gaussians.get_opacity      # [N, 1]
pe = gaussians.get_feature           # [N, 16] (pegmentation feature)
if len(pe.shape) > 2:
    pe = pe.view(pe.shape[0], -1)
print(f"语义特征: {pe}") 
real_sh = gaussians.get_sh
if len(real_sh.shape) > 2:
    real_sh = real_sh.view(real_sh.shape[0], -1)
print(f"真正的 SH 颜色: {real_sh.shape}") # 绝对是 [42458, 48]



print(f"--- Canonical Gaussian 核心结构 ---")
print(f"总点数 N = {xyz.shape[0]}")
print(f"1. Position (mu) shape:   {xyz.shape}")
print(f"2. Rotation (q) shape:    {rot.shape}")
print(f"3. Scale (s) shape:       {scale.shape}")
print(f"4. Opacity (alpha) shape: {opacity.shape}")
print(f"5. SH feature shape:      {real_sh.shape}") 
print(f"6. part feature: {pe.shape}") 

# 拼接成 Transformer 可以接收的特征矩阵
full_features = torch.cat([xyz, rot, scale, opacity, real_sh, pe], dim=-1)
print(f"\n拼接后的完整特征矩阵 shape: {full_features.shape}")