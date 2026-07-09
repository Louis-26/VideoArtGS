import numpy as np

# 直接读取你截图里的那个 npy 文件
npy_path = "../outputs/videoartgs/sapien/168/final/train/ours_20000/joint_value.npy"
data = np.load(npy_path)

# 如果遇到维度是 [100, 3, 1] 这种，把它压平变成 [100, 3]
if data.ndim == 3:
    data = data.squeeze(-1)

print("="*60)
print(" 📂 joint_value.npy 内部数据透视")
print("="*60)
print(f"└── 数据维度 (Shape): {data.shape}  (通常是 [帧数, 关节数])")
print(f"└── 数据类型 (Dtype): {data.dtype}")
print("-" * 60)


print(data[2])



# # 统计一下每个关节最大转了多少度
num_joints = data.shape[1]
for j in range(num_joints):
    j_data = data[:, j]
    min_val, max_val = j_data.min(), j_data.max()
    # 顺便帮你把弧度换算成人类直觉的角度 (Degrees)
    print(f"关节 {j} 运动范围: {min_val:.4f} 到 {max_val:.4f} 弧度 (约 {np.degrees(min_val):.1f}° 到 {np.degrees(max_val):.1f}°)")
print("="*60)