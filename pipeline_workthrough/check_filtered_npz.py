import numpy as np
import os

# 你的 npz 文件路径
file_path = "../data/videoartgs/sapien/168/filtered.npz"

print("=" * 50)
print("🔍 开始检查 3D 轨迹文件 (filtered.npz)")
print("=" * 50)

# 1. 检查文件是否存在，防止路径坑
if not os.path.exists(file_path):
    print(f"❌ 报错: 找不到文件 '{file_path}'")
    print(f"💡 你当前运行脚本的工作目录是: {os.getcwd()}")
    print("👉 请确保你在正确的目录下运行，或者改用绝对路径。")
else:
    print(f"✅ 成功找到文件: {file_path}\n")
    
    # 2. 加载数据
    data = np.load(file_path)
    keys = list(data.keys())
    print(f"📦 文件内包含的矩阵 Keys: {keys}\n")
    
    # 3. 详细检查每个矩阵的内容
    for key in keys:
        arr = data[key]
        print(f"--- 矩阵: [{key}] ---")
        print(f"📐 维度 (Shape): {arr.shape}")
        print(f"🔤 类型 (Dtype): {arr.dtype}")
        
        # 4. 聪明地打印一点样本，防止终端炸裂
        if key == 'coords':
            # 打印第 0 帧，前 2 个点的 XYZ 坐标
            print("👁️  数据样本 (第 0 帧, 前 2 个点的 3D 坐标):")
            print(arr[0, :2, :])
        elif key == 'visibs':
            # 打印第 0 帧，前 5 个点的可见性 (True/False)
            print("👁️  数据样本 (第 0 帧, 前 5 个点的可见性):")
            print(arr[0, :5])
        else:
            # 如果原作者以后加了其他 key，兜底打印一下
            print(f"👁️  数据样本:\n{arr.flatten()[:5]}")
            
        print("")
        
print("✅ 检查完毕！")