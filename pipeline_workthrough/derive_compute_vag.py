import json
import numpy as np
import os

def get_vag_params(model_dir):
    # 1. 静态参数读取 (来自 eval.py 调用的 joint_info.json)
    json_path = os.path.join(model_dir, 'joint_info.json')
    with open(json_path, 'r') as f:
        joint_infos = json.load(f)
    
    # eval.py 只处理 joint_type != 's' 的部分
    # 按照 eval.py 的逻辑，它只取活动关节
    active_joints = [j for j in joint_infos if j.get('joint', 'fixed') != 'heavy']
    
    # 2. 动态参数读取 (来自 eval.py 的第 48 行逻辑)
    npy_path = os.path.join(model_dir, 'joint_value.npy')
    raw_states = np.load(npy_path)
    # 取最后一帧并平移到从 0 开始
    final_state = raw_states[-1].squeeze() - raw_states[0].squeeze()
    
    return active_joints, final_state

# --- 运行提取 ---
model_dir = '../outputs/videoartgs/sapien/168/final/train/ours_20000/'
joints, state = get_vag_params(model_dir)

print(f"检测到的活动关节数: {len(joints)}")
for i, j in enumerate(joints):
    print(f"关节 {i} Axis: {j.get('jointData', {}).get('axis')}")
    print(f"关节 {i} Position: {j.get('center')}")
    print(f"关节 {i} 最终状态预测: {state[i] if state.ndim > 0 else state}")