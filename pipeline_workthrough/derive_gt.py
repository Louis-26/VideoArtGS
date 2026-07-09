import json
import numpy as np
from pathlib import Path

def extract_ground_truth(scene_dir):
    """
    Extract Ground Truth kinematic parameters from the SAPIEN scene directory.
    Returns: gt_axes, gt_positions, joint_types, gt_states_array
    """
    scene_path = Path(scene_dir)
    
    # 1. Extract Static Parameters: Axis, Position, and Range
    joint_info_file = scene_path / 'joint_infos.json'
    if not joint_info_file.exists():
        raise FileNotFoundError(f"File {joint_info_file} not found!")
        
    with open(joint_info_file, 'r') as f:
        joint_infos = json.load(f)
        
    gt_axes = {}
    gt_positions = {}
    gt_ranges = {}
    joint_types = {}
    
    print(f"=== Parsing Static Kinematic Structure ({joint_info_file.name}) ===")
    for idx, joint in enumerate(joint_infos):
        direction = np.array(joint.get('direction', [0, 0, 0]))
        
        # Filter out static parts (where norm of direction is near zero)
        if np.linalg.norm(direction) < 1e-5:
            print(f"[Skipped] Joint {idx}: Static part")
            continue
            
        j_type = joint.get('joint_type')
        position = np.array(joint.get('origin')) 
        dist_max = joint.get('dist_max')
        
        gt_axes[idx] = direction
        gt_positions[idx] = position
        gt_ranges[idx] = dist_max
        joint_types[idx] = j_type
        
        print(f"[Extracted] Joint {idx}: Type '{j_type}'")
        print(f"            -> Axis: {direction}")
        print(f"            -> Position: {position}")
        print(f"            -> Range (dist_max): {dist_max}")

    # 2. Extract Dynamic Parameters: States
    transforms_file = scene_path / 'transforms.json'
    if not transforms_file.exists():
        raise FileNotFoundError(f"File {transforms_file} not found!")
        
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
        
    print(f"\n=== Parsing Dynamic Motion States ===")
    frames = transforms.get('frames', [])
    num_frames = len(frames)
    
    # Extract normalized states [0.0 ~ 1.0]
    normalized_states = np.array([frame.get('state', 0.0) for frame in frames])
    
    # Convert to physical units (radians or meters)
    real_states = {}
    for j_id in gt_axes.keys():
        actual_movement = normalized_states * gt_ranges[j_id]
        real_states[j_id] = actual_movement
        print(f"State array generated for Joint {j_id}! Shape: {actual_movement.shape}")
        # print(actual_movement)  
        
    return gt_axes, gt_positions, joint_types, real_states

if __name__ == "__main__":
    # Point to your specific scene directory
    SCENE_DIR = '../data/videoartgs/sapien/168/' 
    
    try:
        axes, positions, types, states_dict = extract_ground_truth(SCENE_DIR)
        print("\nSuccess! You can now use 'states_dict[j_id]' to calculate errors.")
    except Exception as e:
        print(f"Error occurred: {e}")