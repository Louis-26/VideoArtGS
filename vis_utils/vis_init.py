#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import seaborn as sns
import json


# In[2]:


import os
import numpy as np
import open3d as o3d

OUT_DIR = "./vis_output"
os.makedirs(OUT_DIR, exist_ok=True)

def visualize_point_cloud(xyz, rgb, save_path=None, width=1024, height=1024,
                          point_size=2.0, bg=(1, 1, 1, 1)):
    """Headless point-cloud visualizer using Open3D OffscreenRenderer.
    Saves a PNG instead of opening a window."""
    if save_path is None:
        idx = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
        save_path = os.path.join(OUT_DIR, f"vis_{idx:03d}.png")

    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.asarray(bg, dtype=np.float32))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    renderer.scene.add_geometry("pcd", pcd, mat)

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = max(bbox.get_extent())
    eye = center + np.array([0, 0, extent * 2.5])
    up = np.array([0, 1, 0])
    renderer.setup_camera(60.0, center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    print(f"saved {save_path}  ({xyz.shape[0]} points)")
    del renderer
    return save_path


# ## Show Initialized Point Cloud and Centers

# In[4]:


def vis_init_cano(data_dir, scene_name):
    with torch.no_grad():
        source_path = f'{data_dir}/{scene_name}'    
        pcd_path = f'{source_path}/point_cloud.ply'
        pcd = o3d.io.read_point_cloud(pcd_path)
        xyz, color = np.asarray(pcd.points), np.asarray(pcd.colors)
        
        joint_infos = json.load(open(f"{source_path}/joint_infos.json", "r"))
        center, scale, origin, direction = [], [], [], []
        K = len(joint_infos)
        for joint_info in joint_infos:
            center.append(joint_info['center'])
            scale.append(joint_info['dist_max'])
            if joint_info['joint_type'] == 'r':
                origin.append(joint_info['origin'])
            else:
                origin.append(joint_info['center'])
            direction.append(joint_info['direction'])
        center = np.array(center).reshape(K, 3)
        scale = np.array(scale).reshape(K, 1)
        origin = np.array(origin).reshape(K, 3)
        direction = np.array(direction).reshape(K, 3)
        num_slots = center.shape[0]
        pallete = np.array(sns.color_palette("hls", num_slots))
        pallete[0] = [0, 0, 0]

        # track_data = np.load(f"{data_dir}/{scene_name}/{scene_name}_filtered_vis.npz")
        track_data = np.load(f"{data_dir}/{scene_name}/filtered_vis.npz")
        # print(track_data)
        xyz = track_data["coords"][0]
        print(xyz.max(0), xyz.min(0))
        mask_ids = track_data["mask_ids"]
        color = pallete[mask_ids]

        # mannually correct the center
        
        # center[3] -= np.array([-0.2, +0.2, 0.])
        # center[4] -= np.array([0.2, 0.2, 0.])
        # center_info[:, :3] = center
        # # center_info[:, 3] /= 4
        # np.save(center_info_path, center_info)
        
        xyz_center = (center[None] + np.random.randn(1000, center.shape[0], 3) * 0.01).reshape(-1, 3)
        rgb_center = pallete[None].repeat(1000, 0).reshape(-1, 3)
        xyz_axis = origin[None] + direction[None] * np.linspace(0, np.ones_like(scale[None]) * 0.3, 100)[:, None]
        xyz_axis = xyz_axis.reshape(-1, 3)
        rgb_axis = pallete[None].repeat(100, 0).reshape(-1, 3)
        xyz_vis = np.concatenate([xyz, xyz_center, xyz_axis])
        rgb_vis = np.concatenate([color, rgb_center, rgb_axis])
        visualize_point_cloud(xyz_vis, rgb_vis)


# In[5]:


# data_dir = "../data/videoartgs/sapien"
# data_dir = "../data/videoartgs/realscan"
# scenes = sorted(os.listdir(data_dir))
# scene_names = [os.path.basename(s) for s in scenes if os.path.isdir(os.path.join(data_dir, s))]
# print("'"+"' '".join(scene_names)+"'")
# scene_names = ['30666_new']
# scene_names = ['box_4r', 'cabinet_2r_4p', 'coffeemachine_2r']
# scene_names = ['cabinet_2r_2p']
# scene_names = ['microwave_1r', 'mac_1r', 'chair_1r', 'coffeemachine_2r']
# for scene_name in scene_names:
#     # print(scene_name)
#     vis_init_cano(data_dir, scene_name)



data_dir = "../data/videoartgs/realscan"
scene_names = ['microwave_1r', 'mac_1r', 'chair_1r', 'coffeemachine_2r']

for s in scene_names:
    print(f"=== {s} ===")
    vis_init_cano(data_dir, scene_name=s)