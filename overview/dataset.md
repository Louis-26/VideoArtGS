# Dataset
📦 originally contained by the dataset 
⚙️ added by preprocessing steps
## VideoArtGS-20 

Description: 20 videos of complex articulated objectgs of 10 categories from PartNet-Mobility dataset, with 2-9 movable parts for each object.

Structure:

📁 data/
├── 📁 videoartgs/sapien
│   ├── 📁 [SCENE_NUMBER]/
│   │    ├── 📁 depth/ 📦(sapien)/⚙️(realscan) # all depth images from VGGT processing
│   │    ├── 📁 gt/    📦
│   │    │    ├── 🧾 mobility_v2.json # joint type, axis direction/origin for each part
│   │    │    ├── 🧊 whole_mesh.ply   # whole mesh of the object
│   │    │    ├── 🧊 part_x.ply       # point cloud for each part
│   │    │    └── 🧾 part_info.json   # detailed moving information for each part frame-by-frame
│   │    ├── 📁 images/  📦          # multiview frames
|   |    ├── 🧾 camera.json 📦        # camera extrinsics/intrinsic parameters ($R^{4\times 4}$ matrix) for each frame
|   |    ├── 🧮 filtered.npz ⚙️       # 3D coordinates and visibility for 100 frames(7700 samples for each frame) 
|   |    ├── 🧾 joint_infos_vlm.json ⚙️ # coarse information: number of parts, and joint type for each part 
|   |    ├── 🧾 joint_infos.json ⚙️    # detailed information: part center, joint type, axis direction/origin 
|   |    ├── 🧊 point_cloud.ply  ⚙️    # initial point cloud for the whole object
|   |    └── 🧾 transforms.json ⚙️     # state, time and transform matrix for each frame

## Video2Articulation-S
Description: 73 test videos across 11 categories of synthetic objects from the PartNet-Mobility dataset, and each object has a single movable part

Structure:

📁 data/
├── 📁 v2a/sapien
│   ├── 📁 [SCENE_NUMBER]/
│   │    ├── 📁 depth/ 📦(sapien)/⚙️(realscan) # all depth images from VGGT processing
│   │    ├── 📁 gt/    📦
│   │    │    ├── 🧾 actor_pose.pkl  
│   │    │    ├── 🧾 gt_joint_value.npy 
│   │    │    ├── 🧾 joint_gt_values.txt
│   │    │    ├── 🧾 mobility_v2.json # joint type, axis direction/origin for each part
│   │    │    ├── 🧊 whole_mesh.ply   # whole mesh of the object
│   │    │    ├── 🧊 part_x.ply       # point cloud for each part
│   │    │    └── 🧾 qpos.npy   
│   │    ├── 📁 images/  📦          # multiview frames
|   |    ├── 🧮 filtered.npz ⚙️       # 3D coordinates and visibility for 100 frames(7700 samples for each frame) 
|   |    ├── 🧾 joint_infos.json ⚙️    # detailed information: part center, joint type, axis direction/origin 
|   |    ├── 🧊 point_cloud.ply  ⚙️    # initial point cloud for the whole object
|   |    └── 🧾 transforms.json ⚙️     # state, time and transform matrix for each frame


### reference
- https://huggingface.co/datasets/3dlg-hcvc/video2articulation
- https://huggingface.co/datasets/YuLiu/VideoArtGS-Data/tree/main
- https://github.com/haosulab/SAPIEN