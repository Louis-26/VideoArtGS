# Dataset
## Data structure
📁 dataset/
├── 📁 [SUBSET_NAME]/
│   ├── 📁 [SCENE_NUMBER]/
│   │    ├── 📁 depth/ # all depth images from VGGT processing
│   │    ├── 📁 gt/    
│   │    │    ├── mobility_v2.json # joint type, axis direction/origin for each part
│   │    │    ├── whole_mesh.ply   # whole mesh of the object
│   │    │    ├── part_x.ply       # vary with number of parts
│   │    │    └── part_info.json   # detailed moving information for each part frame-by-frame
│   │    ├── 📁 images/            # multiview frames
|   |    ├──  camera.json  # camera extrinsics parameters ($R^{4\times 4}$ matrix) for each frame
|   |    ├──  filtered.npz # 3D coordinates and visibility for 100 frames(7700 samples for each frame) 
|   |    ├──  joint_infos_vlm.json # coarse information: number of parts, and joint type for each part 
|   |    ├──  joint_infos.json     # detailed information: part center, joint type, axis direction and origin for each part
|   |    ├──  point_cloud.ply      # point cloud for the whole object
|   |    └──  transforms.json      # state, time and transform matrix for each frame

## Video2Articulation-S
description: 73 test videos across 11 categories of synthetic objects from the PartNet-Mobility dataset, and each object has a single movable part

### reference
https://huggingface.co/datasets/3dlg-hcvc/video2articulation

## VideoArtGS-20

description: 20 videos of complex articulated objectgs of 10 categories from PartNet-Mobility dataset, with 2-9 movable parts for each object.


### reference
https://huggingface.co/datasets/YuLiu/VideoArtGS-Data/tree/main


# reference dataset
a collection of 2K articulated objects with motion annotations and rendernig material

## PartNet-Mobility
description: 
### reference
https://github.com/haosulab/SAPIEN