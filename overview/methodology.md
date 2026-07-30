# VideoArtGS pipeline
Data:
- Multiview monocular video frames 
- depth images/camera poses from VGGT
- 3D tracking trajectories from TAPIP3D
- ground truth mesh point cloud for both the whole object and each part
- ground truth articulation parameters(including direction axis/origin, part number/centers/joint type)

## step 1: canonical gaussians initialization
Overview:
Train complete 3D canonical gaussian primitives from N static multiview frames as gaussian primitives initialization.

Corresponding scripts
- [init_cano.py](../init_cano.py)


Input: 
- **First N static** multiview monocular video frames
- depth images/camera poses from VGGT
- frame-by-frame transformation matrix


Model:
- Pretrained and constant VGGT/TAPIP3D 
- Trained 3D Canonical Gaussian Primitives
- Trained Deformation Field

Output:
- predicted rendered images/depth maps 
- segmented mesh and point cloud
- motion type for each part
- articulation parameters(predicted center/axis direction)

## step 2: deformation field initialization
Overview:
Conduct **motion pattern analysis** to distinguish trajectory classes, within `static`/`prismatic`/`revolue`/`noise`. After filtering out `prismatic` or `revolue` trajectories, utilize *PCA* for prismatic direction estimation and *SVD* for revolute axis+origin estimation. 

Derive motion prior from pretrained tracking model TAPIP3D to estimate 


Initialize the deformation field from motion prior


## step 3: canonical gaussian and deformation field joint training


## step 4: 3D deformed gaussians rendering





## step 5: evaluation






# VideoArtGS + Part Articulation Transformer (PAT)
Data:
- Multiview frames from a video
- ground truth mesh point cloud for both the whole object and each part
- ground truth articulation parameters(including direction axis, center location, joint type)


Input: 
- Multiview frames from a video
- frame-by-frame transformation matrix
- number of movable parts, joint type of each part
- camera setting parameters

Model:
- Trained Part Articulation Transformer (PAT)
- Trained 3D Canonical Gaussians
- Trained Deformation Field

Output:
- predicted rendered images/depth maps 
- segmented mesh and point cloud
- motion type for each part
- articulation parameters(predicted center/axis direction)