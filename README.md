# VideoArtGS with PAT
In this branch, we will integrate Part Articulation Transformer from [PARTICULATE](https://arxiv.org/pdf/2512.11798)

## Architecture
Input multi-frames --> Canonical gaussians --> Part Articulation Transformer --> Deformation field --> Output multi-frames

## Advantage over original VideoArtGS
- don't rely on heavy postprocessing procedures, including `depth and pose estimation` from VGGT, `3D tracks` from TAPIP3D

## Advantage over Particulate
- utilize multiview frames instead of input mesh, causing more efficient preprocessing steps

# VideoArtGS pipeline
More detailed description can be found in [VideoArtGS pipeline](./overview/V_methodology.md), with flow chart available [here](https://www.figma.com/design/7dDTR57ZKdyMfOiuJXp0s8/VideoArtGS-PAT-procedure-graph?node-id=13-172&t=a1yyM5Rf8iqJX3WM-1)
## step 1
use `bash scripts/init_cano.sh 1` 

Given multiview frames, transform into canonical gaussians in the form of point cloud

Input
- /DATASET/images/, multiview frames(250 images from different perspectives)
- /DATASET/depth/, depth maps for each frame
- /DATASET/transforms.json, give the camera pose and each frame's intrinsic parameters
- /DATASET/point_cloud.ply, ground truth point cloud

Output
- 3D gaussians after 20000 iterations with number N(N=42458 for scene `168`), for each gaussian, we have
    - position $\mu \in \mathbb{R}^3$
    - rotation $\q \in \mathbb{R}^4$
    - scale $\s \in \mathbb{R}^3$
    - opacity $\alpha \in [0,1]$
    - SH feature $\f \in \mathbb{R}^{48}$ 
    - part segmentation feature $\f \in \mathbb{R}^{16}$
In total, we get $N \times 75$ parameters for each scene as gaussian primitives attributes

visualization:
[point_cloud_init](./assets/images/init_cano_pc.png)


## Step 2

Command
`bash scripts/init_deform.sh 1`

Objective
This stage trains a coordinate-based Multi-Layer Perceptron (MLP) to learn the kinematic priors and the continuous deformation field of the dynamic scene. Instead of updating the canonical Gaussian attributes, it establishes a mapping network that outputs the spatial variations—specifically, the translation offset ($\delta \mu \in \mathbb{R}^3$) and rotation offset ($\delta r \in \mathbb{R}^4$)—for each Gaussian primitive given a specific timestamp $t$.

Key Modules
*   **Segmentation Module (`HybridSeg`)**: Computes the part-belonging probabilities (Part Masks) for each Gaussian primitive. It implicitly learns to group points into rigid kinematic parts without explicit 3D annotations.
*   **Articulation Module (`ArticulationModel`)**: Models the mechanical skeleton constraints, outputting the articulation parameters to drive the grouped primitives.

Inputs
*   `DATASET/joint_infos.json`: the json file containing the joint type, axis and pivot for each part
*   `DATASET/filtered.npz`: Sparse 3D motion trajectories acting as physical tracking supervision. 
    - coords: dimension (100, 7700, 3), `100` frames, `7700` tracked points, each with 3D coordinates.
    - visibs: dimension (100, 7700), `100` frames, `7700` tracked points, value $M_{xy}$ as True/False indicating whether the point `y` is visible in the frame `x`.

Outputs
*   `deform.pth`: The optimized neural network weights serving as a highly compressed physical engine. including
    - segmentation model
        - center, dimension (2,3), centers of each part
        - logscale, dimension (2,3), log scale of each part
        - rot, dimension (2,4), rotation of each part
        - grid, parameter dimension 10035200, map from 3D coordinates to high dimension features
        - mlp, map from dimension to raw probabilities of each part
        - motion_grid, dimension 10035200, map from 3D coordinates to motion features
        - motion_mlp, map from motion features to motion extent
    - articulation model
        - origins, dimension (3,3), origins of each part
        - directions, dimension (3,3), directions of each part
        - qr_s, dimension (4), quaternion real part
        - qd_s, dimension (4), quaternion dual part
        - time_model, total parameter 16899, map from time to state(rotation degree/prismatic distance)

---

## Step 3

Command
`bash scripts/train.sh 1`

Jointly optimize deformation field and canonical gaussians from multiview frames and tracking trajectories.


Inputs 
- `OUTPUT/point_cloud.ply`, point cloud from step 1
- `OUTPUT/deform.pth`, deformation weights from step 2
- `DATASET/filtered.npz`, sparse 3D motion trajectories acting as physical tracking supervision. 
    - coords: dimension (100, 7700, 3), `100` frames, `7700` tracked points, each with 3D coordinates.
    - visibs: dimension (100, 7700), `100` frames, `7700` tracked points, value $M_{xy}$ as True/False indicating whether the point `y` is visible in the frame `x`.

Outputs
- updated `point_cloud.ply`, refined canonical gaussians after joint optimization
- updated `deform.pth`, refined deformation weights after joint optimization



output point cloud
[point cloud after training](./assets/images/pc_after_train.png)
---

## Step 4

Command
`bash scripts/render.sh 1`

Input
- `point_cloud.ply`: trained canonical gaussians
- `deform.pth`: trained deform field


Output
- ground truth multiview images
- depth maps
- `joint_info.json`: The final optimized 3D physical topology (optimized axes, origins, and segmentation centers).
- `joint_value.npy`: The predicted temporal dynamics matrix of shape `[K_joints, N_frames]`, containing the rotation angles $\theta$ for each joint across the video sequence.


## step 5
Quantitative evaluation of the modeled articulated object, assessing both the geometric fidelity of the reconstructed 3D shape and the precision of the kinematic parameter estimation.

Input:
- ground truth axis direction, position and point cloud
- predicted axis direction, position and point cloud

Output: results.csv including
- axis error
- position error
- chamfer distance for whole point cloud
- chamfer distance for moving part point cloud
- chamfer distance for static part point cloud

## step 6(optional)
Compute gif, mp4 and mesh for the articulated scene 


# VideoArtGS+PAT pipeline
More detailed description can be found in [VideoArtGS+PAT pipeline](./overview/V_PAT_methodology.md)
## step 1
This step is consistent with the original VideoArtGS pipeline, where we initialize the canonical Gaussian representation of the scene.
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/init_cano.sh 1 
```
After that, we get the canonical gaussians, including position, rotation, scale, opacity, SH feature and part segmentation feature. The output is stored in `point_cloud.ply`.
Specifically, the dimension for each gaussian primitive is 75, including
- position $\mu \in \mathbb{R}^3$
- rotation $\q \in \mathbb{R}^4$
- scale $\s \in \mathbb{R}^3$
- opacity $\alpha \in \mathbb{R}$
- part segmentation feature $\f \in \mathbb{R}^{16}$
- SH feature $\f \in \mathbb{R}^{48}$

At this time, position $\mu$, rotation $\q$ and part segmentation feature $\f$ (combined with dimension 23) will be used as the input for the PAT model.


## Step 2: Part Articulation Transformer (PAT) Inference
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/init_deform_PAT.sh 1 
```
Objective: Infer kinematic structure directly from the 3D point cloud, replacing motion tracking and joint infos priors.

Input
- point cloud from step 1, including position, rotation and part segmentation feature
    - position dimension 3
    - rotation dimension 4
    - part segmentation feature dimension 16

Output
- deform.pth, with exactly the same structure as the original VideoArtGS pipeline, including segmentation model and articulation model.




## Step 3-6
It stays consistent with the original pipeline.




# PAT Architecture 
Input:
- point cloud 3D coordinates, dimension (N,3)
<!-- - segmentation feature, dimension (N,16) -->

Output, articulation parameters including
- part_ids
- motion_hierarchy
- is_part_revolute
- is_part_prismatic
- revolute_plucker
- revolute_range
- prismatic_axis
- prismatic_range
- closest_point_on_axis


# Loss Analysis
- canonical-to-observation loss: $L_{c2o}$
- render loss



# Evaluation Analysis
In dataset `VideoArtGS`, 
we already have ground truth,
- axis direction $A_{gt} \in \mathbb{R}^3$
- axis position $P_{gt} \in \mathbb{R}^3$
- point cloud, stored in .ply 
    - $P_{whole_gt} \in \mathbb{R}^{N \times 3}$
    - $P_{part_x_gt} \in \mathbb{R}^{M_x \times 3}$
x=1,2,...,k, where k is the number of parts in the scene
we want to compute
- axis direction $A_{pred} \in \mathbb{R}^3$
- axis position $P_{pred} \in \mathbb{R}^3$
- point cloud, stored in .ply 
    - $P_{whole_pred} \in \mathbb{R}^{N \times 3}$
    - $P_{part_x_pred} \in \mathbb{R}^{M_x \times 3}$
x=1,2,...,k, where k is the number of parts in the scene

And then evaluate by 
- axis error $E_A = \arccos(\frac{A_{gt} \cdot A_{pred}}{||A_{gt}|| ||A_{pred}||})$
- position error $E_P = ||P_{gt} - P_{pred}||_2$
- chamfer distance $CD(P_{whole_gt}, P_{whole_pred})$ and
- chamfer distance $CD(P_{part_x_gt}, P_{part_x_pred})$

# Complementary notes
time cost for each step:
- step 1: initialize canonical gaussians, 4 minutes per scene, 20000 iterations, A100-40GB-PCle
- step 2: 12 seconds per scene for PAT integration, A100-40GB-PCle
- step 3: train, 15 minutes per scene, 20000 iterations, A100-40GB-PCle
- step 4: render, 3 minutes per scene, 250 frames, A100-40GB-PCle
