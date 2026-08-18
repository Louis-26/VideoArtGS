# VideoArtGS pipeline
Data:
- multiview monocular video frames 
- ground truth camera intrinsics/poses frame-by-frame
- depth images/camera poses(optional) from VGGT with video frames
- 3D tracking trajectories from TAPIP3D with depth images and video frames
- ground truth mesh point cloud for both the whole object and each part
- ground truth articulation parameters(including articulation axis/origin/range, part number/centers/joint type/time-variant joint states)

## Preliminary step
Overview:
Preprocess the multiview monocular video frames to obtain depth images and camera poses from VGGT, and 3D tracking trajectories from TAPIP3D.

Input:
- multiview monocular video frames

Model:
- Pretrained VGGT model
- Pretrained TAPIP3D model

Output:
- depth images and camera poses from VGGT *given the multiview monocular video frames*
- 3D tracking trajectories from TAPIP3D *given depth images and multiview monocular video frames*


## step 1: canonical gaussians initialization
Overview:
Train 3D canonical gaussian primitives from M static multiview frames as gaussian primitives initialization.

Corresponding scripts
- [init_cano.py](../init_cano.py)
- [init_cano.sh](../scripts/init_cano.sh)
- [SLURM init_cano.sh](../SLURM_execution/SLURM_script/init_cano.sh)

Input: 
- **First M static** multiview monocular video frames ${I_t}_{t=1}^T$, and corresponding depth images
- ground truth camera poses $P$, (extrinsic matrix $ E = P^{-1} \in \mathbb{R}^{4 \times 4}$) and camera intrinsic matrix $K \in \mathbb{R}^{3 \times 3}$ for each selected frame
- initial fused point cloud for the object from data preprocessing steps, with positions and RGB colors for each point



Model/Parameters:
- 3D Gaussian Primitives to be trained, with updated parameters including
    - position $\mu \in \mathbb{R}^{3}$
    - rotation $\r \in \mathbb{R}^{4}$
    - scale $s \in \mathbb{R}^{3}$
    - opacity $\sigma \in \mathbb{R}$
    - SH feature $h \in \mathbb{R}^{48}$

Training:
Optimize 3D Gaussian Primitives to reconstruct the static canonical object representation.

Output:
- Trained 3D **Canonical** Gaussian Primitives, $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$
- optimized canonical Gaussian model checkpoints(.ply) saved in different iterations(5000, 10000, 15000, 20000)

execution:
- SLURM
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch init_cano.sh 1 1 outputs 1 
```
- independent GPU
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/init_cano.sh 1 1 outputs 1
```


## step 2: motion prior analysis
Overview:
Derive motion prior based on the 3D tracking trajectories from TAPIP3D, obtaining articulation parameters initialization for the deformation field.

Corresponding scripts:

- [motion_analysis.py](../data_tools/motion_analysis.py), invoked by [extract_tapip3d_track.py](../data_tools/extract_tapip3d_track.py)


Input:
- 3D tracking trajectories from TAPIP3D, with 3D coordinates and visibility for each tracked point per frame
- number of semantic parts $K$ and joint type for each of **K-1** movable parts


Non-learned Processing / Algorithms:
- Conduct spatial downsampling for each trajectory  
- PCA method+RANSAC for line fitting(prismatic)
- SVD-based plane fitting followed by circle fitting(revolute)
- K means clustering for prismatic and revolute motion trajectories respectively, with iterative outlier removal

Output:
- motion pattern classification(static, prismatic, revolute, noise) per trajectory
- articulation parameters initialization [joint_infos.json] for each part, including joint type, axis direction/origin, 
- segmentation parameters initialization including part radial extent(dist_max), and **part center**
- processed trajectories filtering noisy and invalid ones(filtered.npz)


## step 3: hybrid center-grid part assignment
Overview:
Define the part-assignment module $S_\Phi$ that conducts robust segmentation for both movable and static parts: a field mapping canonical-space 3D positions to part probabilities.

Corresponding scripts:
[videoartgs.py](../scene/videoartgs.py)


Input:
- number of semantic parts $K$ 
- Initialized part center $p_k$ and dist max $s_k$ for each part $k \in \{2, \dots, K\}$
- canonical Gaussian primitives $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$ mean vectors from step 1


Model/Parameters:
- Part Segmentation Module $S_\Phi$
    - hash grid $H_{res}$ and hash motion grid $H_{static}$(each of them as 12 layer multiresolution hash-grid encoders)
    - MLP 1 for $H_{res}$ mapping from center-relative features, the query position, and the `grid` feature to logscale+shift, MLP 2 for $H_{static}$ mapping from the query position and the `motion_grid` feature to static logit
    - center location $p_k \in \mathbb{R}^{3}$, rotation matrix $V_k \in \mathbb{R}^{3 \times 3}$(characterized by quaternion $q_k \in \mathbb{R}^4$), and scale $s_k \in \mathbb{R}^{3}$ for each part $k \in \{2, \dots, K\}$ (suppose the first part is the static slot)

Output:
- assignment probability vector $m_i \in \mathbb{R}^{K}$ for each query point


## step 4: deformation field initialization
Overview:
Initialize and train the deformation field from motion prior in step 2

Corresponding scripts
- [init_deform.py](../init_deform.py)
- [init_deform.sh](../scripts/init_deform.sh)
- [SLURM init_deform.sh](../SLURM_execution/SLURM_script/init_deform.sh)


Input:
- sampled points from trajectory(filtered.npz)
- Initialized articulation parameters $\Psi$(axis direction/origin, joint type) from step 2

Model:
- Deformation field $\mathcal{F}$ including (Segmentation Module $S_\Phi$ from step 3) and (Articulation Module $A_\Psi$ from step 2, with axis direction/origin and time-variant joint states for each part)
- Trained with the tracking loss including both canonical-to-observed and observed-to-observed loss

Output:
- Initialized and trained Deformation field $\mathcal{F}$

execution:
- SLURM
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch init_deform.sh 1 1 outputs 1 
```
- independent GPU
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/init_deform.sh 1 1 outputs 1
```


## step 5: canonical gaussian and deformation field joint training
Overview:
Jointly optimize the canonical Gaussian primitives and deformation field across all video frames 

Corresponding scripts
- [train.py](../train.py)
- [trainer.py](../trainer.py)
- [train.sh](../scripts/train.sh)
- [SLURM train.sh](../SLURM_execution/SLURM_script/train.sh)

Input:
- Video frames ${I_t}_{t=1}^T$ and corresponding depth images
- camera intrinsics and extrinsics/poses frame-by-frame
- 3D tracking points from (filtered.npz)
- Initialized canonical Gaussian primitives $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$ from step 1
- Initialized Deformation field $\mathcal{F}$ from step 4

Model/Parameters:
- Gaussian Primitives 
- Deformation field $\mathcal{F}$ including Segmentation Module $S_\Phi$ and Articulation parameters $\Psi$

Training:
- optimize based on the render loss($\mathcal{L}_{render}=((1-\lambda_{SSIM})\mathcal{L}_1+\lambda_{SSIM}\mathcal{L}_{D-SSIM}+\mathcal{L}_D)$) and canonical-to-observed tracking loss($\mathcal{L}_{c2o}$) 


Output:
- Updated Gaussian Primitives
- Updated Deformation field(articulation parameters $A_{\Psi}$ and segmentation module)


Execution:
- SLURM
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch train.sh \
    --use_multi 1 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```
- independent GPU
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/train.sh \
    --use_multi 1 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```

## step 6: 3D deformed gaussians rendering
Overview:
After finishing training the canonical Gaussian primitives and deformation field, render the deformed Gaussian primitives to get qualitative results, including rendered RGB/depth images, videos/gif and mesh visualization.


Corresponding scripts
- [render.py](../render.py)
- [render_mask.py](../render_mask.py)
- [render.sh](../scripts/render.sh)
- [render_mask.sh](../scripts/render_mask.sh)
- [SLURM render.sh](../SLURM_execution/SLURM_script/render.sh)
- [gif_video_generate.py](../utils/gif_video_generate.py)
- [visualize_mesh.py](../utils/visualize_mesh.py)


Input:
- Updated Gaussian Primitives $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$
- Deformation Field $\mathcal{F}$
- per-frame camera intrinsics/poses and frame timestamps t

Model/Components:
- Trained Deformation field $\mathcal{F}$ including Segmentation Module $S_\Phi$ and Articulation parameters $\Psi$
- gsplat rasterizer and TSDF mesh extractor

Output:
- predicted joint parameters `joint_info.json` and per-frame joint states `joint_value.npy`
- Rendered RGB images and depth maps for each frame
- generated mp4/gif for the articulated scene
- colored mesh visualization files .ply(`meshes/`: per-part + whole, via TSDF) 

Execution:
- SLURM
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch render.sh \
    --use_multi 0 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
    
sbatch render_mask.sh \
    --use_multi 0 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```
- independent GPU
```bash
# need to disable multi-GPU
cd "$(git rev-parse --show-toplevel)"
bash scripts/render.sh \
    --use_multi 0 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs

bash scripts/render_mask.sh \
    --use_multi 0 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```




## step 7: evaluation
Overview:
Compute quantitative evaluation metrics for articulation estimation and geometry reconstruction.


Corresponding scripts
[eval.py](../eval.py)
[eval.sh](../scripts/eval.sh)
[SLURM eval.sh](../SLURM_execution/SLURM_script/eval.sh)


Input:
- ground truth articulation parameters (axis direction, position), time-variant joint states per frame, point cloud 
- predicted articulation parameters (axis direction, position), time-variant joint states per frame

Output:
- axis error mean/standard deviation by consine similarity in degrees between the predicted axis and ground truth axis
- position error mean/standard deviation by line distance between the predicted axisand ground truth axis 
- joint state error mean/standard deviation (error in cm in prismatic, error in degrees in revolute) between the predicted joint state and ground truth joint state
- chamfer distance between predicted and ground truth meshes for the whole scene (CD-w)
- chamfer distance between predicted and ground truth meshes for the movable parts (CD-m)
- chamfer distance between predicted and ground truth meshes for the static parts (CD-s)

Execution:
- SLURM
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch eval.sh \
    --use_multi 1 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```
- independent GPU
```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/eval.sh \
    --use_multi 1 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir outputs
```