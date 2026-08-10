# VideoArtGS + Part Articulation Transformer (PAT)
The main distinction between VideoArtGS and VideoArtGS+PAT is that in the new version, we no longer rely on tedious preprocessing steps including depth maps 3D tracking trajectories extraction.

## Data

Model Input(feedable for the model)
- ground truth multiview monocular video frames 
- ground truth camera intrinsics/poses frame-by-frame

Reference Ground Truth(not feedable for the model)
- ground truth mesh point cloud for both the whole object and each part
- ground truth segmentation parameters(part number/centers)
- ground truth articulation parameters(including direction axis/origin/joint type)
- ground truth motion parameters(time-variant joint states)

## Preliminary step
- foreground mask extraction for each frame
- Utilize COLMAP to extract initial point cloud for the initial 3DGS data backbone with **first M static** video frames


## step 1: canonical gaussians initialization
Overview:
Train 3D canonical gaussian primitives from M static multiview frames as gaussian primitives initialization.

Corresponding scripts
- [init_cano.py](../init_cano.py)
- [init_cano.sh](../scripts/init_cano.sh)
- [SLURM init_cano.sh](../SLURM_execution/SLURM_script/init_cano.sh)

Input: 
- **First M static** multiview monocular video frames ${I_t}_{t=1}^M$
- ground truth camera poses $P$, (extrinsic matrix $ E = P^{-1} \in \mathbb{R}^{4 \times 4}$) and camera intrinsic matrix $K \in \mathbb{R}^{3 \times 3}$ for each selected frame
- initial point cloud derived from COLMAP with **first M static** video frames



Model/Parameters:
- 3D Gaussian Primitives to be trained, with updated parameters including
    - $\mu \in \mathbb{R}^{3}$
    - rotation $\r \in \mathbb{R}^{4}$
    - scale $s \in \mathbb{R}^{3}$
    - opacity $\sigma \in \mathbb{R}$
    - SH feature $h \in \mathbb{R}^{48}$

Training:
Optimize 3D Gaussian Primitives to reconstruct the static canonical object representation, with RGB-only loss, $\mathcal{L}_{cano}=(1-\lambda_{SSIM})\mathcal{L}_1+\lambda_{SSIM}(1-SSIM)$

Output:
- Trained 3D **Canonical** Gaussian Primitives, $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$
- optimized canonical Gaussian model checkpoints(.ply) saved in different iterations(5000, 10000, 15000, 20000)


## step 2: Part Articulation Transformer (PAT) initialization and finetuning
Overview: Initialize the Part Articulation Transformer (PAT) with the customized dataset, then finetune it to enhance the inference performance.

Corresponding scripts
- [PAT_finetune.py](../PAT/PAT_finetune.py)


Input:
- 20 scenes in `VideoArtGS-20`, utilize 15 of them for training
    - training scenes: `100481  101284  101287  101808  101908  103015  103811  10489  10655  1280  168  25493  30666  31249  45194`
    - rest 5 of them for test: `45503  45612  47648  8961  9016`

Model:
Part Articulate Transformer, with N=8 blocks

Finetune:
LoRA update with the 15 training scenes

Output:
- finetuned Part Articulation Transformer (PAT) model 


## step 3: Part Articulation Transformer (PAT) Inference
Overview:
Derive articulation parameters initialization from Part Articulation Transformer(PAT), with the input of Gaussian mean vector and other optional input(**TODO**)



Input:
- Gaussian Primitives core parameters(potentially other inputs)
    - position $\mu \in \mathbb{R}^{3}$

Model:
Finetuned Part Articulation Transformer(PAT) model from [PARTICULATE](https://ruiningli.com/particulate) 


Output:
Initialized segmentation parameters prior for $S_\Phi$, and articulation parameters $A_\Psi$ including the following for each part
- per-Gaussian part labels as segmentation prior $m_i$ for each gaussian primitive index $i \in 1, \dots, N$
- articulation parameters prior
    - articulation axis direction($\mathbb{R}^3$) 
    - articulation axis origin($\mathbb{R}^3$) 
    - joint type for each movable part(prismatic/revolute)
- segmentation parameters prior
    - part radial extent(dist_max) for each part
    - number of parts $K$
    - part center $c_k$ with $k \in 1, ..., K$
- motion parameters prior
    - joint motion range($\mathbb{R}^2$)  




## step 4: hybrid center-grid part assignment in deformation field
Overview:
Initialize the part-assignment module $S_\Phi$ that conducts robust segmentation for both movable and static parts, mapping canonical-space 3D positions to part probabilities.

Corresponding scripts:
[videoartgs.py](../scene/videoartgs.py)


Input:
- segmentation prior $m_i$ for each gaussian primitive index
- part center prior $c_k$ for each part $k \in \{1, \dots, K\}$ 
- number of semantic parts $K$ 
- Initialized part center $p_k$ and part radial extent(dist_max) $s_k$ for each part $k \in \{2, \dots, K\}$
- canonical Gaussian primitives $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$ mean vectors from step 1


Model/Parameters:
- Part Segmentation Module $S_\Phi$
    - hash grid $H_{res}$ and hash motion grid $H_{static}$(each of them as 12 layer multiresolution hash-grid encoders)
    - MLP 1 for $H_{res}$ mapping from center-relative features, the query position, and the `grid` feature to logscale+shift, MLP 2 for $H_{static}$ mapping from the query position and the `motion_grid` feature to static logit
    - center location $p_k \in \mathbb{R}^{3}$, rotation matrix $V_k \in \mathbb{R}^{3 \times 3}$(characterized by quaternion $q_k \in \mathbb{R}^4$), and scale vector $\lambda_k \in \mathbb{R}^{3}$(derived from dist_max) for each part $k \in \{2, \dots, K\}$ (suppose the first part is the static slot)

Output:
- assignment probability vector $m_i \in \mathbb{R}^{K}$ for each query point


## step 5: deformation field initialization
Overview:
Initialize and train the deformation field from motion prior in step 3

Corresponding scripts
- [init_deform_PAT.py](../PAT/init_deform_PAT.py)
- [init_deform_PAT.sh](../scripts/init_deform_PAT.sh)
- [SLURM init_deform_PAT.sh](../SLURM_execution/SLURM_script/init_deform_PAT.sh)


Input:
- Initialized segmentation parameters $S_\Phi$ from step 4
- Initialized articulation parameters $A_\Psi$(axis direction/origin, joint type) from step 3

Model:
- Deformation field $\mathcal{F}$ including `Segmentation Module $S_\Phi$` from step 3 and `Articulation Module $A_\Psi$` from step 2 (with axis direction/origin for each part)
- Trained with the multiview frame loss

Output:
- Initialized and trained Deformation field $\mathcal{F}$


## step 6: canonical gaussian and deformation field joint training
Overview:
Jointly optimize the canonical Gaussian primitives and deformation field across all video frames 

Corresponding scripts
- [train.py](../train.py)
- [trainer.py](../trainer.py)
- [train.sh](../scripts/train.sh)
- [SLURM train.sh](../SLURM_execution/SLURM_script/train.sh)

Input:
- Video frames ${I_t}_{t=1}^T$
- camera intrinsics and extrinsics/poses frame-by-frame
- Initialized canonical Gaussian primitives $\mathcal{G}^c=\{G_i^c \}_{i=1}^N$ from step 1
- Initialized Deformation field $\mathcal{F}$ from step 4

Model/Parameters:
- Gaussian Primitives Parameters
- Deformation field $\mathcal{F}$ including Segmentation Module $S_\Phi$ and Articulation parameters $\Psi$

Training:
- optimize based on the render loss($\mathcal{L}_{RGB_render}=((1-\lambda_{SSIM})\mathcal{L}_1+\lambda_{SSIM}\mathcal{L}_{SSIM}+\mathcal{L}_{seg})$) 


Output:
- Updated Gaussian Primitives
- Updated Deformation field(articulation parameters $A_{\Psi}$ and segmentation module)

## step 7: 3D deformed gaussians rendering
Overview:
Exactly the same procedure as the original VideoArtGS pipeline.
After finishing training the canonical Gaussian primitives and deformation field, render the deformed Gaussian primitives to get qualitative results, including rendered RGB/depth images, videos/gif and mesh visualization.


Corresponding scripts
- [render.py](../render.py)
- [render.sh](../scripts/render.sh)
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
- Rendered RGB images for each frame
- generated mp4/gif for the articulated scene
- colored mesh visualization files .ply(`meshes/`: per-part + whole, via TSDF) 

## step 8: evaluation
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
