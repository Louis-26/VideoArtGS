# overall pipeline for VideoArtGS architecture with PAT integration
Take the scene `168`(faucet) as an example
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
It stays consistent with the original pipeline, where the original pipeline is [here](../README.md)
