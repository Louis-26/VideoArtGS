# Foundational Models

## VGGT

- Input
  - monocular multiview RGB video frames $\{I_t\}_{t=1}^T \in \mathbb{R}^{T \times H \times W \times 3}$

- Output
  - depth maps $\{D_t\}_{t=1}^T \in \mathbb{R}^{T \times H \times W}$
  - depth confidence maps $\{C_t\}_{t=1}^T \in \mathbb{R}^{T \times H \times W}$
  - camera intrinsics $\{K_t\}_{t=1}^T \in \mathbb{R}^{T \times 3 \times 3}$
  - camera extrinsics $\{E_t\}_{t=1}^T \in \mathbb{R}^{T \times 4 \times 4}$
  



## TAPIP3D
Derive tracking trajectories given the dynamic frames from the latter part of the monocular video.
- Input:
  - monocular multiview dynamic RGB video frames $\{I_t\}_{t=M+1}^T \in \mathbb{R}^{(T-M) \times H \times W \times 3}$
  - depth maps $\{D_t\}_{t=M+1}^T \in \mathbb{R}^{(T-M) \times H \times W}$
  - camera intrinsics $\{K_t\}_{t=M+1}^T \in \mathbb{R}^{(T-M) \times 3 \times 3}$
  - camera extrinsics $\{E_t\}_{t=M+1}^T \in \mathbb{R}^{(T-M) \times 4 \times 4}$

- Hyperparameter
  - tracking point total number $P$, default as 8192
  - number of key frames $n_q$

- Output
  - 3D tracking trajectories $\{X_t\}_{t=M+1}^T$ with each $X_t \in \mathbb{R}^{P \times 3}$ representing the 3D positions of the tracking points in frame $t$
  - visibility masks $\{V_t\}_{t=M+1}^T$ with each $V_t \in \{0,1\}^P$ representing the visibility of each tracking point in frame $t$


## SAM 2
It is used in `realscan` dataset when image mask is not available.
- Input: 
  - RGB video frames $\{I_t\}_{t=1}^T$ with each $I_t \in \mathbb{R}^{H \times W \times 3}$
  - certain number of 2D prompts either as **bounding boxes** or **points** on the first frame

- Output: 
  - Binary Instance masks per frame, as $\{M_t\}_{t=1}^T$ where each $M_t \in \{0,1\}^{H \times W}$ is a binary mask for the object of interest in frame $t$.

## VLM(GPT-4o)
- Input: 
  - every 10 frames of RGB dynamic frames(after first $M$ frames) 
  - system text prompts in query for number of joints and joint type

- Output: 
  - Inference of joint number $J$, with part number $K=J+1$ 
  - joint type $\tau_k \in \{\mathcal{R}, \mathcal{P}\}$ for each movable part $k \in \{2, \cdots, K\}$, as either `revolute` or `prismatic`

## PartField
- Pretrained Checkpoint: `particulate/model_ckpt/model_objaverse.ckpt`

- Input
  - 3D sampled point coordinates $P \in \mathbb{R}^{N_p \times 3}$

- Output:
  - Per-point geometric part feature tokens: $\mathbf{F}_{pf} \in \mathbb{R}^{N_p \times 448}$


# Deformation Field 
Prior Injection:
- part center $c_k$, with $k=1, \ldots, K$, fed into segmentation model
- part scale $s_k$, with $k=1, \ldots, K$, fed into segmentation model
- articulation type $tau_k$, with $k=2, \ldots, K$, fed into articulation model
- articulation axis $d_k$, with $k=2, \ldots, K$, fed into articulation model
- articulation origin $o_k$, with $k=2, \ldots, K$, fed into articulation model




## Progressive Band Hash Grid

Multi-resolution hash grid based on tinycudann

- Input
  - 3D point coordinates $P \in \mathbb{R}^{N \times 3}$


- Hyperparameters
  - number of resolution levels $n_1$, default as 12
  - number of features per level $n_2$, default as 2
  - base resolution, resolution at the coarsest resolution level, default as 16
  - per_level_scale, ratio of resolution level to the previous level, default as 2
  - hash table size, default as 2^19
  - progressive update steps, default as 500


- Progressive Frequency Annealing
  - Active level update formula:
    $$\text{current\_level} = \min\left(\text{start\_level} + \left\lfloor \frac{\max(\text{step} - \text{start\_step}, 0)}{\text{update\_steps}} \right\rfloor, \ L\right)$$
  - Channel mask $\mathbf{m}_{band} \in \{0, 1\}^{24}$ with gradient detachment for unactivated high-frequency bands:
    $$\mathbf{F}_{grid} = \mathbf{e} \odot \mathbf{m}_{band} + \text{detach}(\mathbf{e}) \odot (1 - \mathbf{m}_{band})$$
    where $\mathbf{e} \in \mathbb{R}^{N \times 24}$ is the raw trilinearly interpolated feature from the hash grid.

- Output
  - $F_{grid} = [\mathbf{f}^{(0)}, \mathbf{f}^{(1)}, \ldots, \mathbf{f}^{(n_1-1)}] \in \mathbb{R}^{N \times 24}$, with $\mathbf{f}^{(i)} \in \mathbb{R}^{N \times 2}$ 





## Hybrid Part-Assignment MLPs
### static base MLP

- Input
The input is concantenated by $X_c$ and $F_{grid}$:
  - Gaussian primitive coordinates $X_c \in \mathbb{R}^{N \times 3}$
  - multi-resolution spatial features $F_{motion} \in \mathbb{R}^{N \times 24}$ from the hash grid


- Structure
  - 3 layer FC MLP, input: 27-d, hidden_1: 32-d, RELU, hidden_2: 32-d, RELU, output: 1d

- Hyperparameters
  - Annealing temperature $\tau$, default as 0.1

- Output
  - static logits $z_0 \in \mathbb{R}^{N}$

### dynamic part MLP
- Input 
  - centers $c_k \in \mathbb{R}^3$
  - scales $s_k \in \mathbb{R}^3$
  - quaternions $r_k \in \mathbb{R}^4$

- Computation
The input is concantenated by $X_c, F_{grid}, d_k, \lVert d_k \rVert$
  - Gaussian primitive coordinates $X_c \in \mathbb{R}^{N \times 3}$
  - multi-resolution spatial features $F_{grid} \in \mathbb{R}^{N \times 24}$ from the hash grid
  - relative coordinate to the part center $d_k=\frac{R(r_k)(X_c-c_k)}{S_k} \in \mathbb{R}^{N \times 3}$ for each part $k=1,...,K-1$
  - distance to the part center $\lVert d_k \rVert\in \mathbb{R}^{N}$ for each part $k=1,...,K-1$

- Structure
dim=4(K-1)+27
  - 2 layer FC MLP, input: dim, hidden_1: 32-d, RELU, output: 2(K-1)-d


- Output  
  - scaling residual $\delta_s \in \mathbb{R}^{N \times (K-1)}$
  - position shift residual $\delta_b \in \mathbb{R}^{N \times (K-1)}$

Then compute $z_k=-d_k^2 exp(\frac{1}{2}\delta_s)+\frac{1}{2}\delta_b$, with k=1,...,K-1
Then $z_i=[z_{i0}, z_{i1}, \ldots, z_{i,K-1}]$, and $m_i=Gumbel-Softmax(z_i, \tau)$ with annealing temperature $\tau$ to get the final part assignment probability $m \in \mathbb{R}^{N \times K}$

## Time Model
- Input
  - normalized time $t \in [0, 1]$ for each frame
The input is further transformed into a 21-d vector using Fourier feature mapping, as $embed(t)=[t, \sin(\pi t), \cos(\pi t), \ldots, \sin(2^9\pi t), \cos(2^9\pi t)]$

- Structure
  - 3 layer MLP, input: 21-d, hidden_1: 64-d, RELU, hidden_2: 64-d, RELU, output: 1-d


- Output
  - Scalar joint motion state vector $\boldsymbol{\theta}(t) = [\theta_1(t), \theta_2(t), \dots, \theta_{K-1}(t)]^T \in \mathbb{R}^{K-1}$:
    - For revolute joints ($\tau_k = \mathcal{R}$): rotation angle in radians.
    - For prismatic joints ($\tau_k = \mathcal{P}$): translation distance in meters.


## Dual Quaternion Linear Blending (DQLB)
- Input
  - 3D Gaussian mean vectors $\mu \in \mathbb{R}^{N \times 3}$
  - part assignment probabilities $m \in \mathbb{R}^{N \times K}$
  - $q_r \in \mathbb{R}^{K \times 4}$
  - $q_d \in \mathbb{R}^{K \times 4}$

- Output
  - variation of 3D Gaussian mean vectors $\delta_\mu \in \mathbb{R}^{N \times 3}$
  - variation of 3D Gaussian rotation vectors $\delta_r \in \mathbb{R}^{N \times 4}$


# ⭐Particulate Articulation Transformer 

- pretraining checkpoint: `https://huggingface.co/rayli/Particulate/resolve/main/model.pt`
- fine-tuning script: `PAT/PAT_finetune.py`
- inference script: `PAT/init_deform_PAT.py`

- Input:
  - **3D points** $P \in \mathbb{R}^{N_p \times 3}$, with $N_p = 4096$ as the sampled points from the original point cloud $\mathcal{P}$.
  - **3D Point Normals**: $\mathbf{N} \in \mathbb{R}^{N_p \times 3}$, representing the surface normals at each sampled point.
  - **Geometric Part Features from PartField**: $f_i \in \mathbb{R}^{448}$ for $i \in \{1, 2, \ldots, N_p\}$.
  - **3D Tracking Features from VideoArtGS Extension**:
    - `track_geo` (56-d)
      - 3D tracking trajectory displacements across 8 timestamps (24-d)
      - visibility (8-d)
      - motion statistics (3-d)
      - PCA main direction & variances (5-d)
      - motion-type one-hot (4-d)
      - fitted axis direction & origin (6-d)
      - prismatic direction (3-d)
      - confidence/validity indicators (3-d)
    <!-- - `3D TAPIP` (384-d): TAPIP3D `EfficientUpdateFormer` hidden state tokens transferred via inverse-distance weighted k-NN ($k=4$). -->
  - **VGGT depth features** (128-d): PCA-reduced VGGT aggregator visual geometric tokens.
  - **Part Slot Query Tokens**: $P_{max} = 16$ learnable part queries 

- Architecture:
  - **Backbone**: `PAT_B` (Particulate-Base) Transformer Encoder-Decoder.
  - **Layers**: 6 sequential bidirectional Transformer blocks (`blocks[0]` to `blocks[5]`).
  - **Tri-Attention Mechanism per Block**:
    - `attn1`: Query Self-Attention (part queries attend to each other).
    - `attn2`: Point-to-Query Cross-Attention (queries attend to point tokens).
    - `attn3`: Query-to-Point Cross-Attention (point tokens attend to queries).
  - **Multi-Head Decoders**:
    - `point_mask_decoder`: Multi-layer MLP predicting per-point part assignment logits $\in \mathbb{R}^{N_p \times M}$.
    - `part_motion_classifier`: MLP predicting motion type classification $\in \mathbb{R}^{M \times 4}$ (0: none/static, 1: revolute, 2: prismatic, 3: both).
    - `revolute_motion_decoder`: MLP predicting revolute Plücker line parameters + motion limits.
    - `prismatic_motion_decoder`: MLP predicting prismatic direction vector + motion limits.
    - `point_motion_decoder`: MLP predicting per-point closest point on the axis $\in \mathbb{R}^{N_p \times 3}$ (`per_point_closest` mode).
    - `part_hierarchy_decoder`: MLP predicting kinematic parent-child adjacency matrix $\in \mathbb{R}^{M \times M}$.

- Output:
  - **Part Segmentation**: Per-point part membership probabilities $\in [0, 1]^{N_p \times M}$ (argmax yields per-point part ID).
  - **Motion Type**: Classification label for each part (`static`, `revolute`, `prismatic`).
  - **Articulation Axis Direction**: $\mathbf{d}_k \in \mathbb{R}^3$ (unit direction vector).
  - **Articulation Axis Origin**: $\mathbf{o}_k \in \mathbb{R}^3$ (derived from Plücker coordinates via `plucker_to_axis_point`, denormalized back to world coordinate frame).
  - **Joint Motion Limits**: Lower and upper motion bounds $[\theta_k^{\min}, \theta_k^{\max}]$.
  - **Kinematic Hierarchy**: Directed acyclic graph (DAG) representing the kinematic parent-child structure.

- Loss:
  - **Bipartite Matching**: Supervised end-to-end with the **Hungarian Matcher** to find the optimal assignment between $M$ predicted slots and ground-truth parts.
  - **Multi-Task Supervised Losses**:
    1. $\mathcal{L}_{\text{mask}}$: Cross-Entropy loss on per-point part assignment.
    2. $\mathcal{L}_{\text{dice}}$: Soft Dice loss for multi-class point mask segmentation.
    3. $\mathcal{L}_{\text{cls}}$: Cross-Entropy loss on motion type classification.
    4. $\mathcal{L}_{\text{axis}}$: L1 loss and Cosine Distance loss on axis direction $\mathbf{d}_k$.
    5. $\mathcal{L}_{\text{origin}}$ / $\mathcal{L}_{\text{plucker}}$: L1 loss on Plücker coordinates.
    6. $\mathcal{L}_{\text{closest}}$: L1 loss on the predicted closest point on axis.
    7. $\mathcal{L}_{\text{hierarchy}}$: Binary Cross-Entropy on parent-child adjacency matrix.

- Optimization: AdamW
  - Optimizer: `AdamW`
  - Learning Rate: $1 \times 10^{-4}$ (with linear warmup and cosine annealing schedule)
  - Weight Decay: $1 \times 10^{-4}$
  - Batch Size: 1 scene per GPU (due to high point cloud and trajectory dimensionality)

- Evaluation:
  - **Axis Direction Error**: $\arccos(\frac{\mathbf{d}_{\text{pred}} \cdot \mathbf{d}_{\text{gt}}}{\|\mathbf{d}_{\text{pred}}\| \|\mathbf{d}_{\text{gt}}\|}) \times \frac{180^\circ}{\pi}$ (degrees)
  - **Axis Origin Error**: 3D skew line-to-line distance $d(l_{\text{pred}}, l_{\text{gt}})$ (centimeters)
  - **Motion Type Accuracy**: Precision, Recall, and Accuracy on revolute vs. prismatic classification
  - **Part Segmentation mIoU**: Mean Intersection over Union between predicted and GT part point sets

## hyperparameters
- `num_layers`: 6 Transformer blocks
- `hidden_size`: 768
- `n_heads`: 12 attention heads per block (`dim_head = 64`)
- `max_parts`: $M = 16$ part slots
- `num_enc_points`: 40,000 for PartField Triplane encoder; 4,096 for PAT decoder tokens
- `dropout`: 0.1
- `motion_representation`: `per_point_closest` (reconstructs Plücker line from per-point closest points for higher stability)
- `training_epochs`: 150 epochs for fine-tuning on VideoArtGS scenes

## trainable weights
- **Frozen Pretrained Backbone**: The core Transformer blocks loaded from `model.pt` are frozen to preserve generalized 3D articulation priors.
- **LoRA Adaptation**: Low-Rank Adaptation (LoRA, rank $r=8$ or $16$) injected into all linear projections of `attn1` (query self-attention), `attn2` (point-query cross-attention), and `attn3` (query-point cross-attention).
- **Extra Modality Encoders (`extra_embeds`)**: Fully trainable 2-layer MLPs (`nn.LayerNorm` $\to$ `nn.Linear` $\to$ `SiLU` $\to$ `nn.Linear`) projecting `track_geo` (56-d), `track_tapip` (384-d), and `vggt` (128-d) to the 768-d embedding space; the final projection layer is zero-initialized.
- **Output Heads & Decoders**: Task-specific decoders (`point_mask_decoder`, `part_motion_classifier`, `revolute_motion_decoder`, `prismatic_motion_decoder`, `part_hierarchy_decoder`) are fine-tuned to adapt to the VideoArtGS data distribution.





