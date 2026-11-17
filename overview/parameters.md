# Constant/Non-trainable parameters

- $T$: number of frames extracted in a video(total number of frames for the given object)
- $M$: number of canonical static multiview frames
- $N$: number of Gaussian Primitives
- $i$: gaussian primitive index
- $t$: frame index, or time index with different ranges

# Estimated parameters

## segmentation parameters

### part number

number of parts in an articulated object, usually containing $1$ static part and $K-1$ movable parts.

- notation: $K$
- ground truth: provided from `data/sapien/*/gt/mobility_v2.json`
- estimation: directly read from `joint_infos_vlm.json` inferred by VLM
- loss function: no loss function, we usually expect this to be correct, otherwise there is big problem
- evaluation metric: no explicit evaluation metric, but incorrect $K$ can lead to huge `CD loss`

### per-Gaussian part assignment

assignment of each gaussian primitives, as a part id

- notation: $l_i \in \{1, \cdots, K\}$, for $i \in 1, \cdots, N$ where $N$ is the number of gaussian primitives,
- ground truth: doesn' exist
- estimation: derived from the argmax of the assignment probabilities $\mathbf{m}_i = [m_{i0}, m_{i1}, \dots, m_{i,K-1}]^{\top} \in [0, 1]^K$ for $i \in \{1, \dots, N\}$, i.e., $\hat{k}_i = \arg\max_k m_{ik}$
- loss function:
  - $\mathcal{L}_{\text{RGB}} = (1 - \lambda_{\text{SSIM}}) \mathcal{L}_1 + \lambda_{\text{SSIM}} (1 - \text{SSIM})$
  - $\mathcal{L}_{\text{track}} = \frac{1}{|P|} \sum_{p \in P} \left\| \mathcal{F}(x_p^c, \mathbf{m}_p, t) - x_p(t) \right\|_2^2$
  - $\mathcal{L}_{\text{reg}} = 0.1 \times \sum_{k=1}^{K-1} \left\| p_k - \frac{\sum_{i=1}^N m_{ik} \sigma_i \mu_i^c}{\sum_{i=1}^N m_{ik} \sigma_i + \epsilon} \right\|_2^2$
- evaluation metric: CD(Chamfer Distance) loss of static, dynamic and the whole object, as $CD_{static}$, $CD_{dynamic}$, and $CD_{whole}$ respectively.

### part center

part center for each part

- notation: $c_k \in \mathbb{R}^3$, for $k \in 1, \cdots, K$
- ground truth: computed from `part_x.ply` in `data/sapien/*/gt/`
- estimation
  - VideoArtGS: prior derived from the motion analysis, optimized through downstream refinement
  - ArtTrans: prior derived from transformer inference, optimized through downstream refinement
- loss function: $\mathcal{L}_{reg}=0.1 \times \sum_{k=1}^{K-1} \left\| p_k - \frac{\sum_{i=1}^N m_{ik} \sigma_i \mu_i^c}{\sum_{i=1}^N m_{ik} \sigma_i + \epsilon} \right\|_2^2$
- evaluation metric: CD(Chamfer Distance) loss of static, dynamic and the whole object, as $CD_{static}$, $CD_{dynamic}$, and $CD_{whole}$ respectively.

### part scale vector

- notation: $\lambda_k \in \mathbb{R}^3$, for $k \in 2, \cdots, K$
- ground truth: not provided
- estimation: initialize with the largest Euclidean distance between the tracking point and the part center, and then optimize with loss function
- loss function: $\mathcal{L}_{RGB}$ and $\mathcal{L}_{track}$
- evaluation metric: implicitely revealed by CD(Chamfer Distance) loss of static, dynamic and the whole object, as $CD_{static}$, $CD_{dynamic}$, and $CD_{whole}$ respectively.

### part rotation matrix

rotation matrix for each movable part

- notation: $V_k \in \mathbb{R}^{3 \times 3}$, for $k \in 2, \cdots, K$
- ground truth: not provided
- estimation: induced by unit quaternion, initialized with identity matrix, and then optimized with loss function
- loss function: $\mathcal{L}_{RGB}$ and $\mathcal{L}_{track}$
- evaluation metric: implicitly revealed by CD(Chamfer Distance) loss of static, dynamic and the whole object, as $CD_{static}$, $CD_{dynamic}$, and $CD_{whole}$ respectively.

## articulation parameters

### articulation type

- notation: $type$, `prismatic`/`revolute`/`both`/`null`
- ground truth: `data/*/*/gt/mobility_v2.json` 
- estimation
  - VideoArtGS: derived from motion pattern analysis and VLM inference
  - ArtTrans: inferred by the transformer model
- loss function:
  - VideoArtGS: no loss function
  - ArtTrans: cross-entropy loss
- evaluation metric: if the predicted type is not accurate, give the largest penalty

### axis direction

- notation: $d_k \in \mathbb{R}^3$, for $k \in 2, \cdots, K$
- ground truth: `data/*/*/gt/mobility_v2.json`
- estimation:
  - VideoArtGS: derived from motion pattern analysis and VLM inference
  - ArtTrans: inferred by the transformer model
- loss function: $\mathcal{L}_{RGB}$, $\mathcal{L}_{track}$ and $\mathcal{L}_{axis}$
- evaluation metric: arccosine of the dot product between the predicted axis direction and the ground truth axis direction, i.e., $\arccos(\frac{d_k^{\top} d_k^{gt}}{\|d_k\|\|d_k^{gt}\|}) \times \frac{180 ^\circ}{\pi}$

### axis origin
$\hat{\mathbf{x}}_i^t = R_k(\mathbf{d}_k, \theta_t)(\mathbf{x}_i^c - \mathbf{o}_k) + \mathbf{o}_k$ for $i \in 1, \cdots, n$, $k \in 2, ..., K$

- notation: $o_k \in \mathbb{R}^3$, for $k \in 2, \cdots, K$
- ground truth: `data/sapien/*/gt/mobility_v2.json`(provided in `sapien`, not provided in `realscan`)
- estimation:
  - VideoArtGS: derived from motion pattern analysis and VLM inference
  - ArtTrans: inferred by the transformer model
- loss function: $\mathcal{L}_{track} = \mathcal{L}_{c2o} + \mathcal{L}_{o2o}$
  - $\mathcal{L}_{c2o} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i^t - \hat{\mathbf{x}}_i^t\|_2^2$ (canonical-to-observation)
  - $\mathcal{L}_{o2o} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i^{t_1} - \hat{\mathbf{x}}_i^{t_1}\|_2^2$ (observation-to-observation)
  - where $N$ is the number of sampled 3D tracking points, $\mathbf{x}_i^t$ is the tracked point position, and $\hat{\mathbf{x}}_i^t = \mathcal{F}(\mathbf{x}_i^c, t)$ is the predicted position transformed by articulation parameters.
- evaluation metric: $d(l_{pred}, l_{gt}) = \frac{|(o_{pred} - o_{gt}) \cdot (d_{pred} \times d_{gt})|}{\|d_{pred} \times d_{gt}\|}$

## motion parameters

### joint states

- notation: $\theta_{k,t}$ for $k \in 2, \cdots, K$ and $t \in 1, \cdots, T$
- ground truth: `data/sapien/*/gt/part_infos.json`provided in `sapien`, not provided in `realscan`
- estimation: inferred from the trained deformation field
- loss function: $\mathcal{L}_{track}$, $\mathcal{L}_{render}$, and $\mathcal{L}_{reg}$
- evaluation metric: mean absolute error between the predicted joint states and the ground truth joint states, i.e., $\frac{1}{T(K-1)} \limits\sum_{k=2}^K \sum_{t=1}^T |\theta_{k,t}^{pred} - \theta_{k,t}^{gt}|$, `cm` for prismatic joints and `degree` for revolute joints.


### joint motion limits / range
- notation: $\theta_{k}^{min}$, $\theta_{k}^{max}$ for $k \in 2, \cdots, K$
- ground truth: `data/sapien/*/gt/mobility_v2.json`(provided in `sapien`, not provided in `realscan`)
- estimation: 
    - VideoArtGS: inferred from the maximum and minimum of predicted joint states
    - ArtTrans: inferred from the transformer model
- loss function: /
- evaluation metric: mean absolute error between the predicted joint motion limits and the ground truth joint motion limits, i.e., $\frac{1}{K-1} \limits\sum_{k=2}^K |\theta_{k}^{min,pred} - \theta_{k}^{min,gt}| + |\theta_{k}^{max,pred} - \theta_{k}^{max,gt}|$, `cm` for prismatic joints and `degree` for revolute joints.