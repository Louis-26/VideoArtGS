# fused point cloud construction
- Input
  - RGB images $\{I_t\}_{t=1}^M$
  - Depth maps $\{D_t\}_{t=1}^M$
  - camera intrinsics $\{K_t\}_{t=1}^M$ 
  - camera extrinsics $\{E_t\}_{t=1}^M$, so that 
  - mask images $\{M_t\}_{t=1}^M$

- Computation
  - back project to camera space
    - $x_{cam}=\frac{u-c_{x,t}}{f_{x,t}}D_t(u,v)$
    - $y_{cam}=\frac{v-c_{y,t}}{f_{y,t}}D_t(u,v)$
    - $z_{cam}=D_t(u,v)$
  - camera to world transformation
    - $x_{world}=E_t^{-1}x_{cam}$
  - finite difference normal estimation
    - $n=\frac{\Delta_x \times \Delta_y}{\lvert \Delta_x \times \Delta_y \rvert _2}$
    - $\Delta_x=X_{u+1,v}-X_{u-1,v}$
    - $\Delta_y=X_{u,v+1}-X_{u,v-1}$
  - multimodal filtering
  - voxel downsampling and outlier removal
    - average point within a small voxel 
- Output 
  - fused point cloud $\mathcal{P}_{cano}={(x_i,c_i,n_i)}_{i=1}^N$ 


# part radial extent computation
- Input
  - part center $\{c_k\}_{k=1}^K$
  - 3D point cloud $\{x_i\}_{i=1}^N$
 
- Computation
  - for each part $k$, compute the radial extent $s_k$ as the maximum distance from the part center $c_k$ to the points $x_i$ belonging to that part:
    - for movable parts: $s_k = \max_{i \in \text{part } k} \lVert x_i - c_k \rVert _2$, $k=2, ..., K$
    - for static parts: $s_k = \max_{i \in \text{part } k} \lVert x_i - c_k \rVert _2$, $k=1$

- Output
  - part radial extent $\{s_k\}_{k=1}^K$



# motion analysis
- Input 
  - sampled tracking point number $P$
  - 3D tracking trajectory $\{X_i\}_{i=1}^P$ where each $X_i={X_{i,t}}_{t=M+1}^T$

- Classification
  - if $\Delta_{max}=\max_t \lVert x_t -\hat{x} \rVert _2 \lt \epsilon_{static}$: static
  - if distance to the first principal component $\lVert x_t - d \rVert _2 \lt \epsilon_{prismatic}$: prismatic
  - if distance to the fitted circle $\lVert x_t - c \rVert _2 - r^2 \lt \epsilon_{revolute}$: revolute
  - otherwise, noise

- Output
  - trajectory type $\tau_i \in {Static, Prismatic, Revolute, Noise}$
  - direction axis $d_i$ and origin $o_i$

# motion clustering
- Input
  - sample number $P$
  - part number $K$
  - 3D tracking trajectory $\{X_i\}_{i=1}^P$ where each $X_i={X_{i,t}}_{t=M+1}^T$

- Output
  - partial radial extent $\{s_k\}_{k=2}^K$
  - articulation axis prior $\{d_k\}_{k=2}^K$
  - articulation origin prior $\{o_k\}_{k=2}^K$
  - part center prior $\{c_k\}_{k=1}^K$


# 3D Gaussian Splatting

- Input: 
  - **First M static** multiview monocular video frames $\{I_t\}_{t=1}^M \in \mathbb{R}^{M \times H \times W \times 3}$
  - Corresponding monocular depth maps $\{D_t\}_{t=1}^M \in \mathbb{R}^{M \times H \times W}$ 
  - 3D initial fused point cloud $\mathcal{P} \in \mathbb{R}^{N_{init} \times 3}$ with **first M static** video frames 
  - Camera poses(GT for `sapien`, VGGT estimate for `realscan`) $\{P_t\}_{t=1}^M$, (extrinsic matrix $ E = P^{-1} \in \mathbb{R}^{4 \times 4}$) for each selected frame 
  - Camera intrinsics(GT for `sapien`, VGGT estimate for `realscan`) $K \in \mathbb{R}^{3 \times 3}$

- Output
  - Canonical 3D Gaussian primitives $\mathcal{G} = \{g_i\}_{i=1}^N$
    - $\mu \in \mathbb{R}^{3}$
    - rotation $q \in \mathbb{R}^{4}$
    - scale $s \in \mathbb{R}^{3}$
    - opacity $\alpha \in [0, 1]$
    - Spherical Harmonics feature $h \in \mathbb{R}^{48}$

# quaternion computation
- Input
  - articulation type $\tau_k$
  - motion state $\theta_k(t)$

- Computation
  
  - if $\tau_k$ is static: $q_{k,r} = [1, 0, 0, 0]$, $q_{k,d} = [0, 0, 0, 0]$
  - if $\tau_k$ is prismatic: $q_{k,r} = [1, 0, 0, 0]$, $q_{k,d} = \frac{1}{2}[0, d_k\theta_k(t)]$
  - if $\tau_k$ is revolute: $q_{k,r} = [cos(\theta_k(t)/2), sin(\theta_k(t)/2) d_k]$, $q_{k,d} = \frac{1}{2}(t_0 \otimes q_{k,r}-q_{k,r} \otimes t_0)$

- Output
  - real quaternion $q_{k,r}$
  - dual quaternion $q_{k,d}$ 

# rotation or translation computation with quaternion

- Input 
  - 3D point $x_i \in \mathbb{R}^3$
  - articulation origin $o_k \in \mathbb{R}^3$
  - articulation axis $d_k \in \mathbb{R}^3$
  - motion state $\theta_k$

- Computation
  - real quaternion $q_{k,r}=[\cos(\frac{\theta_k}{2}),\sin(\frac{\theta_k}{2})d_k]^\top \in \mathbb{R}^4$
  - dual quaternion $q_{k,d}=\frac{1}{2}(t_0 \otimes q_{k,r}-q_{k,r} \otimes t_0)$
  - get $\hat{q_k}=q_{k,r} + \epsilon q_{k,d}$ where $\epsilon$ is the dual unit
  - $t_0=2(q_{k,d} \otimes q_{k,r}^*)=[0,t]$ 
- Output 
  - rotated point $x_i=q_{k,r} \otimes x_i \otimes q_{k,r}^*+t$ 

