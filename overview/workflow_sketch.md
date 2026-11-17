# VideoArtGS pipeline
1. Given realscan RGB images $\{I_t\}$, t=1...T, From *VGGT*, derive 
- depth maps $\{D_t\}$, t=1,...,T
- camera intrinsics, $\{K_t\}$, t=1,...,T
- camera extrinsics, $\{E_t\}$, t=1,...,T
2. Design systematic prompt, and put RGB images $\{I_t\}$ into *GPT-4o*, infer part number $K$ and joint type $\tau_k$ with $k=2,\cdots,K$ 
3. Put monocular RGB images $\{I_t\}$ into *SAM2*，associated with annotations(bounding box or points), derive binary masks $\{M_t\}$, t=1,...,T
4. Put monocular RGB images $\{I_t\}$ into *TAPIP3D*，compute 3D tracking trajectories, $\{X_t\}_{t=M+1}^T \in \mathbb{R}^{(T-M) \times P \times 3}$, with $X_t=\{X_{i,t}\}_{i=1}^P \in \mathbb{R}^{P \times 3}$  
5. Given the following inputs: 
  - RGB images $\{I_t\}_{t=1}^M$
  - Depth maps $\{D_t\}_{t=1}^M$
  - camera intrinsics $\{K_t\}_{t=1}^M$ 
  - camera extrinsics $\{E_t\}_{t=1}^M$ 
  - mask images $\{M_t\}_{t=1}^M$
Construct fused point cloud $\mathcal{P}_{cano}={(x_i,c_i,n_i)}_{i=1}^N$ by local averaging and downsampling 

1. Construct **canonical 3DGS** given  
  - **First M static** multiview monocular video frames $\{I_t\}_{t=1}^M \in \mathbb{R}^{M \times H \times W \times 3}$
  - Corresponding monocular depth maps $\{D_t\}_{t=1}^M \in \mathbb{R}^{M \times H \times W}$ 
  - 3D initial fused point cloud $\mathcal{P} \in \mathbb{R}^{N_{init} \times 3}$ with **first M static** video frames 
  - Camera poses(GT for `sapien`, VGGT estimate for `realscan`) $\{P_t\}_{t=1}^M$, (extrinsic matrix $ E = P^{-1} \in \mathbb{R}^{4 \times 4}$) for each selected frame 
  - Camera intrinsics(GT for `sapien`, VGGT estimate for `realscan`) $K \in \mathbb{R}^{3 \times 3}$
Derive N Gaussian primitives $\{\mathcal{G}_i\}_{i=1}^N$, and each Gaussian primitive $\mathcal{G}_i$ has the following propeties:
   - mean coordinate $\mu_i \in \mathbb{R}^3$
   - quaternion rotation $r_i \in \mathbb{R}^4$
   - scale vector $s_i \in \mathbb{R}^{3}$
   - opacity $\sigma_i \in \mathbb{R}$
   - spherical harmonics coefficients $h_i \in \mathbb{R}^{48}$
1. Based on 3D tracking information $X \mathbb{R}^{P \times 3}$, conduct motion analysis and motion clustering, then for $k=2,\cdots,K$, derive the priors of
  - articulation axis $\{d_k\} \in \mathbb{R}^3$
  - articulation origin $\{o_k\} \in \mathbb{R}^3$
  - part center $\{c_k\} \in \mathbb{R}^3$ 
Then compute part radial extent $\{s_k\}$ based on each cluster by $a \max \limits_{i \in C_k} ||X_i - c_k||$, $a=0.2$ if $k=2,\cdots, K$, otherwise $a=1$
1. In Deformation Field
inject the following priors into the segmentation model
   - part center $\{c_k\}$  
   - part radial extent $\{s_k\}$  into segmentation model  
inject the following priors into the articulation model
   - articulation axis $\{d_k\}$  
   - articulation origin $\{o_k\}$
Then output the per-Gaussian part assignment $m=\{m_i\}_{i=1}^N$, and $m_i=[m_{i1},\cdots, m_{iK}]$
1. For time $t=1,...,T$, compute joint states $\theta_t$ from the time model
refine articulation axis $\{d_k\}$ and articulation origin $\{o_k\}$, derive dual quaternion $q_k=q_{k,r}+\epsilon q_{k,d}$ for each movable part $k=2,\cdots,K$, then derive $q_{g,i}^t=\sum\limits_{k=1}^K m_{ik}q_k^t$ for all Gaussian primitives $i=1,\cdots,N$, where $m_{ik}$ is the part assignment weight for each Gaussian primitive, and get the corresponding normalized dual quaternion $\hat{q}_{g,i}^t=\frac{q_{g,i}^t}{||q_{g,i}^t||}$, 
Then derive rotation matrix 
$R_i^t = \begin{bmatrix} 
1-2y^2-2z^2 & 2xy-2wz & 2xz+2wy \\ 
2xy+2wz & 1-2x^2-2z^2 & 2yz-2wx \\ 
2xz-2wy & 2yz+2wx & 1-2x^2-2y^2 
\end{bmatrix}$
and $t_i=2(q_{k,d} \otimes q_{k,r}^*)=[0,t]$ 
Then compute $X_{i,t}=R_i^t \cdot X_{i,cano}+t_i$, emanwhile, update rotation of gaussian primitive $r_i^t=q_{i,r}^t \otimes r_i^c$
1.  According to ground truth depth maps $\{D_k\}_{k=1}^T$ and ground truth RGB images $\{I_k\}$, jointly optimize gaussian primitive parameters and part/articulation parameters, and accordingly update dual quaternion transfromation from dual quaternion linear blending.  
2.  render images from camera poses, and compute evaluation metrics
- chamfer distance loss for whole, static and dynamic parts
- articulation error mean+std
- joint state error mean+std
  


# ArtTrans pipeline
