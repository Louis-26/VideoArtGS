# Date
06/08/2026 - 06/14/2026

# tasks
✅1. reproduce table 1 as model evaluation from dataset `Video2Articulation-S (v2a)` from paper page 7

✅2. reproduce table 2 as model evaluation from dataset `VideoArtGS-20 (videoartgs)` from paper page 8

✅3. reproduce qualitative reconstructed results as figure 2 from paper page 7 for dataset `Video2Articulation-S (v2a)`, with the form of  **gif of reconstructed object** and **mesh segmentation results**

✅4. reproduce qualitative reconstructed results as figure 3 from paper page 8 for dataset `VideoArtGS-20 (videoartgs)`, with the form of  **gif of reconstructed object** and **mesh segmentation results**
  
# finished work summary 
Reproduced all tables and qualitative results, including 
- evaluation metric for both `VideoArtGS` dataset and `Video2Articulation` dataset, visible at [here](../experiment_results/paper_reproduce.md), including
    - axis error mean and standard deviation
    - position error mean and standard deviation(only for revolute part)
    - state error mean and standard deviation (only for prismatic part) 
    - Chamfer Distance for whole part (CD-w) mean and standard deviation
    - Chamfer Distance for moving part (CD-m) mean and standard deviation
    - Chamfer Distance for static part (CD-s) mean and standard deviation
- qualitative results for both `VideoArtGS` dataset and `Video2Articulation` dataset
    - video of reconstructed object results, visible under `outputs/{DATASET}/sapien/{SCENE}/final/train/ours_20000/{SCENE}_video.mp4`
    - gif of reconstructed object results, visible under `outputs/{DATASET}/sapien/{SCENE}/final/train/ours_20000/{SCENE}_video.gif`
    - colored mesh segmentation results, visible under `outputs/{DATASET}/sapien/{SCENE}/final/train/ours_20000/{SCENE}_colored.ply`

# potential next steps
- Revise and format the train script to enable parameters for dataset and subset input to control.
- Figure out the 3D visualization tools for point clouds and meshes.(right now cloud compare and blender)
# reference
[Videoartgs: Building Digital Twins Of Articulated Objects From Monocular Video](https://arxiv.org/pdf/2509.17647)