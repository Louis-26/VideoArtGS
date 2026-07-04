# Quantitative Results
code
```bash
cd "$(git rev-parse --show-toplevel)"

python experiment_results/results_summary.py \
    --dataset v2a --subset sapien \
    --split-joint --with-state 
    
python experiment_results/results_summary.py --dataset videoartgs --subset sapien

```
## Table 1: Quantitative evaluation on Video2Articulation-S (v2a) dataset


| Method | Revolute Axis (°) | Revolute Position (cm) | Revolute State (°) | Prismatic Axis (°) | Prismatic State (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|---|---|
|  Reproduced | 0.344 ± 0.419 | 0.469 ± 0.917 | 3.406 ± 13.899 | 0.361 ± 0.439 | 0.670 ± 2.315 | 0.308 | 0.884 | 0.402 |
|  Paper | 0.320 ± 0.440 | 0.420 ± 0.750 | 1.150 ± 2.290 | 0.350 ± 0.450 | 1.030 ± 2.460 | 0.290 | 0.400 | 1.110 |



### Revolute Joint Estimation (n=44)

| Method | Axis (deg) | Position (cm) | State | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|
|  Reproduced (n=44) | 0.344 ± 0.419 | 0.469 ± 0.917 | 3.406 ± 13.899 | 0.264 ± 0.321 | 0.266 ± 0.556 | 0.325 ± 0.402 |
|  Paper | 0.320 ± 0.440 | 0.420 ± 0.750 | 1.150 ± 2.290 | 0.290 ± 0.240 | 0.400 ± 0.320 | 1.110 ± 2.110 |

### Prismatic Joint Estimation (n=29)

| Method | Axis (deg) | State | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|
|  Reproduced (n=29) | 0.361 ± 0.439 | 0.670 ± 2.315 | 0.376 ± 0.282 | 1.821 ± 2.246 | 0.520 ± 0.363 |
|  Paper | 0.350 ± 0.450 | 1.030 ± 2.460 | 0.290 ± 0.240 | 0.400 ± 0.320 | 1.110 ± 2.110 |


## Table 2: Quantitative evaluation on VideoArtGS-20 (videoartgs) dataset
| Method | Axis (deg) | Position (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|
| Reproduced | 0.333 ± 0.827 | 0.099 ± 0.103 | 0.096 ± 0.099 | 0.311 ± 0.738 | 0.243 ± 0.621 |
| Paper | 0.340 ± 0.800 | 0.100 ± 0.100 | 0.090 ± 0.090 | 0.260 ± 0.610 | 0.240 ± 0.580 |



# Qualitative Results
All results have been completed under `outputs/{DATASET}/sapien/{SCENE}/final/train/ours_20000`, where `DATASET` is either `v2a` or `videoartgs`, and `SCENE` is the corresponding scene name, as 
- `{SCENE}_colored.ply` for colored mesh segmentation results
- `{SCENE}_video.gif` for reconstructed object gif results
- `{SCENE}_video.mp4` for reconstructed object video results

This summarizes experimental results for figure 2 and figure 3 from the paper.