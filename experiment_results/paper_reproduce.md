# VideoArtGS
## Quantitative Results
code
```bash
cd "$(git rev-parse --show-toplevel)"

python experiment_results/results_summary.py \
    --dataset v2a --subset sapien \
    --split-joint --with-state 
    
python experiment_results/results_summary.py --dataset videoartgs --subset sapien

```
### Table 1: Quantitative evaluation on Video2Articulation-S (v2a) dataset


| Method | Revolute Axis (°) | Revolute Position (cm) | Revolute State (°) | Prismatic Axis (°) | Prismatic State (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|---|---|
|  Reproduced | 0.344 ± 0.419 | 0.469 ± 0.917 | 3.406 ± 13.899 | 0.361 ± 0.439 | 0.670 ± 2.315 | 0.308 | 0.884 | 0.402 |
|  Paper | 0.320 ± 0.440 | 0.420 ± 0.750 | 1.150 ± 2.290 | 0.350 ± 0.450 | 1.030 ± 2.460 | 0.290 | 0.400 | 1.110 |



Revolute Joint Estimation (n=44)

| Method | Axis (deg) | Position (cm) | State | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|
|  Reproduced (n=44) | 0.344 ± 0.419 | 0.469 ± 0.917 | 3.406 ± 13.899 | 0.264 ± 0.321 | 0.266 ± 0.556 | 0.325 ± 0.402 |
|  Paper | 0.320 ± 0.440 | 0.420 ± 0.750 | 1.150 ± 2.290 | 0.290 ± 0.240 | 0.400 ± 0.320 | 1.110 ± 2.110 |

Prismatic Joint Estimation (n=29)

| Method | Axis (deg) | State | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|
|  Reproduced (n=29) | 0.361 ± 0.439 | 0.670 ± 2.315 | 0.376 ± 0.282 | 1.821 ± 2.246 | 0.520 ± 0.363 |
|  Paper | 0.350 ± 0.450 | 1.030 ± 2.460 | 0.290 ± 0.240 | 0.400 ± 0.320 | 1.110 ± 2.110 |


### Table 2: Quantitative evaluation on VideoArtGS-20 (videoartgs) sapien dataset
=== Per-scene metrics ===
 scene  angle  distance  CD_whole  CD_dynamic  CD_static
100481 0.0134    0.0178    0.1037      0.0199     0.1031
101284 0.1034    0.0686    0.0099      0.0034     0.0083
101287 0.2506    0.0513    0.0099      0.0034     0.0067
101808 3.7933    0.0667    0.1036      0.0167     0.0967
101908 0.1104    0.0951    0.0957      0.0126     0.0943
103015 0.1610    0.2247    0.0480      0.0104     0.0448
103811 0.2342    0.0000    0.4967      3.7152     0.5570
 10489 0.0878    0.2346    0.0661      0.0088     0.0524
 10655 0.0150    0.1645    0.0826      0.0134     0.0650
  1280 0.7231    0.3292    0.0295      0.0888     0.0897
   168 0.4328    0.1591    0.0281      1.0380     3.1172
 25493 0.1082    0.0000    0.0609      0.1570     0.1113
 30666 0.1309    0.0000    0.1844      1.1252     0.2226
 31249 0.0716    0.0118    0.0958      0.1937     0.1055
 45194 0.0932    0.0287    0.1021      0.0127     0.0846
 45503 0.0189    0.0912    0.0832      0.0116     0.0705
 45612 0.1160    0.0511    0.0738      0.0173     0.0608
 47648 0.2064    0.0677    0.0615      0.2630     0.0599
  8961 0.0215    0.0397    0.0359      0.0224     0.0178
  9016 0.0962    0.3188    0.0322      0.0147     0.0254

=== Markdown table ===

| Method | Axis (deg) | Position (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|
| Ours (reproduced) (n=20) | 0.339 ± 0.830 | 0.101 ± 0.103 | 0.090 ± 0.104 | 0.337 ± 0.859 | 0.250 ± 0.685 |
| Ours (paper) | 0.340 ± 0.800 | 0.100 ± 0.100 | 0.090 ± 0.090 | 0.260 ± 0.610 | 0.240 ± 0.580 |

### Table 3: Quantitative evaluation on VideoArtGS-20 (videoartgs) realscan dataset


## Qualitative Results
All results have been completed under `outputs/{DATASET}/sapien/{SCENE}/final/train/ours_20000`, where `DATASET` is either `v2a` or `videoartgs`, and `SCENE` is the corresponding scene name, as 
- `{SCENE}_colored.ply` for colored mesh segmentation results
- `{SCENE}_video.gif` for reconstructed object gif results
- `{SCENE}_video.mp4` for reconstructed object video results

This summarizes experimental results for figure 2 and figure 3 from the paper.

# VideoArtGS+PAT
## Quantitative Results
With 20 scenes in VideoArtGS-20 dataset.
performance per scene
=== Per-scene metrics ===
 scene   angle  distance  CD_whole  CD_dynamic  CD_static
100481 79.4013   29.5709    0.3352     24.5797     0.9236
101284 41.6638    4.1543    0.0388      0.1446     0.0078
101287 43.7547    8.6631    0.0117      0.1466     4.1244
101808  3.8061    0.0468    0.1058      0.0242     0.0996
101908 10.2581    1.0757    0.0953      0.0575     0.1114
103015  8.6050    0.1618    0.2152      0.0120     0.2419
103811 40.4327    0.0000    0.4610      4.9140     2.3420
 10489  0.0615    0.2158    0.0634      0.0102     0.0495
 10655  0.0145    0.1176    0.0835      0.0147     0.0643
  1280 28.9467    0.4805    0.0246      0.0828     0.1085
   168 45.7036   12.7696    0.0296      3.1628     6.8006
 25493 22.0398    0.0000    0.0664      0.6530     0.1192
 30666 35.2883    0.0000    0.1906     12.5942     0.2330
 31249  1.8326    0.0574    0.0915      0.1260     0.0960
 45194 40.3279    0.0525    0.1092      3.8128     1.3774
 45503 28.9833    3.2415    0.0793      0.0176     0.1152
 45612 72.4959   24.0773    0.0796     20.8945     0.6084
 47648 34.6226   10.9388    0.0620     21.0200     0.1913
  8961  0.0177    0.0401    0.0349      0.0214     0.0179
  9016  0.0997    0.3691    0.0322      0.0158     0.0243



