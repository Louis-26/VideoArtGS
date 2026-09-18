
### PAT extra-input ablation (VideoArtGS-20)

| Extra per-point inputs | Axis error (deg) | Point-label acc | n joints |
|---|---|---|---|
| -- (released PAT inputs only) | 1.76 | 0.962 | 20 |
| + track_geo | 1.69 | 0.977 | 20 |
| + track_tapip | 1.32 | 0.969 | 20 |
| + vggt | 1.45 | 0.974 | 20 |
| + track_geo + track_tapip | 1.69 | 0.980 | 20 |
| + track_geo + vggt | 1.70 | 0.984 | 20 |
| + track_tapip + vggt | 1.72 | 0.975 | 20 |
| + all three | 1.74 | 0.983 | 20 |
