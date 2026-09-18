
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

All eight runs use the PAT_3 recipe unchanged (`PAT/PAT_finetune.py --epochs 120
--labels track --train_on_all`, LoRA on attention, the new input branches trained
fully and zero-initialised) and differ only in `--extra_feats`, so the table
isolates the inputs rather than the schedule. Axis error and point-label accuracy
are what the fine-tuning run reports on its held-out joints.

Reading: each modality on its own lowers axis error, `track_tapip` by the most
(1.76 -> 1.32 deg). Combining modalities does **not** compound — every pair and
the full set land back near 1.7 deg — while segmentation keeps improving (0.962
-> 0.984). The extra branches buy part assignment, and past one motion modality
they start competing with the axis heads for capacity.
