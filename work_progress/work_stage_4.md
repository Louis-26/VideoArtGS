# tasks
- complete refined high-level pipeline for VideoArtGS+PAT 
- consider VGGT and 3D tracking feature incorporation into PAT input with various potential approaches
    - feature information during VGGT and TAPIP3D for depth and 3D tracking feature
    - feature extraction with input of depth map and 3D tracking results
with 
    - **3D segmentation ground truth** for part segmentation training
    - **rendered images ground truth** for articulation parameter training
(transformation from non-learning-based to learning-based, with the help of transformer)


# work finished
- derive high-level pipeline(../assets/images/pipeline_revised.png) 
- try VGGT/TAPIP3D internal feature space as additional input to PAT
    - use hidden state of TAPIP3D EfficientUpdateFormer as additional input with dimension 384 (track_tapip)
    - use last-layer VGGT4Track features (multi-view averaged, PCA -> 128) as additional input with dimension 128 (vggt)
- do experiments on PAT with revised input features (xyz + normals + PartField + VGGT + TAPIP3D), summarized [here](../experiment_results/PAT_3/videoartgs_sapien_results.txt)


# potential next steps
- test on video feature encoder to replace VGGT/TAPIP3D extracted features





# 2026-09-04 progress: VGGT + TAPIP3D features as PAT input (in progress)
Analysis result (code as-is): PAT_B takes xyz(3) + normals(3) + PartField(448) = 454 dims per point from the dataset `point_cloud.ply` (not the trained Gaussians; README corrected). Token dim 768.

Implemented (see README "PAT Architecture" and `PAT/pat_extra_feats.py`):
- `track_geo` 56-d (explicit TAPIP3D trajectory + motion stats from `filtered.npz`)
- `track_tapip` 384-d (TAPIP3D EfficientUpdateFormer hidden state, exported via a hook in `third_party/TAPIP3D/inference.py`, `data_tools/extract_tapip3d_feats.py`)
- `vggt` 128-d (VGGT4Track last-layer 2048-d tokens, multi-view averaged, PCA -> 128, `data_tools/extract_vggt_feats.py`; 128 comps keep 97.5% variance)
- new `extra_embeds` branch in `particulate/models.py` (zero-init, checkpoint-equivalent at step 0); raw per-point input 454 -> 1022
- `PAT/PAT_finetune.py`: `--extra_feats`, `--labels track` (track-derived pseudo labels instead of spheres), `--train_on_all`, sidecar json
- `PAT/init_deform_PAT.py`: reads the sidecar and rebuilds the same inputs
- `scripts/videoartgs_pat3_chain.sh`: features -> fine-tune (all 20 sapien scenes, train = test on purpose) -> pipeline into `outputs_PAT_3` / `experiment_results/PAT_3`

Infra note: cvl12 GPU1 fell off the bus (cuInit error 999 for new processes); everything ran on cvl11 via ssh, GPUs restricted with `ALLOWED_GPUS`.

## 2026-09-04 result (outputs_PAT_3, experiment_results/PAT_3/README.md)
Chain finished 08:50 EDT (features -> fine-tune -> pipeline), all 20 scenes, no failures.

| run | Axis (deg) | Position (cm) | State | CD-w | CD-m | CD-s |
|---|---|---|---|---|---|---|
| orig (VideoArtGS) | 0.339 ± 0.830 | 0.101 ± 0.103 | 0.428 | 0.090 | 0.337 | 0.250 |
| PAT_1 (released + PartField) | 5.228 ± 8.274 | 1.678 ± 3.268 | 7.044 | 0.099 | 1.489 | 0.665 |
| PAT_2 (LoRA FT, 15/5) | 6.788 ± 10.454 | 1.696 ± 3.105 | 7.136 | 0.101 | 1.548 | 0.581 |
| PAT_3 (+track_geo/track_tapip/vggt, train=test) | 3.907 ± 5.770 | 3.765 ± 8.798 | 7.278 | 0.097 | 1.369 | 0.551 |

- PAT-level (fine-tune eval, in-train scenes): axis 14.8 -> 1.5 deg, label acc 0.89 -> 0.98. The model reproduces its targets; but the targets are `joint_infos.json` (motion analysis), so the injected axes differ from the original init by only 0.2-2 deg. The final per-scene swings (47648 21.6 -> 0.8 deg; 45194 7.6 -> 20.9 deg; 100481/101284 position 27-30 cm) come from train.py, not from the prior.
- Gap to orig is structural: PAT pipeline skips the track-loss stage-2 optimisation of the deformation field (`--iterations 1`) and still inits segmentation from joint_infos spheres.
- Next: fine-tune on real GT axes (gt/mobility_v2.json) to measure the ceiling; run init_deform track optimisation after PAT injection; init segmentation from PAT labels; ablate modalities on the held-out split.
