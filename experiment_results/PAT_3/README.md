# PAT_3: PAT with TAPIP3D + VGGT input features (2026-09-04)

Pipeline: stage-1 canonical Gaussians copied from `outputs_PAT` (identical, PAT-independent) -> `PAT/init_deform_PAT.py` with `particulate/model_ckpt/trained_PAT_model.pt` -> `train.py` (20k it, track loss 0.5) -> render -> eval. Command: `bash scripts/videoartgs_pat_pipeline.sh --use_multi 1 --keep_logs 1 --mode 1 --output_dir outputs_PAT_3 --save_dir PAT_3 --PAT_model_pth particulate/model_ckpt/trained_PAT_model.pt` (run on cvl11, GPUs 4,5,6,0,2).

## PAT checkpoint

`trained_PAT_model.pt` = released `pat_model.pt` + LoRA (r=8, all attention Linears) + new `extra_embeds` branches, fine-tuned 120 epochs on **all 20 videoartgs-sapien scenes (train = test, deliberately)**, GT = `joint_infos.json` axes (from the TAPIP3D motion analysis, not `gt/mobility_v2.json`) and track-derived part labels (`--labels track`).

Extra per-point inputs (raw input 454 -> 1022 dims, token dim 768 unchanged):

| modality | dim | source |
|---|---|---|
| track_geo | 56 | TAPIP3D trajectories (`filtered.npz`): displacement at 8 stamps, visibility, motion stats, fitted axis, motion-type one-hot, kNN confidence |
| track_tapip | 384 | TAPIP3D EfficientUpdateFormer hidden state per track (hook on `point_updater.flow_head`), kNN-transferred |
| vggt | 128 | VGGT4Track last aggregator layer (2048-d frame‖global tokens), 24 canonical views averaged per point, PCA 2048->128 (97.5% var.) |

PAT-level eval on the 5 former test scenes (in the training set): mean axis error 14.82 deg -> 1.54 deg, matched-slot label accuracy 0.891 -> 0.984 (details in `pat_finetune_metrics.md`).

## Pipeline results, videoartgs-sapien (n=20)

| run | Axis (deg) | Position (cm) | State (deg/cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|
| orig (VideoArtGS, track-trained init_deform) | 0.339 ± 0.830 | 0.101 ± 0.103 | 0.428 ± 0.839 | 0.090 ± 0.104 | 0.337 ± 0.859 | 0.250 ± 0.685 |
| PAT (released ckpt, first integration) | 28.059 ± 24.778 | 3.978 ± 6.665 | 9.025 ± 15.368 | 0.096 ± 0.103 | 4.133 ± 7.732 | 0.856 ± 1.631 |
| PAT_1 (released ckpt + PartField, xyz/normals) | 5.228 ± 8.274 | 1.678 ± 3.268 | 7.044 ± 14.460 | 0.099 ± 0.120 | 1.489 ± 2.824 | 0.665 ± 1.336 |
| PAT_2 (LoRA fine-tuned, sphere labels, 15/5 split) | 6.788 ± 10.454 | 1.696 ± 3.105 | 7.136 ± 14.841 | 0.101 ± 0.111 | 1.548 ± 3.589 | 0.581 ± 1.128 |
| **PAT_3 (this run: + track_geo/track_tapip/vggt, train = test)** | 3.907 ± 5.770 | 3.765 ± 8.798 | 7.278 ± 15.033 | 0.097 ± 0.108 | 1.369 ± 3.089 | 0.551 ± 1.069 |

Per-scene axis error (deg) and position error (cm):

| scene | axis PAT_1 | axis PAT_2 | axis PAT_3 | pos PAT_1 | pos PAT_2 | pos PAT_3 | CD-m PAT_1 | CD-m PAT_3 |
|---|---|---|---|---|---|---|---|---|
| 168 | 0.36 | 5.38 | 5.33 | 8.49 | 10.25 | 9.98 | 1.560 | 1.490 |
| 1280 | 0.38 | 0.61 | 0.57 | 0.39 | 0.41 | 0.44 | 0.128 | 0.141 |
| 8961 | 0.02 | 0.02 | 0.01 | 0.06 | 0.03 | 0.08 | 0.023 | 0.025 |
| 9016 | 0.09 | 0.10 | 0.10 | 0.41 | 0.34 | 0.35 | 0.019 | 0.019 |
| 10489 | 0.09 | 0.15 | 0.10 | 0.22 | 0.17 | 0.21 | 0.010 | 0.010 |
| 10655 | 0.02 | 0.02 | 0.02 | 0.14 | 0.24 | 0.23 | 0.013 | 0.013 |
| 25493 | 0.12 | 0.13 | 0.11 | 0.00 | 0.00 | 0.00 | 0.148 | 0.154 |
| 30666 | 4.87 | 4.12 | 5.74 | 0.00 | 0.00 | 0.00 | 3.161 | 4.495 |
| 31249 | 0.07 | 0.06 | 0.07 | 0.03 | 0.02 | 0.03 | 0.067 | 0.076 |
| 45194 | 7.61 | 9.31 | 20.87 | 0.05 | 0.06 | 0.04 | 0.831 | 0.522 |
| 45503 | 0.02 | 0.02 | 0.02 | 0.12 | 0.13 | 0.13 | 0.013 | 0.012 |
| 45612 | 0.18 | 0.28 | 2.10 | 0.83 | 0.84 | 2.35 | 0.071 | 0.335 |
| 47648 | 21.65 | 24.89 | 0.83 | 1.27 | 2.49 | 0.43 | 5.256 | 0.488 |
| 100481 | 12.68 | 31.94 | 14.51 | 11.86 | 7.73 | 29.94 | 7.150 | 5.495 |
| 101284 | 24.81 | 26.32 | 7.47 | 5.20 | 7.64 | 27.19 | 1.152 | 1.252 |
| 101287 | 5.75 | 6.22 | 4.63 | 4.07 | 3.17 | 3.53 | 0.004 | 0.012 |
| 101808 | 3.81 | 3.79 | 3.81 | 0.05 | 0.06 | 0.08 | 0.020 | 0.022 |
| 101908 | 0.12 | 0.37 | 0.16 | 0.10 | 0.09 | 0.09 | 0.013 | 0.013 |
| 103015 | 0.16 | 0.14 | 0.13 | 0.24 | 0.25 | 0.19 | 0.005 | 0.009 |
| 103811 | 21.74 | 21.89 | 11.53 | 0.00 | 0.00 | 0.00 | 10.127 | 12.802 |
| PAT_1 mean / median | 5.23 / 0.27 | – | – | 1.68 / 0.18 | – | – | 1.489 | – |
| PAT_2 mean / median | 6.79 / 0.49 | – | – | 1.70 / 0.20 | – | – | 1.548 | – |
| PAT_3 mean / median | 3.91 / 0.70 | – | – | 3.76 / 0.20 | – | – | 1.369 | – |

## Reading the numbers

- Mean axis error is the best of the PAT variants (3.91 deg vs 5.23 PAT_1 / 6.79 PAT_2) but the median got worse (0.70 vs 0.27) and the mean position error doubled (3.77 cm vs 1.68), driven by 100481 (29.9 cm) and 101284 (27.2 cm). Big wins: 47648 (21.6 -> 0.8 deg, CD-m 5.3 -> 0.5), 103811 (21.7 -> 11.5 deg), 101284 axis (24.8 -> 7.5). Regressions: 45194 (7.6 -> 20.9 deg), 168 (0.4 -> 5.3), 45612 (0.2 -> 2.1).
- At PAT level the fine-tuned model reproduces its training targets almost exactly (1.5 deg mean, 98% label accuracy). In the bridge logs the injected axes differ from the original `joint_infos.json` init by only 0.2-2 deg for almost every joint (`logs/logs_init_deform_PAT_<scene>.txt`), because the fine-tune GT *is* `joint_infos.json`. So PAT_3 mostly re-injects the motion-analysis axes; the remaining per-scene swings come from the 20k-iteration joint training (train.py), not from the initial axis.
- That also explains the gap to `orig` (0.34 deg): the original pipeline trains the deformation field with track losses in stage 2 (`init_deform.py`, ~2 h for 20 scenes), whereas the PAT pipeline injects PAT's axes/centers zero-shot (`--iterations 1`) and leaves everything to train.py. The bottleneck is the stage-2 optimisation and the segmentation init (centres/dist_max still come from `joint_infos.json`), not the axis prior.
- 100481 is the one scene where the tracks themselves are weak (only 1.7% of filtered tracks move; 39-59% of the moving-part GT points have a track within 3 cm), so the track-based inputs and labels cannot help there.

## Caveats
- train = test on all 20 scenes: these numbers say nothing about generalisation. For a fair run drop `--train_on_all` (5 held-out scenes: 100481 101284 103811 45194 47648).
- Stage 1 was reused from `outputs_PAT`; stages 2-5 were run fresh.

## Next steps worth trying
1. Fine-tune against the real GT axes (`gt/mobility_v2.json`) instead of `joint_infos.json`, still train = test, to measure the pipeline ceiling when PAT is perfect.
2. Run the track-loss stage-2 optimisation (`init_deform.py`) *after* the PAT injection instead of `--iterations 1`.
3. Use PAT's part labels to initialise the segmentation module (centres / extents) instead of the `joint_infos.json` spheres.
4. Ablate the three modalities (track_geo only vs + track_tapip vs + vggt) with the held-out split.

## Files
- `videoartgs_sapien_results.txt` – eval summary from `utils/results_summary.py`
- `pat_finetune_metrics.md` – PAT-level before/after eval and loss curve
- `logs/PAT_finetune_PAT3.txt`, `logs/pipeline_PAT3.txt`, `logs/logs_*_<scene>.txt` – full logs
- `particulate/model_ckpt/trained_PAT_model.pt` + `.json` sidecar; `particulate/model_ckpt/finetune_PAT3/` (label dumps, metrics.json, lora_adapter.pt)
