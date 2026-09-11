# VideoArtGS / VideoArtGS+PAT Experiment Summary

This document summarizes the completed orig and PAT_1--PAT_4 experiments. The VideoArtGS-SAPIEN table below uses the same 20 scenes for every run. Lower is better for all metrics.

## Metrics

- **Axis (deg):** Sign-invariant angular error between the predicted and ground-truth articulation axes.
- **Position (cm):** Distance between the predicted and ground-truth revolute axis lines. This metric is not applicable to prismatic joints.
- **State (deg/cm):** Joint-state error in degrees for revolute joints and centimeters for prismatic joints.
- **CD-w / CD-m / CD-s (cm):** Chamfer distance for the whole, movable, and static geometry, respectively.

CD-m is the most informative reconstruction metric for comparing PAT variants. CD-w is dominated by static geometry and may remain close to 0.1 cm even when movable-part reconstruction fails badly.

## Overall Results on VideoArtGS-20 SAPIEN

| Run | Axis (deg) | Position (cm) | State (deg/cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---:|---:|---:|---:|---:|---:|
| **orig** | **0.339 +/- 0.830** | **0.101 +/- 0.103** | **0.428 +/- 0.839** | **0.090 +/- 0.104** | **0.337 +/- 0.859** | **0.250 +/- 0.685** |
| PAT_1 | 5.228 +/- 8.274 | 1.678 +/- 3.268 | 7.044 +/- 14.460 | 0.099 +/- 0.120 | 1.489 +/- 2.824 | 0.665 +/- 1.336 |
| PAT_2 | 6.788 +/- 10.454 | 1.696 +/- 3.105 | 7.136 +/- 14.841 | 0.101 +/- 0.111 | 1.548 +/- 3.589 | 0.581 +/- 1.128 |
| PAT_3 | 3.907 +/- 5.770 | 3.765 +/- 8.798 | 7.278 +/- 15.033 | 0.097 +/- 0.108 | 1.369 +/- 3.089 | 0.551 +/- 1.069 |
| PAT_4 | 4.139 +/- 6.055 | 1.771 +/- 5.622 | 6.859 +/- 14.485 | 0.098 +/- 0.114 | 1.465 +/- 3.159 | 0.311 +/- 0.565 |

This is an end-to-end pipeline comparison, but it is not a clean generalization table:

- PAT_1 is zero-shot and does not train PAT on the VideoArtGS objects.
- PAT_2 uses a 15-object training split and a 5-object test split for PAT adaptation.
- PAT_3 and PAT_4 deliberately train on all 20 objects, including the evaluation objects. They are upper-bound/overfitting checks with train = test.
- After PAT initialization, VideoArtGS still optimizes a separate Gaussian and deformation model for each scene.

## orig: Reproduced VideoArtGS Baseline

### Objective

Reproduce the original VideoArtGS pipeline and establish the baseline used by all PAT comparisons.

### Pipeline

1. Train canonical 3D Gaussians from the static multi-view frames.
2. Use TAPIP3D trajectories and the hand-crafted motion-analysis pipeline to generate joint_infos.json.
3. Optimize the deformation field with track supervision for 10,000 iterations.
4. Jointly optimize the Gaussians and deformation field for 20,000 iterations with RGB/depth rendering losses and track loss.
5. Render the results and evaluate articulation and reconstructed geometry.

### Results and Interpretation

- The reproduction closely matches the reported paper result: the reproduced axis error is 0.339 deg, compared with 0.340 deg in the paper.
- orig remains substantially better than every current PAT pipeline on Axis, Position, State, and all three CD metrics.
- Its advantage does not come only from joint_infos.json. Unlike the PAT pipelines, orig performs a full 10k track-supervised deformation warm-up before joint training.

Result files:

- [VideoArtGS-20 SAPIEN](orig/videoartgs_sapien_results.txt)
- [Video2Articulation-S](orig/v2a_sapien_results.txt)
- [VideoArtGS real scan](orig/videoartgs_realscan_results.txt)

The Video2Articulation-S reproduction contains 73 completed scenes. Articulation outputs also exist for seven real-scan scenes, but all real-scan CD values are the failure/default value of 100. The real-scan CD results are therefore not yet valid quantitative results.

## PAT_1: Released PAT Checkpoint with Zero-Shot Inference

### Objective

Test whether the correctly released Particulate PAT checkpoint can replace the hand-crafted VideoArtGS articulation initialization without training PAT on any VideoArtGS object.

### Configuration

- PAT checkpoint: released particulate/model_ckpt/pat_model.pt.
- Per-point PAT input: xyz (3), normals (3), and PartField features (448).
- PAT predicts part segmentation and articulation parameters zero-shot.
- The legacy bridge retains the slot count, joint types, centers, and dist_max values from the motion-analysis joint_infos.json file. It replaces only the axis direction/origin of matched moving slots with PAT predictions.
- The deformation field is saved after only one initialization iteration. The pipeline skips the 10k track-supervised warm-up used by orig and proceeds directly to the regular 20k joint VideoArtGS training.

### Results and Interpretation

- Axis error increases from 0.339 deg for orig to 5.228 deg.
- CD-m increases from 0.337 cm to 1.489 cm.
- The main failure scenes are 100481, 101284, 103811, 45194, and 47648.
- The failures are associated with inaccurate or incomplete PAT part segmentation and the weak sphere-based initialization of the downstream segmentation field.

Result file: [PAT_1 results](PAT_1/videoartgs_sapien_results.txt)

## PAT_2: LoRA Fine-Tuning with a 15/5 Object Split

### Objective

Test whether adapting the released PAT model on VideoArtGS training objects improves performance on held-out objects.

### Configuration

- The base architecture and input features are the same as in PAT_1.
- Training starts from the released pat_model.pt checkpoint. This is fine-tuning, not full PAT training.
- Rank-8 LoRA is applied to the attention projections while the PAT backbone and heads remain frozen by default.
- Fifteen scenes are used for training. The five held-out scenes are 100481, 101284, 103811, 45194, and 47648.
- Part labels are sphere pseudo-labels constructed from the motion-analysis centers and dist_max values. Articulation targets also come from joint_infos.json.
- The pipeline uses the same legacy bridge and one-iteration deformation initialization as PAT_1.

### Results and Interpretation

- The 20-scene aggregate is slightly worse than PAT_1: Axis changes from 5.228 to 6.788 deg, and CD-m changes from 1.489 to 1.548 cm.
- At the PAT level, held-out mean axis error changes from approximately 14.12 deg before LoRA to 26.11 deg after LoRA.
- The following table recomputes the final end-to-end results on the same five held-out objects:

| Run | Axis | Position | State | CD-w | CD-m | CD-s |
|---|---:|---:|---:|---:|---:|---:|
| PAT_1 zero-shot | 17.698 | 3.676 | 20.366 | 0.188 | 4.903 | 1.417 |
| PAT_2 LoRA | 22.871 | 3.584 | 19.290 | 0.196 | 5.395 | 1.287 |

This experiment provides no evidence that the LoRA setup improves unseen-object generalization. The sphere pseudo-labels are also too weak and noisy to provide reliable part-segmentation supervision.

Result file: [PAT_2 results](PAT_2/videoartgs_sapien_results.txt)

## PAT_3: Additional Tracking and VGGT Inputs

### Objective

Test whether richer video and tracking features allow PAT to reproduce the VideoArtGS motion priors more accurately.

### Configuration

- Retain xyz, normals, and 448-dimensional PartField features.
- Add three per-point modalities:
  - track_geo (56-d): trajectory positions, visibility, and fitted motion statistics from filtered.npz;
  - track_tapip (384-d): hidden features from the TAPIP3D EfficientUpdateFormer;
  - vggt (128-d): multi-view VGGT tokens reduced with PCA.
- The raw feature dimension becomes 454 + 568 = 1022 before projection into the existing 768-dimensional PAT token space.
- Rank-8 LoRA is injected into all 72 attention linear layers. The new feature-embedding branches are also trained.
- Training runs for 120 epochs on all 20 scenes with --train_on_all. The five nominal test scenes are therefore included in the training set.
- Segmentation targets are track-derived pseudo-labels. Articulation targets remain the motion-analysis estimates from joint_infos.json.
- The pipeline uses the legacy PAT-to-VideoArtGS bridge.

### PAT-Level Results

- Mean axis error on the five monitored, in-training scenes improves from 14.82 to 1.54 deg.
- Hungarian-matched point-label accuracy improves from 0.891 to 0.984.

These results demonstrate that the LoRA model can fit the supplied training targets. They do not measure generalization because the monitored scenes are included in training, and the targets are motion-analysis estimates rather than ground truth.

### End-to-End Results

- Axis error improves relative to PAT_1 and PAT_2 but remains much worse than orig.
- Position error is the worst among all five pipelines at 3.765 cm.
- CD-m is 1.369 cm, approximately four times the orig result.
- Large per-scene changes after PAT inference show that the downstream train.py optimization, rather than only the transformer prediction, determines the final articulation parameters.

Result files:

- [PAT_3 pipeline results](PAT_3/videoartgs_sapien_results.txt)
- [PAT_3 fine-tuning metrics](PAT_3/pat_finetune_metrics.md)

## PAT_4: GT-Supervised PAT Upper-Bound Experiment

### Objective

Measure the ceiling of the PAT component and determine whether accurate PAT segmentation and articulation initialization improve the final VideoArtGS pipeline.

### Changes from PAT_3

- The input features and LoRA setup remain unchanged.
- GT segmentation from gt/part_k.ply replaces the track-derived pseudo-labels.
- GT axes and origins from gt/mobility_v2.json replace the joint_infos.json articulation targets.
- Training still uses all 20 scenes, so train = test.
- The pat_vlm bridge replaces the legacy bridge:
  - it retains the VLM/motion-analysis slot count, joint types, and order;
  - it fills each slot's center, extent, direction, and origin from the corresponding PAT part.

### PAT and Bridge Results

- PAT-level mean axis error improves from 13.11 to 0.05 deg.
- Hungarian-matched point-label accuracy improves from 0.915 to 0.997.
- Across all 20 scenes, the bridge initializes axes at 0.23 deg mean / 0.04 deg median error. The original motion-analysis initialization has 2.20 deg mean / 1.37 deg median error.
- Revolute origin error is 0.36 cm for the PAT bridge, compared with 0.59 cm for motion analysis.

### End-to-End Results and Diagnosis

Despite the nearly perfect PAT output, the final pipeline reaches 4.139 deg axis error and 1.465 cm CD-m. Accurate PAT initialization is frequently damaged during the following 20k VideoArtGS optimization:

- 25493: 0.03 -> 11.93 deg;
- 30666: 0.04 -> 15.43 deg;
- 45194: 0.02 -> 16.26 deg;
- 100481: 0.04 -> 17.17 deg;
- 103811: 0.03 -> 3.47 deg.

The main downstream problem is that PAT's per-point segmentation is saved as pat_segmentation.ply but is not used to initialize HybridSeg. The bridge transfers only center/dist_max spheres. Gaussians belonging to large movable parts may therefore be initialized in the static slot. The photometric and track losses subsequently move otherwise correct axes to compensate for incorrect segmentation.

Result files:

- [PAT_4 full summary](PAT_4/README.md)
- [PAT_4 pipeline results](PAT_4/videoartgs_sapien_results.txt)
- [PAT_4 fine-tuning metrics](PAT_4/pat_finetune_metrics.md)
- [PAT bridge initialization versus GT](PAT_4/init_vs_gt.md)
- [PAT bridge per-joint checks](PAT_4/bridge_check_all.txt)

## Conclusions Supported by the Completed Experiments

1. The reproduced orig VideoArtGS pipeline is correct and remains the strongest current baseline.
2. Replacing the track-trained deformation initialization with a one-step PAT injection causes a substantial performance drop.
3. PAT_2 shows that LoRA training with sphere pseudo-labels does not improve held-out-object performance.
4. PAT_3 shows that the augmented transformer can fit tracking and motion-analysis targets when the training and evaluation objects overlap.
5. PAT_4 shows that PAT can fit GT segmentation and articulation almost perfectly when train = test, but downstream VideoArtGS training destroys part of this accurate initialization.
6. PAT_2, PAT_3, and PAT_4 are not fully trained PAT experiments. Every run starts from the released PAT checkpoint and updates only LoRA and selected additional branches.
7. PAT_3 and PAT_4 cannot be reported as unseen-object generalization results because every evaluation object is included in PAT training.

## Required Work Before the Final PAT Experiment

1. Transfer PAT per-point labels directly into HybridSeg instead of using only center/extent spheres.
2. Add a track-supervised deformation warm-up after PAT injection and test early-stage axis freezing or regularization.
3. Use the existing PAT_4 upper-bound checkpoint to verify that accurate segmentation and axes survive downstream optimization.
4. Implement full PAT training from random initialization with all PAT parameters trainable. Loading pat_model.pt and then unfreezing it would still be fine-tuning rather than training from scratch.
5. Freeze object-level training, validation, and test manifests. Every view and joint belonging to an unseen asset must be excluded from the training set.
6. Report seen-object/different-view and unseen-object results separately. Test-time RGB/depth adaptation must be reported as a separate setting rather than mixed into the zero-shot generalization table.
