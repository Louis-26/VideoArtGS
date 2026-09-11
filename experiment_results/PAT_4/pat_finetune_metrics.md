# PAT fine-tune with GT supervision (trained_PAT_gt_model.pt), 2026-09-04

Config: released `pat_model.pt` + LoRA r=8 on all 72 attention Linears + `extra_embeds` branches (track_geo 56, track_tapip 384, vggt 128), 120 epochs x 20 scenes, 2048 pts/step, grad-accum 4, extra_dropout 0.2, `--train_on_all` (TEST_SCENES are in the training set: overfitting check, not generalisation).

Difference to PAT_3: `--labels gt --joint_gt gt --bridge pat_vlm`.
- per-point part labels = nearest surface sample of the GT part meshes `gt/part_k.ply` (k=0 static base, k>=1 the k-th joint of `utils.metrics.read_gt(gt/mobility_v2.json)`), 300k area-proportional samples per scene, point-to-GT-surface median distance 2-6 mm on every scene;
- articulation targets (type, revolute direction + per-point closest point on axis, prismatic direction) from `gt/mobility_v2.json` rotated into the dataset frame exactly like `eval.py` does;
- the sidecar json tells `PAT/init_deform_PAT.py` to use the `pat_vlm` bridge (PAT part -> `joint_infos.json` slot by VLM type, centre / dist_max / axis / origin from PAT).

Eval on TEST_SCENES (100481 101284 103811 45194 47648), 4096 points, slots Hungarian-matched to GT labels, axis angle sign-invariant, origin error = distance between axis lines in world units (m):

```
[Eval BEFORE (zero-shot) [test scenes are in train]]  scene | joint | type | axis angle err (deg) | rev origin err (world)
    100481 |  1  |  r  |      0.09  |  0.0163
    100481 |  2  |  r  |      0.03  |  0.0125
    101284 |  1  |  r  |      0.04  |  0.0281
    101284 |  2  |  r  |      0.06  |  0.0061
    103811 |  1  |  p  |     62.43  |     --
    103811 |  2  |  p  |     74.60  |     --
    103811 |  3  |  p  |      6.76  |     --
    103811 |  4  |  p  |     21.23  |     --
    103811 |  5  |  p  |     23.49  |     --
    103811 |  6  |  p  |     72.77  |     --
     45194 |  1  |  r  |      0.04  |  0.0128
     45194 |  2  |  r  |      0.03  |  0.0489
     45194 |  3  |  p  |      0.16  |     --
     45194 |  4  |  p  |      0.10  |     --
     47648 |  1  |  r  |      0.02  |  0.0282
     47648 |  2  |  r  |      0.07  |  0.0559
     47648 |  3  |  r  |      0.04  |  0.0637
     47648 |  4  |  r  |      0.05  |  0.0715
     47648 |  5  |  p  |      0.24  |     --
     47648 |  6  |  p  |      0.06  |     --
  mean axis angle error: 13.11 deg over 20 joints; mean point-label accuracy (matched slots): 0.915

[Eval AFTER (LoRA-FT)]  scene | joint | type | axis angle err (deg) | rev origin err (world)
    100481 |  1  |  r  |      0.03  |  0.0094
    100481 |  2  |  r  |      0.07  |  0.0146
    101284 |  1  |  r  |      0.13  |  0.0030
    101284 |  2  |  r  |      0.37  |  0.0096
    103811 |  1  |  p  |      0.02  |     --
    103811 |  2  |  p  |      0.03  |     --
    103811 |  3  |  p  |      0.02  |     --
    103811 |  4  |  p  |      0.00  |     --
    103811 |  5  |  p  |      0.09  |     --
    103811 |  6  |  p  |      0.04  |     --
     45194 |  1  |  r  |      0.02  |  0.0072
     45194 |  2  |  r  |      0.02  |  0.0036
     45194 |  3  |  p  |      0.00  |     --
     45194 |  4  |  p  |      0.02  |     --
     47648 |  1  |  r  |      0.03  |  0.0111
     47648 |  2  |  r  |      0.02  |  0.0140
     47648 |  3  |  r  |      0.02  |  0.0077
     47648 |  4  |  r  |      0.04  |  0.0032
     47648 |  5  |  p  |      0.04  |     --
     47648 |  6  |  p  |      0.00  |     --
  mean axis angle error: 0.05 deg over 20 joints; mean point-label accuracy (matched slots): 0.997
```

Loss trajectory (per-epoch means):
```
[FT] epoch   0/120  point_mask 0.555  dice 0.100  motion_hierarchy 0.000  classification 0.017  axis_revolute 0.208  axis_prismatic 0.029  point_closest_point_on_axis 0.014  (0.0 min)
[FT] epoch  12/120  point_mask 0.031  dice 0.021  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.183  axis_prismatic 0.006  point_closest_point_on_axis 0.012  (0.4 min)
[FT] epoch  24/120  point_mask 0.015  dice 0.012  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.157  axis_prismatic 0.006  point_closest_point_on_axis 0.009  (0.8 min)
[FT] epoch  36/120  point_mask 0.012  dice 0.009  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.134  axis_prismatic 0.006  point_closest_point_on_axis 0.010  (1.2 min)
[FT] epoch  48/120  point_mask 0.011  dice 0.009  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.128  axis_prismatic 0.006  point_closest_point_on_axis 0.008  (1.5 min)
[FT] epoch  60/120  point_mask 0.010  dice 0.009  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.134  axis_prismatic 0.006  point_closest_point_on_axis 0.008  (1.9 min)
[FT] epoch  72/120  point_mask 0.009  dice 0.007  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.110  axis_prismatic 0.006  point_closest_point_on_axis 0.008  (2.3 min)
[FT] epoch  84/120  point_mask 0.009  dice 0.007  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.098  axis_prismatic 0.006  point_closest_point_on_axis 0.007  (2.7 min)
[FT] epoch  96/120  point_mask 0.007  dice 0.005  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.110  axis_prismatic 0.006  point_closest_point_on_axis 0.006  (3.1 min)
[FT] epoch 108/120  point_mask 0.008  dice 0.006  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.100  axis_prismatic 0.006  point_closest_point_on_axis 0.006  (3.4 min)
[FT] epoch 119/120  point_mask 0.010  dice 0.007  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.104  axis_prismatic 0.006  point_closest_point_on_axis 0.006  (3.8 min)
```

Notes
- Segmentation (point_mask + dice) and classification losses reach ~0.01 / 0.000; prismatic-axis L1 0.006; the per-point closest-point loss 0.006 (normalised units).
- The revolute-axis L1 plateaus at ~0.10 although the sign-invariant eval angle is 0.05 deg: the released loss is a signed L1 on the direction, so joints whose GT direction points opposite to the model's preferred sign each contribute ~2/3. This is irrelevant downstream: `ArticulationModel` learns the signed angle, so only the axis line matters.
- Compared to PAT_3 (motion-analysis `joint_infos.json` as targets): 1.54 deg -> 0.05 deg mean axis error, 0.984 -> 0.997 matched-slot label accuracy on the same 5 scenes; revolute origin errors 0.3-1.5 cm (PAT_3: 0.7-6 cm).

Full log: logs/PAT_finetune_PAT_4.txt; label dumps: particulate/model_ckpt/finetune_PAT_4/gt_labels_<scene>.ply; metrics.json in the same folder; sidecar particulate/model_ckpt/trained_PAT_gt_model.json.
