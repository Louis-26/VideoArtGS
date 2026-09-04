# PAT fine-tune with extra inputs (trained_PAT_model.pt), 2026-09-04

Config: LoRA r=8 on all attention Linears + extra_embeds (track_geo 56, track_tapip 384, vggt 128), labels=track, 120 epochs x 20 scenes, extra_dropout 0.2, --train_on_all (TEST_SCENES are in the training set: this is an overfitting check, not generalization).

Eval on TEST_SCENES (100481 101284 103811 45194 47648), 4096 points, slots Hungarian-matched to GT labels:

```
[Eval BEFORE (zero-shot) [test scenes are in train]]  scene | joint | type | axis angle err (deg) | 
    100481 |  1  |  r  |      9.34  |  0.0469
    100481 |  2  |  r  |      2.32  |  0.0138
    101284 |  1  |  r  |      1.14  |  0.0664
    101284 |  2  |  r  |      0.52  |  0.0223
    103811 |  1  |  p  |     74.55  |     --
    103811 |  2  |  p  |     21.40  |     --
    103811 |  3  |  p  |      7.19  |     --
    103811 |  4  |  p  |     23.42  |     --
    103811 |  5  |  p  |     72.98  |     --
    103811 |  6  |  p  |     63.71  |     --
     45194 |  1  |  r  |      0.79  |  0.0509
     45194 |  2  |  r  |      0.19  |  0.0130
     45194 |  3  |  p  |      0.19  |     --
     45194 |  4  |  p  |      1.49  |     --
     47648 |  1  |  r  |      1.73  |  0.0336
     47648 |  2  |  r  |      1.95  |  0.0747
     47648 |  3  |  r  |      1.39  |  0.0592
     47648 |  4  |  r  |      1.67  |  0.0872
     47648 |  5  |  p  |      8.94  |     --
     47648 |  6  |  p  |      1.59  |     --
  mean axis angle error: 14.82 deg over 20 joints; mean point-label accuracy (matched slots): 0.891

[Eval AFTER (LoRA-FT)]  scene | joint | type | axis angle err (deg) | rev origin err (world)
    100481 |  1  |  r  |      9.43  |  0.0422
    100481 |  2  |  r  |      2.25  |  0.0129
    101284 |  1  |  r  |      1.04  |  0.0180
    101284 |  2  |  r  |      0.38  |  0.0127
    103811 |  1  |  p  |      0.51  |     --
    103811 |  2  |  p  |      0.26  |     --
    103811 |  3  |  p  |      0.60  |     --
    103811 |  4  |  p  |      0.26  |     --
    103811 |  5  |  p  |      0.94  |     --
    103811 |  6  |  p  |      1.05  |     --
     45194 |  1  |  r  |      0.78  |  0.0082
     45194 |  2  |  r  |      0.16  |  0.0069
     45194 |  3  |  p  |      0.15  |     --
     45194 |  4  |  p  |      1.44  |     --
     47648 |  1  |  r  |      1.81  |  0.0140
     47648 |  2  |  r  |      1.83  |  0.0537
     47648 |  3  |  r  |      1.40  |  0.0111
     47648 |  4  |  r  |      1.65  |  0.0591
     47648 |  5  |  p  |      3.71  |     --
     47648 |  6  |  p  |      1.10  |     --
  mean axis angle error: 1.54 deg over 20 joints; mean point-label accuracy (matched slots): 0.984
```

Loss trajectory (per-epoch means):
```
[FT] epoch   0/120  point_mask 1.789  dice 0.264  motion_hierarchy 0.000  classification 0.058  axis_revolute 0.297  axis_prismatic 0.188  (0.0 min)
[FT] epoch   6/120  point_mask 0.267  dice 0.165  motion_hierarchy 0.000  classification 0.001  axis_revolute 0.275  axis_prismatic 0.204  (0.2 min)
[FT] epoch  12/120  point_mask 0.261  dice 0.148  motion_hierarchy 0.000  classification 0.014  axis_revolute 0.192  axis_prismatic 0.203  (0.5 min)
[FT] epoch  18/120  point_mask 0.150  dice 0.113  motion_hierarchy 0.000  classification 0.001  axis_revolute 0.133  axis_prismatic 0.160  (0.7 min)
[FT] epoch  24/120  point_mask 0.115  dice 0.100  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.070  axis_prismatic 0.004  (0.9 min)
[FT] epoch  30/120  point_mask 0.084  dice 0.078  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.068  axis_prismatic 0.009  (1.1 min)
[FT] epoch  36/120  point_mask 0.075  dice 0.065  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.052  axis_prismatic 0.004  (1.3 min)
[FT] epoch  42/120  point_mask 0.086  dice 0.077  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.050  axis_prismatic 0.004  (1.5 min)
[FT] epoch  48/120  point_mask 0.072  dice 0.059  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.049  axis_prismatic 0.003  (1.7 min)
[FT] epoch  54/120  point_mask 0.088  dice 0.077  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.059  axis_prismatic 0.004  (1.8 min)
[FT] epoch  60/120  point_mask 0.079  dice 0.067  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.060  axis_prismatic 0.004  (2.0 min)
[FT] epoch  66/120  point_mask 0.070  dice 0.062  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (2.2 min)
[FT] epoch  72/120  point_mask 0.062  dice 0.053  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.051  axis_prismatic 0.004  (2.4 min)
[FT] epoch  78/120  point_mask 0.062  dice 0.055  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.004  (2.6 min)
[FT] epoch  84/120  point_mask 0.075  dice 0.059  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (2.8 min)
[FT] epoch  90/120  point_mask 0.061  dice 0.054  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (3.0 min)
[FT] epoch  96/120  point_mask 0.055  dice 0.050  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.053  axis_prismatic 0.005  (3.2 min)
[FT] epoch 102/120  point_mask 0.060  dice 0.056  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (3.3 min)
[FT] epoch 108/120  point_mask 0.048  dice 0.050  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (3.5 min)
[FT] epoch 114/120  point_mask 0.052  dice 0.049  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (3.7 min)
[FT] epoch 119/120  point_mask 0.057  dice 0.051  motion_hierarchy 0.000  classification 0.000  axis_revolute 0.048  axis_prismatic 0.003  (3.8 min)
```

Full log: logs/PAT_finetune_PAT3.txt; label dumps: particulate/model_ckpt/finetune_PAT3/gt_labels_<scene>.ply; metrics.json in the same folder.
