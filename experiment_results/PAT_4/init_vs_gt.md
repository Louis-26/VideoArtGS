# Deform-field init quality: PAT (GT-supervised) bridge vs. motion-analysis `joint_infos.json`

Per scene, the moving slots written by `PAT/init_deform_PAT.py --bridge pat_vlm` (`outputs_PAT_4/videoartgs/sapien/<scene>/init/joint_infos_pat.json`) and the original `data/videoartgs/sapien/<scene>/joint_infos.json` are both matched to `gt/mobility_v2.json` with `utils.metrics.find_eval_perm_axis` (the same matching eval.py uses). Axis error in degrees (sign-invariant), revolute origin error = distance between the axis lines in cm. Values are the mean over the moving joints of the scene.

All 20 scenes keep the slot list of `joint_infos.json` exactly (count, VLM joint types, order); every moving slot received a type-compatible PAT part (no motion-analysis fallback was needed). Where two slots share a type the assignment order can differ from the motion-analysis order (e.g. 100481, 101287: the two revolute doors are swapped), which is harmless because each slot carries its own centre / axis / origin.

| scene | slots | axis PAT init | axis joint_infos | origin PAT init | origin joint_infos |
|---|---|---|---|---|---|
| 100481 | 3 | 0.04 | 5.86 | 0.22 | 1.51 |
| 101284 | 3 | 0.26 | 0.84 | 0.36 | 1.68 |
| 101287 | 3 | 0.20 | 1.40 | 0.45 | 0.21 |
| 101808 | 3 | 2.73 | 7.72 | 0.58 | 0.99 |
| 101908 | 4 | 0.03 | 1.42 | 0.39 | 1.14 |
| 103015 | 4 | 0.05 | 5.82 | 0.43 | 1.28 |
| 103811 | 7 | 0.03 | 0.57 | 0.00 | 0.00 |
| 10489 | 3 | 0.02 | 0.92 | 0.45 | 0.10 |
| 10655 | 3 | 0.04 | 1.69 | 0.98 | 0.34 |
| 1280 | 3 | 0.10 | 4.84 | 0.17 | 1.23 |
| 168 | 3 | 0.06 | 1.64 | 0.21 | 1.32 |
| 25493 | 4 | 0.03 | 0.37 | 0.00 | 0.00 |
| 30666 | 10 | 0.04 | 0.91 | 0.00 | 0.00 |
| 31249 | 5 | 0.02 | 1.08 | 0.02 | 0.16 |
| 45194 | 5 | 0.02 | 0.68 | 0.16 | 0.21 |
| 45503 | 4 | 0.03 | 1.23 | 0.23 | 0.10 |
| 45612 | 7 | 0.04 | 1.91 | 0.67 | 0.25 |
| 47648 | 7 | 0.03 | 2.84 | 0.20 | 0.19 |
| 8961 | 3 | 0.37 | 0.91 | 0.91 | 0.38 |
| 9016 | 3 | 0.48 | 1.34 | 0.73 | 0.72 |
| **mean / median** | | **0.23 / 0.04** | 2.20 / 1.37 | **0.36 / 0.29** | 0.59 / 0.30 |

Worst PAT init: 101808 (3.7 deg and 1.7 deg on its two revolute joints; joint_infos: 4.9 / 10.5 deg). Everything else is below 0.5 deg and below 1 cm.

Raw per-joint output of the check script (`check_bridge.py`, scratch) is in `bridge_check_all.txt` next to this file.
