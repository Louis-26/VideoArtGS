# VideoArtGS-20: baselines reproduced in-house

20 PartNet-Mobility objects, 67 movable parts. Axis error in degrees, position and
all Chamfer distances in cm. "ours-run" rows are our reproductions; "paper" rows
are the numbers printed in the VideoArtGS paper, for reference.

An unpredicted joint is scored at the protocol's 90 deg / 100 cm default rather
than at the placeholder axis the assemblers write to keep `joint_info.json`
aligned with the meshes. See README.md, "Unpredicted joints" -- this is applied
to every method equally, and it is why ArticulateAnything's axis error here
(38.49) sits closer to the published 43.65 than our earlier lenient scoring did.

| Method | n | Axis (deg) | Position (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|
| RSRD / 4D-DPM (ours-run) | 20 | 85.55 ± 6.85 | 107.68 ± 27.00 | 28.59 ± 31.14 | 91.58 ± 14.89 | 40.94 ± 36.36 |
| Video2Articulation (ours-run) | 20 | 57.00 ± 23.94 | 24.58 ± 15.39 | 0.56 ± 1.23 | 41.73 ± 15.87 | 7.53 ± 11.33 |
| ArticulateAnything (ours-run) | 20 | 38.49 ± 43.09 | 22.78 ± 30.05 | 1.58 ± 1.33 | 31.31 ± 43.66 | 3.68 ± 10.53 |
| VideoArtGS + PAT b1 (ours-run) | 20 | 0.33 ± 0.81 | 0.10 ± 0.10 | 0.09 ± 0.11 | 0.30 ± 0.65 | 0.24 ± 0.64 |
| *ArticulateAnything (paper)* | 20 | 43.65 ± 44.72 | 15.66 ± 36.20 | 16.10 | 17.66 | 16.04 |
| *Video2Articulation (paper)* | 20 | 48.88 ± 24.18 | 37.04 ± 31.82 | 5.07 | 30.63 | 10.22 |
| *VideoArtGS (paper)* | 20 | 0.34 ± 0.80 | 0.10 ± 0.10 | 0.09 | 0.26 | 0.24 |

## Notes on each reproduction

**ArticulateAnything** reproduces close to its published axis error and much
better on Chamfer distance (1.58 vs 16.10). The gap is a mesh-scale convention:
it retrieves a PartNet-Mobility template and we register it to the observed
object with a scale-aware fit, which the published evaluation appears not to do.

**Video2Articulation** reproduces close to its published numbers on every metric
except axis error, where ours is worse (57.00 vs 48.88). Its arbitration between
revolute and prismatic is the sensitive part on these objects.

**RSRD / 4D-DPM** is by a wide margin the weakest, and our run independently
reaches the conclusion the VideoArtGS paper states when it drops RSRD from
Table 2 -- that it fails to segment parts. Of the 20 objects:

  * 3 produce no model at all (103811, 31249 diverge in GARField at every seed
    tried; 47648 lost its tracking run to the node failure below);
  * 8 collapse to a single cluster once the object is cropped out of the
    background, so no movable part can be read out and every movable part is
    scored at the failure default;
  * 9 recover 1-2 movable parts, against 2-9 in the ground truth.

The ceiling is the scan, not the tracker: only 0.1-13% of each DiG model's
gaussians land on the object, the rest spreading out to 40 m of empty
background. README.md describes the cause (flat-white compositing) and what to
change first if these numbers are revisited.

## Provenance

RSRD's final numbers were assembled and scored on CPU: the node's NVIDIA driver
wedged partway through the runs and no GPU was reachable afterwards. `eval.py`
was replaced by `tools/eval_cpu.py`, which reuses the official metric code for
everything except the Chamfer distance and agrees with the official evaluator to
0.03% median / 2.85% worst case. Open3D's Poisson meshing was replaced by
marching cubes. Both substitutions are documented in README.md.
