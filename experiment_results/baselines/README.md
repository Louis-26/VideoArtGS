# Baseline reproduction on VideoArtGS-20

This records what was run, and what was changed from the released code and why,
so every number in the tables can be defended.

Scope: **VideoArtGS-20 only**. All three baselines are re-run here, including
RSRD -- in the VideoArtGS paper RSRD's numbers are quoted from
Video2Articulation (marked with a dagger) and Table 2 drops it entirely, so
these are measured numbers where the paper had none.

Everything below lives under `baselines/`; the drivers are
`baselines/run_v2a_pipeline.sh` and `baselines/chain_rsrd_local.sh`. Results and
the per-method commentary are in `videoartgs20_comparison.md`.

## How the three baselines were run

All are scored with **this repo's own metric**, not with their own evaluation
code, so every row in the table comes from one implementation. That metric is
`eval.py` for ArticulateAnything, Video2Articulation and the PAT ablation; for
RSRD, and for the strict re-scoring of all four, it is `tools/eval_cpu.py`,
which reuses `eval.py`'s own code for everything but the Chamfer distance and
agrees with it to 0.03% median (see "Evaluation on CPU").
A method only has to emit, per scene,
`<out>/videoartgs/sapien/<scene>/final/train/ours_20000/` containing
`joint_info.json` and `meshes/{whole_mesh,part_0,part_1,...}.ply`
(`part_0` = static base). Note `compute_recon_error` samples 10k points from
mesh *faces*, so point clouds are not a valid substitute.

### Video2Articulation (iTACO, 3DV 2026)

It models a single movable part. The VideoArtGS paper extends it by isolating
video segments in which only one part moves. VideoArtGS-20 makes this exact:
every scene is 150 static multi-view frames followed by one smooth
close/open/close cycle per part, each in its own disjoint ~40-frame window
(part *k* moves over frames 159+50k .. 199+50k). `tools/prep_v2a_subvideos.py`
cuts the resulting **67 single-part sub-videos** out of the 20 scenes.

Per sub-video: MonST3R moving map -> AutoSeg-SAM2 part segmentation -> coarse
prediction -> refinement for both joint types. The joint type is then chosen by
Video2Articulation's own rule (`evaluate.py`: revolute if its refined axis passes
within 0.15 of the surface and its loss is lower, else prismatic). The per-part
moving maps are merged onto the shared canonical surface to produce one mesh per
part, which is the "merge the moving map to extract multiple part meshes" step
the paper describes.

### RSRD / 4D-DPM (Kerr et al. 2024)

RSRD builds an object model from a multi-view scan -- a GARField grouping field
plus a DINO-embedded gaussian model (DiG) -- then tracks a demonstration video
against it. VideoArtGS-20 supplies both: frames 0-149 are a static multi-view
block in the canonical pose, and the rest is the articulation.

**The interactive step.** The released pipeline requires a person to open the
DiG viewer, click a point on the object, crop to it, choose a group level and
press "Cluster Scene" before tracking can start. `scripts/run_tracker_headless.py`
replaces both halves programmatically.

*The crop* is a box carved from the alpha channel of the same static frames the
scan was built from (`tools/rsrd_object_crop.py`): a voxel survives if it
projects inside the object silhouette in every view that sees it, and the
gaussians inside the resulting box are kept. Because the cameras sit in a ring,
silhouettes alone cannot bound the object vertically, so a voxel must also be in
frame in 90% of views -- every camera here points at the object. The carved boxes
land within 0.1 of the ground-truth object extent on all 20 scenes. Cropping then
subsets every entry of `model.gauss_params` by that index set, which is exactly
what the viewer's `_update_crop_vis` does once the user has dragged the box. This
uses only RSRD's own input; the renders are RGBA and every baseline reads them.

Skipping the crop is not an option, though we first assumed it was on the grounds
that VideoArtGS-20 renders the object alone. It does, but on a blank background,
and splatfacto answers a blank background by scattering gaussians through it:
across the 20 scenes only 0.1-13% of gaussians land on the object and the rest
spread out to 40 m, forty times the object's size. Clustering the uncropped scene
therefore returns the same two groups at every group scale -- the object and
everything else -- and the meshes read out of it are background, which puts every
Chamfer distance in the hundreds of metres. After cropping, the group scale
becomes informative again (4/4/3/2 clusters over the sweep on 100481, where
before it was 2 everywhere).

*The clustering* is `_cluster_scene` with the viewer calls removed, swept over the
group scale and kept at the scale whose cluster count is closest to the object's
part count *plus one*, since a usable segmentation needs the base the joints are
measured against as well as the movable parts. That is the judgement the README
asks a person to make by eye, made reproducibly. Everything downstream is RSRD's
own code.

**Scan quality is the ceiling here, not the tracker.** The crop makes the numbers
meaningful but cannot add gaussians that were never placed on the object: six
scenes end up with fewer than 500 object gaussians (9016 has one), which is too
few for HDBSCAN to split into parts at any scale. The cause is upstream of RSRD
-- compositing the renders onto flat white, as nerfstudio's blender convention
does, leaves splatfacto no reason to prefer the object over the background. Giving
RSRD masked or textured-background inputs would need GARField and DiG retrained
for all 20 scenes and was out of budget; it is the first thing to change if these
numbers are ever revisited.

**Reading out an articulation.** RSRD predicts no joints; it returns an SE(3)
delta per part per frame. The joint is recovered from that motion: take the frame
where a part moved furthest from rest and decompose the transform into a screw
axis. Negligible rotation means prismatic, which is how Video2Articulation makes
the same call.

**The released version pins do not work.** RSRD's README asks for nerfstudio
1.1.4 with gsplat 1.4.0; that combination cannot import (nerfstudio 1.1.4 needs
`gsplat.cuda_legacy`, dropped in gsplat 1.4.0), and `dig.py` calls
`self.strategy`, which only exists from nerfstudio 1.1.5. The combination that
runs is **nerfstudio 1.1.5 + gsplat 1.4.0 + numpy 1.26.4 + cupy 13.3.0**;
transformers must stay below the release that requires torch >= 2.5,
tiny-cuda-nn has to be built separately, and cuML 24.10 loads CUDA 11 libraries
that ship only inside the `nvidia-*-cu11` wheels.

**Two scenes cannot be built at all.** Seven scenes initially failed to cluster.
The cause is not the clustering: GARField's training diverges, leaving NaN in
`grouping_field.enc_list.{0,1}.params` and `instance_net.params`, so every
grouping feature is non-finite and HDBSCAN is handed an empty matrix. Re-running
GARField under a different seed fixes it for five of them (101808, 25493, 8961 at
seed 1; 101908, 47648 in a sweep over seeds 2-7). **103811 and 31249 diverge at
every seed tried**, so RSRD produces no model for them; they are scored at the
failure default, as the evaluation protocol prescribes. 103811 has 30 movable
parts and 31249 is a drawer unit whose parts are near-identical, which is the
kind of geometry GARField's contrastive grouping loss is least stable on.

**A static part has to be reserved.** RSRD labels nothing as the base. We call
the cluster that moved least the static part and the next `n_gt` the movable
ones; when a scene clusters into exactly `n_gt` groups, taking all of them as
movable would leave no base and every Chamfer distance would fall back to the
failure value.

One problem is specific to a shared cluster: gsplat JIT-compiles its CUDA
extension and re-checks the build on every run, so concurrent jobs sharing
`~/.cache/torch_extensions` over NFS lose the lock file and all die. The
extension cache has to sit on node-local disk.

### Evaluation on CPU

Partway through the RSRD runs the node's NVIDIA driver wedged: `nvidia-smi` stopped
responding and every process touching CUDA -- including `import torch` and
`import open3d`, which probes the driver at import -- blocked in an unkillable
D state. The account had expired on the other nodes, so `eval.py` could not be
run anywhere for the final RSRD scoring.

`tools/eval_cpu.py` computes the same numbers without CUDA. It does not
reimplement the protocol: the pieces that decide *what* is compared -- GT joint
parsing, the prediction reader, the axis/position metric, the permutation search
and the 100 / 90 failure defaults -- are lifted out of `utils/metrics.py` with
`ast` and executed unchanged, so they cannot drift. Only the Chamfer distance is
substituted: `sample_points_from_meshes` -> `trimesh.sample.sample_surface` (both
uniform by triangle area) and `chamfer_distance(single_directional=False)` -> the
same quantity via `scipy.spatial.cKDTree`.

Re-scoring ArticulateAnything, whose official `result.csv` files predate the
driver failure, agrees to **0.03% median, 2.85% worst case over 39 metric
values** -- the residual is Monte-Carlo noise in the 10k-point sampling, and the
deterministic axis and position metrics match exactly.

Meshing moved to CPU for the same reason: `tools/mesh_cpu.py` builds the surface
with an occupancy grid and marching cubes instead of Open3D's Poisson. It
recovers a unit sphere's extent to 0.7%, and the metric only samples points from
faces.

**Unpredicted joints.** When a method cannot predict a part, the assemblers still
write a joint entry so `joint_info.json` and `part_k.ply` stay aligned, flagged
`not_predicted` with a default +Z axis. `load_joint_infos` has no notion of that
flag, so those defaults are scored as real predictions, and one that happens to
line up with the GT scores 0 deg. The tables below use `--strict`, which drops
them so the part goes through the 90 deg / 100 cm branch the protocol already
applies to missing joints. This is applied to every method equally.

### ArticulateAnything (ICLR 2025)

Run per scene on the full dynamic video with `gpt-4o`, in its multi-joint mode
(`joint_actor.targetted_affordance=false`), plus a per-part variant on the same
sub-videos Video2Articulation gets. It retrieves a PartNet-Mobility object and
articulates it, so its URDF is in the retrieved object's frame; predictions are
brought into the VideoArtGS world frame by a scale-aware registration seeded
with all 24 axis-aligned rotations and refined by ICP. Predicted parts are
ordered by surface area because `eval.py` scores the first K predicted parts.

## Deliberate deviations from the released code

| What | Why |
|---|---|
| Open3D Poisson instead of NKSR for mesh extraction | NKSR is uninstallable: `nksr.huangjh.tech` no longer resolves and PyPI carries only a 1.2 kB placeholder. |
| MonST3R/AutoSeg-SAM2 taken from Video2Articulation's *pinned forks* (`willipwk/*`) | Upstream has different CLI flags and output names; the fork is what its loader expects. |
| `--mask_type monst3r` (not `gt`) | The `gt` path crashes in `joint_refinement.py` (`best_moving_vectors` is only defined on the monst3r branch), so monst3r is the configuration the paper used. |
| ArticulateAnything: `conda run -n articulate-anything` replaced by an absolute interpreter path | `conda run` resolves to the wrong interpreter on this cluster. |
| ArticulateAnything: retrieval skips 0-byte template renders | 129 of the 2347 released `robot_frontview.png` are empty and crash mesh retrieval; re-rendering them rewrites the URDFs in place, which is riskier than dropping 3% of retrieval candidates. |
| ICP alignment is given 24 seeds and its best fit is kept | Deliberately generous to the baseline: any residual misalignment should be the method's error, not the harness's. |
| RSRD: the viewer's click-and-crop replaced by a silhouette-carved box (`tools/rsrd_object_crop.py`) | The released step is interactive and cannot run over 20 scenes. Skipping it is not an option: ~90% of the gaussians are background floaters and GARField groups those instead of the object. The carved boxes match GT object extent to within 0.1 on all 20 scenes. |
| RSRD: the group-scale sweep targets `n_parts + 1` clusters | A usable segmentation needs the base the joints are measured against as well as the movable parts. |
| RSRD: only clusters surviving the crop are eligible to be base or movable | A cluster that is entirely background has no points inside the object box; naming it the base writes an empty `part_0.ply` and scores CD-s at the failure default instead of measuring anything. |
| RSRD scored with `tools/eval_cpu.py`, meshed with marching cubes | The node's CUDA driver wedged mid-run and no GPU was reachable afterwards; `import torch` and `import open3d` both block. The substitution reuses the official metric code except for the Chamfer distance and agrees to 0.03% median. |
| Unpredicted joints scored at the 90 deg / 100 cm default, not at their placeholder axis | `load_joint_infos` has no notion of the `not_predicted` flag, so a default +Z axis that happens to match the GT would score 0 deg. Applied to every method equally. |

## Results

See `videoartgs20_comparison.md` (regenerate with
`python baselines/tools/summarize_baselines.py --methods ...`) and
`ablation_pat_inputs.md` (`python baselines/tools/summarize_ablation.py`).
Qualitative rows are in `baselines/qualitative/<scene>.png`.
