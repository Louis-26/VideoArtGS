# Reproducing the four methods on VideoArtGS-20

What was run, in what order, and every trap that cost real time. Written so that
someone with the repo and a GPU node can re-run any of the four end to end.

| Method | Role | Env | Cost for 20 scenes |
|---|---|---|---|
| ArticulateAnything (ICLR 2025) | baseline | `aa` (py3.9) | ~2 h, OpenAI API |
| Video2Articulation / iTACO (3DV 2026) | baseline | `videoartgs` + `monst3r` + `autoseg` | ~6 h on 12 GPUs |
| RSRD / 4D-DPM (Kerr 2024) | baseline | `rsrd` (py3.10) | ~3 GPU-h per scene |
| VideoArtGS + PAT | ours | `videoartgs` | ~40 min per scene |

Results are in `experiment_results/baselines/videoartgs20_comparison.md`; the
justification for every deviation is in `experiment_results/baselines/README.md`.

---

## 0. Things that bite regardless of which method you run

Read this section first. Most of the time lost in this reproduction was lost
here, not in the methods.

### `conda activate` silently does not change the interpreter

On this cluster the activated env inherits `PATH` from the parent and `python`
stays whatever it was. Every job in this repo therefore calls the interpreter by
absolute path:

```bash
/research/cvl-ylu174/Anaconda3/envs/<env>/bin/python ...
```

Never `conda activate <env> && python`, and never `conda run -n <env>` —
ArticulateAnything shells out with `conda run` internally and had to be patched
for exactly this reason.

### Cap the thread count or the GPUs sit idle

Numpy/torch each grab all 256 cores by default. Running 12 jobs that way drove
load average to 683–917 with GPU utilisation at **0%** — every process was
fighting for cores instead of feeding the card. MonST3R went from 25 min to
2.5 min per sub-video once capped. `tools/gpu_pool.py --threads 8` exports

```
OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
```

If you dispatch jobs by hand, export those yourself.

### Put the torch extension cache on local disk

gsplat JIT-compiles a CUDA extension and re-checks the build on every run.
Concurrent jobs sharing `~/.cache/torch_extensions` over NFS lose the lock file
and **all** die. Always:

```bash
export TORCH_EXTENSIONS_DIR=/tmp/$USER_torchext
```

### `pkill -f '<pattern>'` kills your own shell

The pattern matches the invoking command line, so the shell dies with exit 144
before killing anything. Use a bracket (`dem[o].py`) or, better, collect PIDs
first and `xargs kill` them — and remember your own `bash -c` line still contains
the literal you are grepping for if it appears anywhere in the command.

### Gate pipeline stages on artifacts, not on log strings

NFS makes log writes unreliable and a pool can exit early leaving "ok" lines that
do not correspond to real outputs. `run_rsrd_stages.sh` waits on
`ls <glob> | wc -l` reaching the scene count, never on a string in a log. Copy
that pattern.

### The evaluation contract

Any method only has to emit, per scene:

```
<out>/videoartgs/sapien/<scene>/final/train/ours_20000/
├── joint_info.json
└── meshes/{whole_mesh,part_0,part_1,...}.ply     # part_0 is the static base
```

`utils/metrics.compute_recon_error` samples 10k points **from mesh faces**, so a
point cloud saved as `.ply` scores as a failure (100) even though it loads fine.
Every assembler in `baselines/tools/assemble_*.py` exists to produce exactly this
layout.

### If the node's GPU driver wedges

Killing many CUDA jobs at once can leave the driver unresponsive: `nvidia-smi`
hangs forever and the processes sit in **D state**, which `kill -9` cannot clear.
It does not necessarily recover, and resetting a GPU needs root. The nasty part
is that this is not confined to GPU work — `import torch` and `import open3d`
both block, because Open3D probes the driver at import. `CUDA_VISIBLE_DEVICES=""`
does not help; the library still opens `/dev/nvidia*`.

numpy, cv2, trimesh, scipy and skimage do **not** touch the driver. That is
enough to keep working:

* `tools/mesh_cpu.py` — surface meshing by occupancy grid + marching cubes
  instead of Open3D Poisson (recovers a unit sphere's extent to 0.7%);
* `tools/eval_cpu.py` — the metric without CUDA. It does not reimplement the
  protocol: GT parsing, the prediction reader, the axis/position metric, the
  permutation search and the 100/90 failure defaults are lifted out of
  `utils/metrics.py` with `ast` and executed unchanged. Only the Chamfer
  distance is substituted (`trimesh.sample.sample_surface` +
  `scipy.spatial.cKDTree`). Validated against official pre-failure results at
  **0.03% median, 2.85% worst case** over 39 metric values.

```bash
python baselines/tools/eval_cpu.py --out-root <outputs>/videoartgs/sapien --validate
```

### Scoring joints a method could not predict

The assemblers write a placeholder joint entry (default +Z axis, flagged
`not_predicted`) so `joint_info.json` stays aligned with `part_k.ply`.
`utils/metrics.load_joint_infos` has no notion of that flag, so those defaults
are scored as real predictions — and one that happens to line up with the GT
scores **0 deg**, flattering whichever method failed most. Use `--strict`:

```bash
python baselines/tools/eval_cpu.py --out-root <outputs>/videoartgs/sapien \
    --csv-name result_strict.csv --strict
```

which drops them so the part goes through the protocol's own 90 deg / 100 cm
branch. Apply it to every method or to none. Evidence it is the right reading:
it moved ArticulateAnything's axis error from 33.24 to 38.49, *closer* to its
published 43.65.

---

## 1. The dataset fact everything depends on

Each VideoArtGS-20 scene is **150 static multi-view frames**, then one smooth
close/open/close cycle per movable part, each part in its own disjoint ~40-frame
window: part *k* moves over frames `159+50k .. 199+50k`. 20 scenes, 67 movable
parts total.

That structure is what makes the paper's "isolate segments where only a single
part moves" reproducible, and it is what the per-part baselines consume:

```bash
python baselines/tools/prep_v2a_subvideos.py     # 67 single-part sub-videos
python baselines/tools/prep_gt_part_masks.py     # per-pixel GT part labels
```

Renders are RGBA with the **alpha channel as the object mask**, and
`depth/*.png` is `uint16` millimetres (divide by 1e3). Both facts matter below.

---

## 2. Video2Articulation (iTACO)

Pipeline: MonST3R moving map → AutoSeg-SAM2 part segmentation → coarse
prediction (LoFTR + RANSAC) → per-joint-type refinement → merge per scene.

### Setup

```bash
bash baselines/setup_v2a_env.sh
bash baselines/setup_monst3r.sh
bash baselines/setup_autoseg.sh
```

**Use the pinned forks.** V2A's submodules point at `willipwk/monst3r` and
`willipwk/AutoSeg-SAM2`, not upstream. Upstream has different CLI flags and
output filenames and its loader will not find anything.

**SEA-RAFT weights only come down via `gdown <file-id>`.** `--fuzzy` is
unsupported by the installed gdown, and the HuggingFace mirror is a different
variant that produces different flow.

**NKSR is dead.** V2A's mesh backend needs `nksr.huangjh.tech`, which no longer
resolves; PyPI carries a 1.2 kB placeholder. Replaced with Open3D Poisson (and
now `mesh_cpu.py` when no GPU is available).

### The background problem — worth an hour of anyone's time

VideoArtGS-20 renders objects on a **transparent** background. Composite that
onto a flat colour and MonST3R breaks: it solves camera motion before it can
call anything dynamic, and with no static structure in view it labels
**50–93% of every frame dynamic** where the GT moving part is 0.2–1.6%. 26 of
63 sub-videos then produced no coarse prediction at all.

Fix — render a world-fixed textured environment sphere behind the object:

```bash
python baselines/tools/add_env_background.py     # RADIUS=6, cameras orbit at ~4
```

Coarse-stage success went **37/63 → 57/65**. A ground plane does not work: the
cameras orbit through negative z and would be occluded. V2A's own dataset does
the same thing with a wall and ground (the `_bg` in its scene names).

**Only MonST3R needs the environment.** AutoSeg-SAM2 runs on the *object-only*
frames: its output is masked to the object anyway, and the background texture
makes SAM propose far more masks (5× slower).

### Resolution

The loader (`baselines/video2articulation/vartgs_data.py`) renders everything at
**512×512** (`VARTGS_RES`), not the native 800×800. At 800 the refinement wants
~90 GB and will not fit an 80 GB card. 512×512 is 262k pixels against the 307k
of V2A's own sim data, so this matches the method's scale rather than shrinking
below it — ~19 s coarse, ~4 min for 400 refinement steps, 18 GB.

Everything the loader hands out must be at that one resolution. Mixing
full-resolution depth with scaled intrinsics silently puts the surface in the
wrong place, and **AutoSeg masks must be resized before being intersected with
the object mask**, not after.

### Run

```bash
bash baselines/run_v2a_pipeline.sh
```

which is coarse → refine → `assemble_v2a_results.py` → eval → summary. Use
`--mask_type monst3r`; the `gt` path crashes in `joint_refinement.py` because
`best_moving_vectors` is only defined on the monst3r branch, so monst3r is the
configuration the paper used.

---

## 3. ArticulateAnything

Pipeline: VLM retrieves a PartNet-Mobility template from the video (gpt-4o),
emits a URDF, which is then registered into the VideoArtGS world frame.

### Setup

```bash
bash baselines/setup_aa.sh
python baselines/tools/make_part_videos.py    # or make_scene_videos.py
```

The OpenAI key lives in `.env` at the repo root and is passed as the `API_KEY`
env var. Do not paste it into job files you intend to share — the generated
`jobs/aa_*.jobs` contain it inline.

### Pitfalls

**`conda run -n articulate-anything` resolves the wrong interpreter.** Patched
`make_cmd` to call the env python by absolute path.

**129 of the 2347 released `robot_frontview.png` are 0 bytes** and crash mesh
retrieval. `obj_selector.py` is patched to skip unreadable images — re-rendering
them rewrites the URDFs in place, which is riskier than dropping 3% of the
retrieval candidates.

**OpenAI rate limit (30k TPM)** forces the pool down to a single slot; more
parallelism just produces 429s.

**A failed run looks like a successful one.** `articulate.py` exits 0 even when
no URDF was written, so the job command ends with an explicit existence check on
`joint_actor/iter_*/seed_*/mobility.urdf` — without it the pool reports 20/20
success over partially empty output.

### Registration

AA articulates a *retrieved* template, so its URDF is in the retrieved object's
frame. `assemble_aa_results.py` brings predictions into the VideoArtGS world
frame with a **scale-aware registration seeded with all 24 axis-aligned
rotations, refined by ICP**, keeping the best fit. This is deliberately generous:
any residual misalignment should be the method's error, not the harness's. It is
also why our CD (1.58) is far better than the published 16.10 — the published
evaluation appears not to fit scale.

Predicted parts are ordered by surface area, because `eval.py` scores the first
K predicted parts.

```bash
bash baselines/chain_aa_part.sh    # merge + score
```

---

## 4. RSRD / 4D-DPM — the hard one

Pipeline: GARField grouping field → DiG (DINO-embedded gaussians) → clustering →
tracking → `keyframes.txt` (part_deltas `[T,G,7]`, wxyz+xyz) → screw-axis
decomposition.

### The released version pins do not work

RSRD's README asks for nerfstudio 1.1.4 with gsplat 1.4.0. That combination
**cannot import**: nerfstudio 1.1.4 needs `gsplat.cuda_legacy`, dropped in gsplat
1.4.0, and `dig.py` calls `self.strategy`, which only exists from 1.1.5. The
combination that runs:

**nerfstudio 1.1.5 + gsplat 1.4.0 + numpy 1.26.4 + cupy 13.3.0 + cuML 24.10**

plus: transformers must stay below the release requiring torch ≥ 2.5;
tiny-cuda-nn has to be built separately (copying a prebuilt
`tinycudann` + `tinycudann_bindings` from a working env is far faster than
building); cuML 24.10 loads CUDA 11 libraries that ship only inside the
`nvidia-*-cu11` wheels; moviepy 2.x removed `moviepy.editor`. `setup_rsrd.sh`
encodes all of this — twelve separate dependency problems.

### Data

```bash
python baselines/tools/prep_rsrd_data.py    # 150 static frames -> nerfstudio scan
```

This composites alpha over flat white, which is nerfstudio's blender convention
— **and is the root cause of the biggest problem below.**

### Replacing the interactive step

The released `run_tracker.py` requires a person to open the DiG viewer, click the
object, crop to it, pick a group level and press "Cluster Scene".
`rsrd/scripts/run_tracker_headless.py` does both programmatically.

**The crop is not optional — this was the single biggest mistake in this
reproduction.** We first skipped it, reasoning that VideoArtGS-20 renders the
object alone so there is no background to click away. The object *is* alone, but
on a blank background, and splatfacto answers a blank background by scattering
gaussians through it. Across the 20 scenes only **0.1–13% of each DiG model's
gaussians land on the object**; the rest spread out to 40 m, forty times the
object's size. GARField then groups the halo, so every scene returns the same
**2 clusters at every group scale** — "the object" vs "everything else" — and the
meshes read out of it are background. Every Chamfer distance came out in the
hundreds of metres (CD-w ≈ 49801 cm), which is easy to misread as a metric bug.

`tools/rsrd_object_crop.py` replaces the click with a box carved from the static
views' alpha:

* a voxel survives if it projects **inside the silhouette in every view that
  sees it**;
* and it must be **in frame in ≥90% of views** — the cameras sit in a ring, so
  silhouettes alone cannot bound the object vertically and leave an unbounded
  column above and below it.

Carved boxes match GT object extent to within 0.1 on all 20 scenes. Cropping then
subsets every entry of `model.gauss_params` by that index set, which is exactly
what the viewer's `_update_crop_vis` does. After cropping, the group scale
becomes informative again (4/4/3/2 clusters over the sweep on 100481, where
before it was 2 everywhere).

Do **not** use a per-gaussian silhouette test instead of the box — a gaussian
centre on the object's surface can project a pixel or two outside it, and a
strict per-point test empties whole scenes (two of six test scenes returned zero
points).

### Three more RSRD-specific fixes

**`allow_single_cluster=True` collapses genuinely separable groups.** RSRD sets
it; with it off the same features give 2–5 clusters. Off is what lets the method
segment at all here. (Do not report "RSRD cannot segment" without checking this
— the first diagnosis was wrong.)

**Target `n_parts + 1` clusters,** not `n_parts`: a usable segmentation needs the
base the joints are measured against as well as the movable parts.

**Only clusters that survive the crop may be base or movable.** A cluster that is
entirely background has zero points inside the object box; naming it the base
writes an empty `part_0.ply` and scores CD-s at the failure default instead of
measuring anything.

**After cropping, `compute_roi` can fail.** It renders the model and needs
accumulation > 0.8 somewhere; ~2900 gaussians render too transparent and it
raises `max(): Expected reduction dim`. Expand the carved box (`--crop-expand
0.5`) the way a person dragging it in the viewer would.

### GARField diverges on some scenes

Seven scenes initially failed to cluster. The cause is **not** the clustering:
GARField training diverges, leaving NaN in `grouping_field.enc_list.{0,1}.params`
and `instance_net.params`, so every grouping feature is non-finite and HDBSCAN is
handed an empty matrix — surfacing as a cuML RMM abort (which no exception
handler can catch; `--cpu-cluster` swaps in scikit-learn's HDBSCAN, which fails
gracefully) or as sklearn's "min_samples must be at most 0".

Re-running GARField under a different seed fixes five (101808, 25493, 8961 at
seed 1; 101908, 47648 over seeds 2–7). **103811 and 31249 diverge at every seed
tried** and are genuine RSRD failures — 103811 has 30 movable parts and 31249 is
a drawer unit whose parts are near-identical, the geometry GARField's contrastive
loss is least stable on.

```bash
bash baselines/sweep_gf_seeds.sh     # seed sweep for the failures
bash baselines/run_rsrd_stages.sh    # GARField -> DiG -> track -> merge -> eval
```

### Reading out an articulation

RSRD predicts no joints; it returns an SE(3) delta per part per frame. The joint
comes from taking the frame where a part moved furthest from rest and
decomposing the transform into a screw axis; negligible rotation means prismatic.
Poses live in the **dataparser frame**, so `dataparser_transforms.json` must be
inverted (`world = (x/scale - t) @ R`) before anything is compared to GT. Getting
this wrong is another way to produce metre-scale Chamfer distances.

### What the result means

RSRD is by a wide margin the weakest, and this run independently reaches the
conclusion the VideoArtGS paper states when it drops RSRD from Table 2 — that it
fails to segment parts. Of 20 objects: 3 produce no model, 8 collapse to a single
cluster after cropping, 9 recover 1–2 movable parts against 2–9 in GT.

**The ceiling is the scan, not the tracker.** Six scenes end up with fewer than
500 object gaussians (9016 has *one*), too few for HDBSCAN at any scale. If these
numbers are ever revisited, the first thing to change is the flat-white
compositing in `prep_rsrd_data.py` — give RSRD masked or textured-background
inputs and retrain GARField + DiG for all 20 scenes.

---

## 5. VideoArtGS + PAT (ours)

```bash
bash scripts/init_cano.sh
bash scripts/init_deform_PAT.sh
bash scripts/train_PAT.sh --mode 1 --output_dir outputs_PAT_5b1
bash scripts/eval.sh --mode 1 --output_dir outputs_PAT_5b1
```

PAT supplies part segmentation and joint initialisation. Two checkpoints with
easily confused roles: `model_objaverse` is **PartField**, `pat_model.pt` is
**PAT_B**. PAT's direction estimate is ~6× better than the baseline
initialisation but its origin is ~3× worse — the bridge replaces both, which is
why the earlier PAT_1–4 variants regressed.

The input-modality ablation (`experiment_results/baselines/ablation_pat_inputs.md`)
holds the PAT_3 recipe fixed and varies only `--extra_feats` over `track_geo`
(56), `track_tapip` (384) and `vggt` (128):

```bash
python PAT/PAT_finetune.py --epochs 120 --labels track --train_on_all \
    --extra_feats track_tapip
```

Each modality alone lowers axis error (`track_tapip` most, 1.76 → 1.32 deg), but
combining them does not compound — every pair lands back near 1.7 deg — while
point-label accuracy keeps improving (0.962 → 0.984). The extra branches buy part
assignment; past one motion modality they compete with the axis heads.

---

## 6. Putting the table together

```bash
python baselines/tools/summarize_baselines.py --csv-name result_strict.csv --methods \
    "RSRD=outputs_baseline_rsrd" \
    "Video2Articulation=outputs_baseline_v2a" \
    "ArticulateAnything=outputs_baseline_aa" \
    "VideoArtGS+PAT b1=outputs_PAT_5b1"

python baselines/tools/render_qualitative.py --methods \
    "RSRD=outputs_baseline_rsrd" "Video2Articulation=outputs_baseline_v2a" \
    "ArticulateAnything=outputs_baseline_aa" "VideoArtGS=outputs_PAT_5b1"
```

| Method | n | Axis (deg) | Position (cm) | CD-w (cm) | CD-m (cm) | CD-s (cm) |
|---|---|---|---|---|---|---|
| RSRD / 4D-DPM | 20 | 85.55 ± 6.85 | 107.68 ± 27.00 | 28.59 ± 31.14 | 91.58 ± 14.89 | 40.94 ± 36.36 |
| Video2Articulation | 20 | 57.00 ± 23.94 | 24.58 ± 15.39 | 0.56 ± 1.23 | 41.73 ± 15.87 | 7.53 ± 11.33 |
| ArticulateAnything | 20 | 38.49 ± 43.09 | 22.78 ± 30.05 | 1.58 ± 1.33 | 31.31 ± 43.66 | 3.68 ± 10.53 |
| VideoArtGS + PAT b1 | 20 | 0.33 ± 0.81 | 0.10 ± 0.10 | 0.09 ± 0.11 | 0.30 ± 0.65 | 0.24 ± 0.64 |

`render_qualitative.py` is trimesh-only, so it still runs when the CUDA driver is
unavailable.

---

## 7. If you are starting over, do these first

1. Export the thread caps and `TORCH_EXTENSIONS_DIR` before anything else
   (§0) — it is a 10× difference.
2. Render the environment sphere before running MonST3R (§2), not after.
3. For RSRD, carve the crop box and sanity-check it against the GT object extent
   **before** training anything downstream (§4). One `carve_box` call takes 3 s
   per scene and would have saved a day.
4. Check `part_0.ply` is non-empty and that mesh extents are object-scale before
   trusting any Chamfer number. A CD in the hundreds is a frame or crop bug, not
   a bad method; a CD of exactly 100.00 is the failure default, not a
   measurement.
