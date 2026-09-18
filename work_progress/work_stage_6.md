# tasks
1. collect and compare the baselines (2026-09-16)

- Videoartgs baselines reproduction(qualitative, quantitative)

  - ✅ articulateanything — all 20 scenes run with gpt-4o, scored with this repo's
    `eval.py`; see [baselines README](../experiment_results/baselines/README.md)
  - ⏭️ RSRD — **not reproduced, cited instead.** In the VideoArtGS paper RSRD's
    numbers are themselves taken from Video2Articulation (marked †) and Table 2
    drops RSRD entirely for failing to segment parts; running it also needs
    per-scene manual segmentation in a browser viewer.
  - 🔄 video2articulation — 67 single-part sub-videos (one per movable part),
    MonST3R + AutoSeg-SAM2 preprocessing, coarse prediction, refinement per
    joint type, then merged back per scene
- Videoartgs+PAT(already finished✅)

2. read [sample paper](../reference_paper/README.md) for paper writing skills 

3. finish Methodology + Experiment in paper
4. Ablation study

   - ✅ vs videoartgs — reproduced VideoArtGS matches the paper (0.34° / 0.10)
   - ✅ transformer architecture — PAT extra-input ablation over all 8 subsets of
     `{track_geo, track_tapip, vggt}`, see
     [ablation table](../experiment_results/baselines/ablation_pat_inputs.md).
     Each modality alone lowers axis error (1.76 -> 1.32° for `track_tapip`);
     combinations do not compound, they trade axis accuracy for segmentation
     accuracy (0.962 -> 0.984 point-label).
