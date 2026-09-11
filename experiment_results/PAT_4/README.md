This time, we utilized ground truth of segmentation and articulation to further supervise the fine-tuning of PAT model

# PAT_4: GT-supervised PAT (segmentation + articulation GT) -> deform-field init via `pat_vlm` bridge (2026-09-04)

Pipeline: stage-1 canonical Gaussians copied from `outputs_PAT` (identical, PAT-independent) -> `PAT/init_deform_PAT.py` with `particulate/model_ckpt/trained_PAT_gt_model.pt` (`--bridge auto` -> `pat_vlm` from the sidecar) -> `train.py` (20k it) -> render -> eval. Command chain: `LABELS=gt JOINT_GT=gt BRIDGE=pat_vlm CKPT_NAME=trained_PAT_gt_model.pt OUTPUT_DIR=outputs_PAT_4 SAVE_DIR=PAT_4 bash scripts/videoartgs_pat3_chain.sh` (= `videoartgs_pat_pipeline.sh --use_multi 1 --keep_logs 1 --mode 1 --output_dir outputs_PAT_4 --save_dir PAT_4 --PAT_model_pth particulate/model_ckpt/trained_PAT_gt_model.pt`). Run on cvl11, GPUs 6/0/2/3 (0/2/3 shared with other users): fine-tune 4 min, init 12 min, train 2h15m, render 44 min, eval 4 min.

## What changed vs. PAT_3


|                        | PAT_3                                                                                       | PAT_4                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| PAT inputs             | xyz + normals + PartField + track_geo/track_tapip/vggt                                      | same                                                                                                                              |
| segmentation targets   | track-based pseudo labels (`--labels track`)                                                | GT part meshes`gt/part_k.ply`, surface-sampled (`--labels gt`)                                                                    |
| articulation targets   | `joint_infos.json` (motion analysis)                                                        | `gt/mobility_v2.json` via `utils.metrics.read_gt` (`--joint_gt gt`)                                                               |
| bridge to deform field | legacy: keep`joint_infos.json` centre / dist_max, replace axis + origin of the matched slot | `pat_vlm`: keep the slot list (VLM count / types / order), fill centre, dist_max (0.2 x part radius), axis, origin from PAT parts |
| checkpoint             | `trained_PAT_model.pt`                                                                      | `trained_PAT_gt_model.pt` (+ `.json` sidecar)                                                                                     |

Training: 20 scenes, train = test (deliberate upper-bound check). PAT-level eval on the 5 former test scenes: 13.11 deg -> **0.05 deg** mean axis error, label accuracy 0.915 -> **0.997**, revolute origin error 0.3-1.5 cm (`pat_finetune_metrics.md`).
