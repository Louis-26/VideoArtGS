#!/bin/bash
# End-to-end chain for the "PAT + VGGT/TAPIP3D input features" experiment (outputs_PAT_3):
#   1. wait until the per-scene extra features exist
#        data_tools/extract_tapip3d_feats.py -> <scene>/pat_extra/tapip3d_feats.npz
#        data_tools/extract_vggt_feats.py    -> <scene>/pat_extra/vggt128.npy
#   2. fine-tune PAT on all 20 videoartgs-sapien scenes (train = test, on purpose)
#        -> particulate/model_ckpt/trained_PAT_model.pt (+ .json sidecar)
#   3. run the regular VideoArtGS+PAT pipeline with that checkpoint into outputs_PAT_3
#        -> experiment_results/PAT_3/videoartgs_sapien_results.txt
# Usage (from any node with GPUs; ALLOWED_GPUS restricts gpu_utils.sh on shared nodes):
#   ALLOWED_GPUS=0,2,4,5,6 TRAIN_GPU=5 bash scripts/videoartgs_pat3_chain.sh
set -o pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
source /research/cvl-ylu174/Anaconda3/etc/profile.d/conda.sh
conda activate videoartgs

export ALLOWED_GPUS=${ALLOWED_GPUS:-"0,2,4,5,6"}
TRAIN_GPU=${TRAIN_GPU:-5}
EPOCHS=${EPOCHS:-150}
EXTRA=${EXTRA:-"track_geo,track_tapip,vggt"}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_PAT_3}
SAVE_DIR=${SAVE_DIR:-PAT_3}
CKPT=particulate/model_ckpt/trained_PAT_model.pt
mkdir -p logs
echo "🚀 chain started $(date) on $(hostname); ALLOWED_GPUS=$ALLOWED_GPUS TRAIN_GPU=$TRAIN_GPU EPOCHS=$EPOCHS EXTRA=$EXTRA"

# ---- 1. wait for the feature extraction jobs ----
while true; do
    n=$(ls data/videoartgs/sapien/*/pat_extra/tapip3d_feats.npz 2>/dev/null | wc -l)
    m=$(ls data/videoartgs/sapien/*/pat_extra/vggt128.npy 2>/dev/null | wc -l)
    if [ "$n" -ge 20 ] && [ "$m" -ge 20 ]; then break; fi
    echo "⏳ waiting for features: tapip3d $n/20, vggt $m/20 ($(date))"
    sleep 60
done
echo "✅ features ready $(date)"

# ---- 2. fine-tune PAT ----
if [ ! -f "$CKPT" ] || [ "${RETRAIN:-0}" = "1" ]; then
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -u PAT/PAT_finetune.py \
        --epochs "$EPOCHS" \
        --extra_feats "$EXTRA" \
        --labels track \
        --train_on_all \
        --extra_dropout 0.2 \
        --save_name trained_PAT_model.pt \
        --out_dir particulate/model_ckpt/finetune_PAT3 2>&1 | tee logs/PAT_finetune_PAT3.txt
    if [ ! -f "$CKPT" ]; then echo "❌ fine-tuning did not produce $CKPT"; exit 1; fi
else
    echo "⏭️ $CKPT exists, skipping fine-tuning"
fi
echo "✅ PAT fine-tuned $(date)"

# ---- 3. VideoArtGS + PAT pipeline ----
bash scripts/videoartgs_pat_pipeline.sh \
    --use_multi 1 \
    --keep_logs 1 \
    --mode 1 \
    --output_dir "$OUTPUT_DIR" \
    --save_dir "$SAVE_DIR" \
    --PAT_model_pth "$CKPT" 2>&1 | tee logs/pipeline_PAT3.txt
echo "🎉 chain finished $(date)"
touch logs/PAT3_chain_done
