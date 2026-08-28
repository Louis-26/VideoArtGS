#!/bin/bash

# ====================================================
# [Step 1: Import GPU utils]
# ====================================================
source "$(git rev-parse --show-toplevel)/scripts/gpu_utils.sh"

# ====================================================
# [Step 2: Parse terminal inputs]
# ====================================================
MODE=1
USE_MULTI=0
KEEP_LOGS=0
OUTPUT_DIR="outputs"
SAVE_DIR="orig"
MIN_MEM=2048               

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multi) USE_MULTI="$2"; shift ;;
        --keep_logs) KEEP_LOGS="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        *) echo "❌ Error: Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ====================================================
# [Step 3: Initialize environment]
# ====================================================
init_env "$USE_MULTI" "$KEEP_LOGS" "$MIN_MEM"

source $(git rev-parse --show-toplevel)/scripts/scene_set.sh
parse_mode "$MODE"

model_name=final
seed=0
iter=20000

# ====================================================
# [Step 4: Evaluating loop with proper GPU scheduling]
# ====================================================
for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    echo "========================================="
    echo "📊 Evaluating scene: ${scene}"
    model_path=${OUTPUT_DIR}/${dataset}/${subset}/${scene}/${model_name}
    
    # 🌟 Fix 1: Use eval-specific skip detection (check if result.csv exists)
    if [ -f "${model_path}/train/ours_${iter}/result.csv" ]; then
        echo "⏭️ [SKIP] Scene ${scene} already evaluated."
        continue
    fi
    
    # Precise GPU indexing using loop index 'i'
    GPU_IDX=${GPUS[$((i % NUM_GPUS))]}
    export CUDA_VISIBLE_DEVICES=$GPU_IDX

    # 🌟 Fix 2: Execute eval.py and remove unnecessary rendering arguments
    CMD="python eval.py \
            --dataset ${dataset} \
            --subset ${subset} \
            --scene_name ${scene} \
            --model_path ${model_path} \
            --iteration ${iter}"

    if [ "$USE_MULTI" -eq 1 ]; then
        echo "➡️  [Dispatch] Deploying scene ${scene} to GPU ${GPU_IDX} (background)"
        # 🌟 Fix 3: Change log name to logs_eval_ to protect previous logs
        eval "$CMD" > logs/logs_eval_${scene}.txt 2>&1 &  
        sleep 2 

        # Batch waiting mechanism
        if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ]; then
            echo "⏳ GPU queue full ($NUM_GPUS/$NUM_GPUS), waiting..."
            wait
        fi
    else
        echo "➡️  [Sequential] Deploying scene ${scene} to GPU ${GPU_IDX}"
        eval "$CMD"  
    fi
done

wait

# generate the evaluation metrics
SUMMARY_FLAGS="--with-state"
if [ "${dataset}" == "v2a" ]; then
    SUMMARY_FLAGS="${SUMMARY_FLAGS} --split-joint"
fi

mkdir -p experiment_results/"$SAVE_DIR"

python utils/results_summary.py \
    --dataset "${dataset}" \
    --subset "${subset}" \
    --output_dir "${OUTPUT_DIR}" \
    ${SUMMARY_FLAGS} \
    > "experiment_results/${SAVE_DIR}/${dataset}_${subset}_results.txt" 2>&1

echo "🎉 Evaluation finished successfully!"

# ====================================================
# [Step 5: Cleanup]
# ====================================================
finish_env