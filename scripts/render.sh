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
RES=2
OUTPUT_DIR="outputs"
MIN_MEM=10240

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multi) USE_MULTI="$2"; shift ;;
        --keep_logs) KEEP_LOGS="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
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
# [Step 4: Training loop with proper GPU scheduling]
# ====================================================
for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    echo "========================================="
    echo "🎬 Rendering scene: ${scene}"
    model_path=${OUTPUT_DIR}/${dataset}/${subset}/${scene}/${model_name}
    
    if [ -d "${model_path}/train/ours_${iter}/renders/-1" ]; then
        echo "⏭️ [SKIP] Scene ${scene} already rendered."
        continue
    fi
    
    # Precise GPU indexing using loop index 'i'
    GPU_IDX=${GPUS[$((i % NUM_GPUS))]}
    export CUDA_VISIBLE_DEVICES=$GPU_IDX

    CMD="python render.py \
            --dataset ${dataset} \
            --subset ${subset} \
            --scene_name ${scene} \
            --model_path ${model_path} \
            --resolution ${RES} \
            --iteration ${iter} \
            --white_background"

    if [ "$USE_MULTI" -eq 1 ]; then
        echo "➡️  [Dispatch] Deploying scene ${scene} to GPU ${GPU_IDX} (background)"
        eval "$CMD" > logs/logs_render_${scene}.txt 2>&1 &  
        sleep 2 

        # Batch waiting mechanism
        if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ]; then
            echo "⏳ GPU queue full ($NUM_GPUS/$NUM_GPUS), waiting..."
            wait
        fi
    else
        echo "➡️  [Sequential] Deploying scene ${scene} to GPU ${GPU_IDX}"
        eval "$CMD" > logs/logs_render_${scene}.txt 2>&1
    fi
done
python visualization.py \
    --dataset ${dataset} \
    --subset ${subset} \
    --output_dir ${OUTPUT_DIR} > logs/logs_visualization.txt 2>&1

wait
echo "🎉 Rendering finished successfully!"

# ====================================================
# [Step 5: Cleanup]
# ====================================================
finish_env