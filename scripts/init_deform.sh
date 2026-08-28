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
MIN_MEM=5120    

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

model_name=init
seed=0

# ====================================================
# [Step 4: Deform Initialization loop with GPU scheduling]
# ====================================================
for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    echo "========================================="
    echo "🎬 Initializing deformation field for scene: ${scene}"
    model_path="${OUTPUT_DIR}/${dataset}/${subset}/${scene}/${model_name}"
    
    # Check if the initialization has already completed
    if [ -d "${model_path}/deform/iteration_10000" ]; then
        echo "⏭️ [SKIP] Scene ${scene} deformation initialization already completed."
        continue
    fi
    
    # Precise GPU indexing using loop index 'i'
    GPU_IDX=${GPUS[$((i % NUM_GPUS))]}
    export CUDA_VISIBLE_DEVICES=$GPU_IDX

    CMD="python init_deform.py \
            --dataset ${dataset} \
            --subset ${subset} \
            --scene_name ${scene} \
            --model_path ${model_path} \
            --iterations 10000 \
            --seed ${seed}"

    if [ "$USE_MULTI" -eq 1 ]; then
        echo "➡️  [Dispatch] Deploying scene ${scene} to GPU ${GPU_IDX} (background)"
        # Log named logs_init_deform_ to avoid conflicts
        eval "$CMD" > logs/logs_init_deform_${scene}.txt 2>&1 &  
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
echo "🎉 Deformation initialization finished successfully!"

# ====================================================
# [Step 5: Cleanup]
# ====================================================
finish_env