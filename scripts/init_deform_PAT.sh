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
PAT_MODEL_PTH="particulate/model_ckpt/pat_model.pt"
PAT_NUM_POINTS=65536

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multi) USE_MULTI="$2"; shift ;;
        --keep_logs) KEEP_LOGS="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --PAT_model_pth) PAT_MODEL_PTH="$2"; shift ;;
        --pat_num_points) PAT_NUM_POINTS="$2"; shift ;;
        *) echo "❌ Error: Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# PAT's point-mask decoder allocates N x max_parts x hidden*4, so peak memory is
# linear in --pat_num_points: measured 31.3 GiB at 65536, 16.1 at 32768, 8.6 at
# 16384. A fixed MIN_MEM=5120 let init_env pick GPUs that then OOM'd, so derive
# it (~0.5 MiB/point + ~4 GiB for the weights and PartField).
MIN_MEM=$(( 4096 + PAT_NUM_POINTS / 2 ))

# Fail before dispatching N background jobs that would each die on the same
# missing checkpoint (relative paths resolve against the repo root).
REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ "$PAT_MODEL_PTH" = /* ]]; then
    PAT_CKPT_ABS="$PAT_MODEL_PTH"
else
    PAT_CKPT_ABS="${REPO_ROOT}/${PAT_MODEL_PTH}"
fi
if [ ! -f "$PAT_CKPT_ABS" ]; then
    echo "❌ Error: PAT checkpoint not found: ${PAT_CKPT_ABS}"
    exit 1
fi
echo "🧠 PAT checkpoint: ${PAT_MODEL_PTH}"
echo "🧮 Decoder points: ${PAT_NUM_POINTS} (requires ~$((MIN_MEM / 1024)) GiB free per GPU)"

# ====================================================
# [Step 3: Initialize environment]
# ====================================================
init_env "$USE_MULTI" "$KEEP_LOGS" "$MIN_MEM"

source $(git rev-parse --show-toplevel)/scripts/scene_set.sh
parse_mode "$MODE"

model_name=init
seed=0

# ====================================================
# [Step 4: PAT Deform Initialization loop with GPU scheduling]
# ====================================================
for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    echo "========================================="
    echo "🎬 Initializing PAT deformation field for scene: ${scene}"
    model_path="${OUTPUT_DIR}/${dataset}/${subset}/${scene}/${model_name}"
    
    # Check if the PAT initialization has already completed
    if [ -d "${model_path}/deform/iteration_1" ]; then
        echo "⏭️ [SKIP] Scene ${scene} PAT deformation initialization already completed."
        continue
    fi
    
    # Precise GPU indexing using loop index 'i'
    GPU_IDX=${GPUS[$((i % NUM_GPUS))]}
    export CUDA_VISIBLE_DEVICES=$GPU_IDX

    CMD="python PAT/init_deform_PAT.py \
            --dataset ${dataset} \
            --subset ${subset} \
            --scene_name ${scene} \
            --model_path ${model_path} \
            --iterations 1 \
            --seed ${seed} \
            --pat_num_points ${PAT_NUM_POINTS} \
            --PAT_model_pth ${PAT_CKPT_ABS}"

    if [ "$USE_MULTI" -eq 1 ]; then
        echo "➡️  [Dispatch] Deploying scene ${scene} to GPU ${GPU_IDX} (background)"
        # Log named logs_init_deform_PAT_ to avoid conflicts
        eval "$CMD" > logs/logs_init_deform_PAT_${scene}.txt 2>&1 &  
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
echo "🎉 PAT Deformation initialization finished successfully!"

# ====================================================
# [Step 5: Cleanup]
# ====================================================
finish_env