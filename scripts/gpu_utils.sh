#!/bin/bash
# Filename: gpu_utils.sh
# A reusable tool library for GPU and environment initialization

function init_env() {
    # 1. Parse parameters
    USE_MULTI=${1:-0}
    KEEP_LOGS=${2:-0}
    MIN_MEM=${3:-10240} # by default we set 10GB as the minimum requirement

    # Export variables so the calling script can use them
    export USE_MULTI KEEP_LOGS MIN_MEM

    mkdir -p logs time_logs

    # 2. Check SLURM/local environment and record time
    if [ -n "$SLURM_JOB_ID" ]; then
        IS_SLURM=1
        echo "📝 [System] Running under SLURM environment (Job ID: $SLURM_JOB_ID)."
    else
        IS_SLURM=0
        START_TIME=$(date +%s)
        SCRIPT_NAME=$(basename "$0" .sh)
        TIME_LOG_FILE="time_logs/time_${SCRIPT_NAME}.txt"
        echo "========================================" > "$TIME_LOG_FILE"
        echo "🚀 The work starts at: $(date)" >> "$TIME_LOG_FILE"
        echo "⏱️ [System] Running locally (Not SLURM). Time tracking activated."
    fi
    export IS_SLURM START_TIME TIME_LOG_FILE
    echo "------------------------------------------------"

    # 3. Automatically detect CUDA Architecture
    echo "🔍 Automatically detecting current GPU CUDA Architecture..."
    CUDA_ARCH=$(python -c "import torch; major, minor = torch.cuda.get_device_capability(); print(f'{major}.{minor}')" 2>/dev/null)

    if [ -z "$CUDA_ARCH" ]; then
        echo "❌ Warning: No CUDA was detected."
        exit 1
    fi
    export TORCH_CUDA_ARCH_LIST=$CUDA_ARCH
    echo "✅ TORCH_CUDA_ARCH_LIST automatically set to: $TORCH_CUDA_ARCH_LIST"
    echo "------------------------------------------------"

    # 4. Dynamically find available GPUs (> ${MIN_MEM}GB free)
    AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' -v min_mem="$MIN_MEM" '$2 > min_mem {print $1}'))
    
    if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
        echo "❌ Warning: No available GPUs with > ${MIN_MEM} GB free memory found! Please try again later."
        exit 1
    fi

    GPU_TYPE=$(nvidia-smi --id=${AVAILABLE_GPUS[0]} --query-gpu=name --format=csv,noheader)

    if [ "$USE_MULTI" -eq 1 ]; then
        GPUS=("${AVAILABLE_GPUS[@]}")
        echo "🔥 Grabbed ${#GPUS[@]} x [${GPU_TYPE}] GPUs!"
        [ "$IS_SLURM" -eq 0 ] && echo "GPU Resources: ${#GPUS[@]} x [${GPU_TYPE}] GPUs" >> "$TIME_LOG_FILE"
        echo "🚀 [Full Firepower] Multi-GPU concurrent mode activated! GPUs in use: ${GPUS[*]}"
    else
        GPUS=(${AVAILABLE_GPUS[0]})
        echo "🔥 Grabbed 1 x [${GPU_TYPE}] GPU!"
        echo "🐢 [Single Agent] Single-GPU mode (default). Using GPU: ${GPUS[0]}"
    fi

    NUM_GPUS=${#GPUS[@]}

    # Convert the GPU array to a comma-separated string for CUDA_VISIBLE_DEVICES
    GPU_LIST=$(IFS=,; echo "${GPUS[*]}")
    export GPUS NUM_GPUS GPU_LIST
    echo "------------------------------------------------"
}

function finish_env() {
    if [ "$USE_MULTI" -eq 1 ]; then
        wait
        echo "🎉 Mission Accomplished! All multi-GPU concurrent tasks are finished!"
    else
        echo "🎉 Mission Accomplished! Single-GPU sequential tasks are finished!"
    fi

    if [ "$IS_SLURM" -eq 0 ]; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "✅ The work ends at: $(date)" >> "$TIME_LOG_FILE"
        echo "⏱️ Total elapsed time: $ELAPSED seconds ($(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s)" >> "$TIME_LOG_FILE"
    fi

    if [ "$KEEP_LOGS" -eq 0 ]; then
        echo "🧹 Cleaning up logs..."
        rm -rf logs/*
        echo "🧹 Logs cleaned."
    else
        echo "💾 Logs retained as per user request."
    fi
}