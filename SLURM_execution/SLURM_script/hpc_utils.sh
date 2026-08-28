# Function to clear previous logs matching the current job name
clear_previous_logs() {
    OVERWRITE_LOGS=${OVERWRITE_LOGS:-true}
    # Use absolute paths to ensure reliability
    local OUTPUT_DIR="${SLURM_SUBMIT_DIR}/../SLURM_output"
    local OUTCOME_DIR="${SLURM_SUBMIT_DIR}/../SLURM_outcome"
    
    if [ "$OVERWRITE_LOGS" = "true" ]; then
        # Clean up output, error, and time logs excluding the current job ID
        find "$OUTPUT_DIR" -name "${SLURM_JOB_NAME}_slurm_*.out" ! -name "*_${SLURM_JOB_ID}.out" -type f -delete 2>/dev/null
        find "$OUTPUT_DIR" -name "${SLURM_JOB_NAME}_slurm_*.err" ! -name "*_${SLURM_JOB_ID}.err" -type f -delete 2>/dev/null
        find "$OUTCOME_DIR" -name "time_${SLURM_JOB_NAME}_*.txt" ! -name "*_${SLURM_JOB_ID}.txt" -type f -delete 2>/dev/null
        echo "🧹 [Cleanup] Deleted previous logs."
    fi
}

# Function to initialize the timer and log file
count_time_on(){
    # Ensure TIME_LOG_FILE is globally accessible
    export TIME_LOG_FILE="${SLURM_SUBMIT_DIR}/../SLURM_outcome/time_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
    export START_TIME=$(date +%s)
    echo "========================================" >> "$TIME_LOG_FILE"
    echo "🚀 The work starts at: $(date)" >> "$TIME_LOG_FILE"
}

# Function to calculate elapsed time and finalize logs
count_time_off(){
    local END_TIME=$(date +%s)
    local ELAPSED=$(( END_TIME - START_TIME ))
    echo "✅ The work ends at: $(date)" >> "$TIME_LOG_FILE"
    echo "⏳ Total time: $ELAPSED seconds" >> "$TIME_LOG_FILE"
    echo "📊 Time converted: $((ELAPSED/60)) minutes $((ELAPSED%60)) seconds" >> "$TIME_LOG_FILE"
    echo "========================================" >> "$TIME_LOG_FILE"
}