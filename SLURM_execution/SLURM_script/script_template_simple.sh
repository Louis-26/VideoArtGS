#!/bin/bash
#SBATCH --job-name=baseline                # Name of the job
#SBATCH --account=enalisn1_gpu             # Account to charge resources to
#SBATCH --partition=a100                   # Partition (queue) to run in
#SBATCH --qos=qos_gpu                      # Quality of Service (priority class)

#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of MPI tasks


#SBATCH --cpus-per-task=12                  # CPU cores per task
#SBATCH --mem=48G                          # Memory per CPU core (alternative)

#SBATCH --time=02:00:00                    # Walltime limit (HH:MM:SS)

#SBATCH --gres=gpu:1                       # Request 1 GPU (classic method)


#SBATCH --output=../SLURM_output/%x_slurm_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_slurm_%j.err             # Stderr file

#SBATCH --mail-type=ALL                             # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address
TIME_LOG_FILE="${SLURM_SUBMIT_DIR}/../SLURM_outcome/time_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
cd $(git rev-parse --show-toplevel)

# start time record
start_time=$(date +%s)
echo "========================================" >> "$TIME_LOG_FILE"
echo "🚀 The work starts at: $(date)" >> "$TIME_LOG_FILE"

# main script here
# MAIN_SCRIPT_EXECUTION_COMMAND

# end time record
end_time=$(date +%s)
echo "✅ The work ends at: $(date)" >> "$TIME_LOG_FILE"

elapsed=$(( end_time - start_time ))
echo "⏳ Total time: $elapsed seconds" >> "$TIME_LOG_FILE"

# transform time to minutes and seconds
echo "📊 Time converted: $((elapsed/60)) minutes $((elapsed%60)) seconds" >> "$TIME_LOG_FILE"
echo "========================================" >> "$TIME_LOG_FILE"


