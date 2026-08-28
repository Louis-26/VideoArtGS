#!/bin/bash
#SBATCH --job-name=baseline                # Name of the job
#SBATCH --account=enalisn1                 # Account to charge resources to
#SBATCH --partition=parallel                   # Partition (queue) to run in
#SBATCH --qos=normal                      # Quality of Service (priority class)

#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of MPI tasks


#SBATCH --cpus-per-task=6                  # CPU cores per task
#SBATCH --mem=24G                          # Memory

#SBATCH --time=02:00:00                    # Walltime limit (HH:MM:SS)

#SBATCH --output=../SLURM_output/%x_slurm_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_slurm_%j.err             # Stderr file

#SBATCH --mail-type=ALL                             # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address
TIME_LOG_FILE="${SLURM_SUBMIT_DIR}/../SLURM_output/time_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
cd $(git rev-parse --show-toplevel)

# start time record
start_time=$(date +%s)
echo "========================================" >> "$TIME_LOG_FILE"
echo "🚀 The work starts at: $(date)" >> "$TIME_LOG_FILE"

# main script here
cd "$(git rev-parse --show-toplevel)/data"
unzip -q VideoArtGS-20.zip
unzip -q realscan.zip

mkdir -p videoartgs
mv realscan videoartgs/
mv VideoArtGS-20 videoartgs/sapien
mkdir sapien
mv *_joint_*_bg_view_* sapien/
unzip -q outputs.zip

cd "$(git rev-parse --show-toplevel)/data"
rm -rf *.zip

# end time record
end_time=$(date +%s)
echo "✅ The work ends at: $(date)" >> "$TIME_LOG_FILE"

elapsed=$(( end_time - start_time ))
echo "⏳ Total time: $elapsed seconds" >> "$TIME_LOG_FILE"

# transform time to minutes and seconds
echo "📊 Time converted: $((elapsed/60)) minutes $((elapsed%60)) seconds" >> "$TIME_LOG_FILE"
echo "========================================" >> "$TIME_LOG_FILE"


