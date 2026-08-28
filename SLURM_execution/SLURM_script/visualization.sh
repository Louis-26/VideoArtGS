#!/bin/bash
#SBATCH --job-name=visualization                # Name of the job
#SBATCH --account=enalisn1_gpu             # Account to charge resources to
#SBATCH --partition=a100                   # Partition (queue) to run in
#SBATCH --qos=qos_gpu                      # Quality of Service (priority class)

#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of MPI tasks


#SBATCH --cpus-per-task=12                  # CPU cores per task
#SBATCH --mem=48G                          # Memory per CPU core (alternative)

#SBATCH --time=04:00:00                    # Walltime limit (HH:MM:SS)

#SBATCH --gres=gpu:1                       # Request 1 GPU (classic method)


#SBATCH --output=../SLURM_output/%x_slurm_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_slurm_%j.err             # Stderr file

#SBATCH --mail-type=ALL                             # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address
source "${SLURM_SUBMIT_DIR}/hpc_utils.sh"
clear_previous_logs
count_time_on

# main script here
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh
conda activate videoartgs
cd $(git rev-parse --show-toplevel)

mkdir -p logs
for d in data/videoartgs/realscan/*/; do
    obj=$(basename "$d")
    if [[ -f "$d/filtered_vis.npz" ]]; then
        echo "[skip] $obj already done"
        continue
    fi
    echo "=== $obj ===" | tee -a logs/extract_realscan.log
    python data_tools/extract_tapip3d_track.py \
        --data_dir ./data/videoartgs/realscan \
        --video_name "$obj" \
        --reprocess 2>&1 | tee -a logs/extract_realscan.log
done

count_time_off


