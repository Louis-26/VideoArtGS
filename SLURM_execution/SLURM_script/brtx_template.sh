#!/bin/bash
#SBATCH --job-name=RGBT_Scenes_experiment
#SBATCH --partition=ba100
#SBATCH --mem=180G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ylu174@alumni.jhu.edu
#SBATCH --output=../SLURM_output/%x_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_%j.err             # Stderr file


source /brtx/605-nvme2/ylu174/Anaconda3/etc/profile.d/conda.sh
conda activate gsplat
python script.py