#!/bin/bash
#SBATCH --job-name=template_execution      # Name of the job
#SBATCH --account=svillar3_gpu             # Account to charge resources to
#SBATCH --partition=a100                   # Partition (queue) to run in
#SBATCH --qos=qos_gpu                      # Quality of Service (priority class)

#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of MPI tasks
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH --cpus-per-task=6                  # CPU cores per task

#SBATCH --mem=48G                          # Total memory
#SBATCH --mem-per-cpu=8G                   # Memory per CPU core (alternative)

#SBATCH --time=02:00:00                    # Walltime limit (HH:MM:SS)

#SBATCH --gres=gpu:1                       # Request 1 GPU (classic method)
#SBATCH --gpus=1                           # Request GPUs (new syntax)
#SBATCH --gpus-per-node=1                  # GPUs per node
#SBATCH --gpu-bind=single:1                # Bind task to one GPU
#SBATCH --gpu-freq=high                    # Request high GPU frequency

#SBATCH --constraint=gpu                   # Node hardware constraint
##SBATCH --nodelist=node001                 # Force specific node
##SBATCH --exclude=node005,node006          # Exclude specific nodes

##SBATCH --distribution=block               # Task distribution across nodes
##SBATCH --cpu-bind=cores                   # Bind CPUs to cores
##SBATCH --hint=nomultithread               # Disable hyperthreading

##SBATCH --exclusive                        # Exclusive access to node
##SBATCH --share                            # Allow node sharing (opposite)

##SBATCH --array=1-10%2                     # Job array (10 jobs, 2 at a time)

##SBATCH --dependency=afterok:123456        # Run after another job succeeds
##SBATCH --kill-on-invalid-dep=yes          # Cancel if dependency fails

##SBATCH --priority=1000                    # Manual job priority
##SBATCH --nice=100                         # Lower scheduling priority

##SBATCH --requeue                          # Allow job to be requeued
##SBATCH --no-requeue                       # Disable requeueing

##SBATCH --signal=B:USR1@60                 # Send signal 60s before timeout

##SBATCH --checkpoint=120                   # Checkpoint every 120 seconds
##SBATCH --checkpoint-dir=/scratch/chkpt    # Where checkpoints go
##SBATCH --restart-dir=/scratch/chkpt       # Restart from checkpoint

##SBATCH --chdir=/home/user/project          # Set working directory
##SBATCH --export=ALL                       # Export all environment variables
##SBATCH --get-user-env                     # Load user login environment

#SBATCH --output=%x_slurm_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=%x_slurm_%j.err             # Stderr file

##SBATCH --open-mode=append                 # Append to output files
##SBATCH --comment="Full SBATCH reference"  # Arbitrary job comment

#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address

##SBATCH --wait                             # Wait for job to finish (sbatch)
##SBATCH --parsable                         # Parsable output from sbatch
##SBATCH --verbose                          # Verbose scheduler output

##SBATCH --licenses=matlab:1                # Request licensed software
##SBATCH --reservation=my_reservation       # Use reserved resources

# =========================
# Job execution starts here
# =========================
