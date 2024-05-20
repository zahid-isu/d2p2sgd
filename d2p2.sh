#!/bin/bash
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --mail-user=nsaadati@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load miniconda3
source activate dimat 
./run_cnn_cifar.sh