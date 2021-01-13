#!/bin/bash

#SBATCH --job-name=gr_ho
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate /gpfs/users/piatc/.conda/envs/gr

# Go to the directory where the job has been submitted 
cd ${SLURM_SUBMIT_DIR}

# Execution
python -u hyperoptimization.py -d french_tweets.csv -n 50 --max_size 100000