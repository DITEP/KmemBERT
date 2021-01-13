#!/bin/bash

#SBATCH --job-name=gr_t
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
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
python -u training.py -e 3 -d french_tweets.csv -max 100000 -lr 5e-5 -k 100 #-v medical_voc/french_tweets.csv_1000_0.csv