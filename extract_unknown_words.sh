#!/bin/bash

#SBATCH --job-name=gr_uw
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_med
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=16

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate /gpfs/users/piatc/.conda/envs/gr

# Go to the directory where the job has been submitted 
cd ${SLURM_SUBMIT_DIR}

# Execution
python -u extract_unknown_words.py -p french_tweets.csv