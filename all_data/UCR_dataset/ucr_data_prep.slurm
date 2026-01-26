#! /bin/bash

#SBATCH --job-name=ts-train
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00

# 1) Load your modules (must come before sourcing your venv) 
module purge                            
module load python/python-3.11.4       

echo "Running on $(hostname), Python: $(python --version)"

srun --exclusive -N1 -n1 python ucr_data_prep.py ./ --window 30
