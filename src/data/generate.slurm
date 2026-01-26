#! /bin/bash

#SBATCH --job-name=gen_data
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

# 1) Load your modules (must come before sourcing your venv) 
module purge                             # clear any inherited modules :contentReference[oaicite:8]{index=8}
module load python/python-3.11.4         # load the Python you need :contentReference[oaicite:9]{index=9}

echo "Running on $(hostname), Python: $(python --version)"

cd ../../
srun sh src/data/generate.sh
