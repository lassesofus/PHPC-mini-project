#!/bin/bash
#BSUB -J profile_jacobi
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs_outputs/output_%J.log
#BSUB -e logs_errors/error_%J.err
#BSUB -R "select[model == XeonGold6226R]"

lscpu

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

kernprof -l -v simulate.py 12