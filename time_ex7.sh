#!/bin/bash
#BSUB -J jit_timing
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs_outputs/output_%J.log
#BSUB -e logs_errors/error_%J.err
#BSUB -R "select[model == XeonGold6226R]"

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# REGULAR
time python -u simulate_ex_6.py 4
time python -u simulate_ex_6.py 4
time python -u simulate_ex_6.py 4
# NUMBA JIT
time python -u simulate_ex_7.py 4
time python -u simulate_ex_7.py 4
time python -u simulate_ex_7.py 4