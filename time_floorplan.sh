#!/bin/bash
#BSUB -J floorplan_simulate
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o output_%J.log
#BSUB -e error_%J.err
#BSUB -R "select[model == XeonGold6226R]"

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Test number of floorplans
#N = 10

time python simulate.py 10

