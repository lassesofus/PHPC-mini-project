#!/bin/bash
#BSUB -J simulate_ex_6
#BSUB -q hpc
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 8:00
#BSUB -o output_%J.log
#BSUB -e error_%J.err
#BSUB -R "select[model == XeonGold6226R]"

lscpu

# Activate your conda environment.
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

processes_list="1 2 4 8 16 32"
output_file="1_6_timing_data.txt"
echo "n_proc,real_time,user_time,sys_time" > $output_file

# Assume simulate_ex_5.py loads at most 100 floorplans and expects a second argument
# for the number of processes.
for n_proc in $processes_list; do
    echo "Running simulate_ex_6.py with $n_proc processes"
    # Capture the time output.
    times=$( { time python simulate_ex_6.py $n_proc; } 2>&1 | grep -E 'real|user|sys' | awk '{print $2}' | tr '\n' ',' | sed 's/,$//' )
    echo "$n_proc,$times" >> $output_file
done

# Call your plotting script, for example:
python3 1_5_plotting.py