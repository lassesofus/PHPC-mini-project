#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def convert_time(t_str):
    """
    Convert a string of the form 'XmYs' to seconds.
    For example, '1m41.352s' becomes 101.352 seconds.
    """
    # Use a regular expression to extract minutes and seconds.
    m = re.match(r'(?:(\d+)m)?([\d\.]+)s', t_str)
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    else:
        # Fall back to a float conversion, if possible.
        return float(t_str)

# Read the CSV file (adjust filename if necessary).
data = pd.read_csv("1_6_timing_data.txt")

# Convert time columns from string to float seconds.
data['real_time'] = data['real_time'].apply(convert_time)
data['user_time'] = data['user_time'].apply(convert_time)
data['sys_time'] = data['sys_time'].apply(convert_time)

# Compute speedup relative to single-process real time.
baseline = data.loc[data['n_proc'] == 1, 'real_time'].iloc[0]
data['speedup'] = baseline / data['real_time']
print(data['speedup'])

plt.figure(figsize=(6,4))
plt.plot(data['n_proc'], data['speedup'], marker='o', label='Observed speed-up')

# Plot Amdahl's law curves for different parallel fractions.
# Choose parallel fractions (p) e.g. 90%, 95% and 99%.
parallel_fractions = [0.93]
# Create an array for number of processes.
n_processes = np.arange(1, data['n_proc'].max()+1)
for p in parallel_fractions:
    amdahl_speedup = 1 / ((1 - p) + (p / n_processes))
    plt.plot(n_processes, amdahl_speedup, linestyle='--', label=f"Amdahl (F={p})")

plt.xlabel("Number of Processes")
plt.ylabel("Speed-up")
plt.xticks(np.arange(0, data['n_proc'].max()+1, 5))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_plot.png", dpi=300)
plt.show()