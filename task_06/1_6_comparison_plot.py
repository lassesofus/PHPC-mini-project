import matplotlib.pyplot as plt
import csv
import re
import numpy as np

def parse_time_to_minutes(time_str):
    """
    Convert a time string formatted as 'XmYs' into total minutes.
    Example: "21m28.212s" -> (21*60 + 28.212)/60 minutes.
    """
    match = re.match(r'(\d+)m([\d\.]+)s', time_str)
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))
        total_minutes = (minutes * 60 + seconds) / 60.0
        return total_minutes
    else:
        raise ValueError(f"Time string '{time_str}' is not in the expected format.")

def load_data(filename):
    """
    Load timing data from a CSV file.
    Expected header: n_proc,real_time,user_time,sys_time
    Returns a dictionary: { n_proc: real_time_in_minutes }
    """
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_proc = int(row['n_proc'])
                real_time = parse_time_to_minutes(row['real_time'])
                data[n_proc] = real_time
            except Exception as e:
                print("Error parsing row:", row, e)
    return data

def compute_speedup(data):
    """
    Compute speedup using the time for 1 process as baseline:
    Speedup(n) = T(1) / T(n)
    """
    if 1 not in data:
        raise ValueError("Data does not contain information for 1 process (baseline).")
    baseline = data[1]
    speedup = { n: baseline / data[n] for n in data }
    return speedup

def amdahl_speedup(n, p):
    """
    Compute Amdahl's speedup with parallelizable fraction p.
    n: number of processes
    p: parallel fraction (0<p<=1), serial fraction = 1 - p.
    
    Returns: speedup(n) = 1 / ((1-p) + p/n)
    """
    return 1.0 / ((1 - p) + p / n)

# Load timing data from files.
data_static = load_data("1_5_timing_data.txt")
data_dynamic = load_data("1_6_timing_data.txt")

# Compute speedups from data.
speedup_static = compute_speedup(data_static)
speedup_dynamic = compute_speedup(data_dynamic)

# Sort speedup data by number of processes.
n_proc_static = sorted(speedup_static.keys())
speedup_static_list = [speedup_static[n] for n in n_proc_static]
n_proc_dynamic = sorted(speedup_dynamic.keys())
speedup_dynamic_list = [speedup_dynamic[n] for n in n_proc_dynamic]

# Prepare Amdahl curves.
max_n = max(max(n_proc_static), max(n_proc_dynamic))
n_vals = np.arange(1, max_n+1)

# Assumed parallel fractions (modify these as needed).
p_static = 0.91  # e.g., 95% of the work is parallel for static scheduling.
p_dynamic = 0.93  # e.g., 85% of the work is parallel for dynamic scheduling.

amdahl_static = [amdahl_speedup(n, p_static) for n in n_vals]
amdahl_dynamic = [amdahl_speedup(n, p_dynamic) for n in n_vals]

plt.figure(figsize=(8,6))
# Plot measured speedups.
plt.plot(n_proc_static, speedup_static_list, marker='s',
         linestyle='-', label='Static scheduling')
plt.plot(n_proc_dynamic, speedup_dynamic_list, marker='o',
         linestyle='-', label='Dynamic scheduling')

# Overlay Amdahl curves.
plt.plot(n_vals, amdahl_static, 'b--', label=f"Amdahl, F={p_static}")
plt.plot(n_vals, amdahl_dynamic, 'r--', label=f"Amdahl, F={p_dynamic}")

plt.xlabel("Number of Processes")
plt.ylabel("Speed-up")
#plt.title("Speedup Comparison with Amdahl's Law")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_comparison.png")
plt.show()