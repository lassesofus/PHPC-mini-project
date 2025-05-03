import matplotlib.pyplot as plt
import csv
import re

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

# Load data from both timing files.
data_1_6 = load_data("1_6_timing_data.txt")
data_1_5 = load_data("1_5_timing_data.txt")

# Sort keys to align the x-axis (number of processes).
n_proc_1_6 = sorted(data_1_6.keys())
real_time_1_6 = [data_1_6[n] for n in n_proc_1_6]

n_proc_1_5 = sorted(data_1_5.keys())
real_time_1_5 = [data_1_5[n] for n in n_proc_1_5]

plt.figure(figsize=(8,6))
plt.plot(n_proc_1_6, real_time_1_6, marker='o', label='Dynamic scheduling')
plt.plot(n_proc_1_5, real_time_1_5, marker='s', label='Static scheduling')
plt.xlabel("Number of Processes")
plt.ylabel("Real Execution Time (minutes)")
#plt.title("Comparison of Real Time for Different Process Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("timing_comparison.png")
plt.show()