import pandas as pd
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv('1_12_results.txt', 
                 sep=",",
                 header=0,  # Assumes the file has a header row
                 dtype={
                     "building_id": int,
                     "mean_temp": float,
                     "std_temp": float,
                     "pct_above_18": float,
                     "pct_below_15": float
                 })


# Plotting the distribution of mean temperatures
plt.figure(figsize=(8, 6))
plt.hist(df["mean_temp"], bins=10, edgecolor="black")
plt.xlabel("Mean Temperature (ºC)")
plt.ylabel("Count")
plt.title("Distribution of Mean Temperatures")
plt.grid(True)
# Place grid below plot elements
ax = plt.gca()
ax.set_axisbelow(True)
plt.savefig("mean_temp_distribution.png")
plt.show()

print(f"Average mean temperature: {df['mean_temp'].mean():.2f} ºC")
print(f"Average standard deviation of temperature: {df['std_temp'].mean():.2f} ºC")
print(f"Number of building with at least 50% of their area above 18ºC: {len(df[df['pct_above_18'] >= 50])}")
print(f"Number of building with at least 50% of their area below 15ºC: {len(df[df['pct_below_15'] >= 50])}")

