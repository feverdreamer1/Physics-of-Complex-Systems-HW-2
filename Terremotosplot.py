import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import skew, kurtosis

# Set base path
base_path = r"C:\Users\shara\Desktop\Fisica Sistemas Complejos\PowerLaws\Terremotos"

# 1. Load and clean data
csv_file = os.path.join(base_path, "Terremotos.csv")
df = pd.read_csv(csv_file, sep=';', skipinitialspace=True)
df.columns = df.columns.str.strip()

# 2. Filter valid magnitudes (>= 2.5)
df['Mag.'] = pd.to_numeric(df['Mag.'], errors='coerce')
df = df.dropna(subset=['Mag.'])
df = df[df['Mag.'] >= 2.5] 

# --- STATISTICAL CHARACTERIZATION ---
magnitudes = df['Mag.'].values
N_events = len(magnitudes)
mean_mag = np.mean(magnitudes)
var_mag = np.var(magnitudes)
skew_mag = skew(magnitudes)
kurt_mag = kurtosis(magnitudes)

# 3. Calculate frequencies for plotting and regression
mag_min = df['Mag.'].min()
mag_max = df['Mag.'].max()

bins = np.arange(mag_min, mag_max + 0.2, 0.1)
counts, bin_edges = np.histogram(df['Mag.'], bins=bins)
magnitudes_bins = bin_edges[:-1]

# Filter empty bins
valid_idx = counts > 0
mag_valid = magnitudes_bins[valid_idx]
counts_valid = counts[valid_idx]

# --- LINEAR REGRESSION (b-value) ---
log_counts = np.log10(counts_valid)
m, c = np.polyfit(mag_valid, log_counts, 1) # y = mx + c

b_value = -m 

print("\n--- DATA CHARACTERIZATION (EARTHQUAKES) ---")
print(f"Total events (N): {N_events}")
print(f"Mean: {mean_mag:.4f}")
print(f"Variance (2nd moment): {var_mag:.4f}")
print(f"Skewness (3rd moment): {skew_mag:.4f}")
print(f"Kurtosis (4th moment): {kurt_mag:.4f}")
print(f"Power Law Exponent (b-value): {b_value:.4f}")
print("-------------------------------------------\n")

theoretical_counts = 10**(m * mag_valid + c)

# 4. Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Fit removed for clarity)
ax1.bar(magnitudes_bins, counts, width=0.08, color='skyblue', edgecolor='black', label='Data')
ax1.set_title('Frequency vs Magnitude')
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Number of Earthquakes')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Log-Y Scale (Gutenberg-Richter Law)
ax2.scatter(mag_valid, counts_valid, color='red', zorder=5, label='Data')
ax2.plot(mag_valid, theoretical_counts, color='green', linewidth=2.5, label=f'Regression (b={b_value:.2f})')
ax2.set_yscale('log')
ax2.set_title('Gutenberg-Richter Law (Magnitude >= 2.5)')
ax2.set_xlabel('Magnitude')
ax2.set_ylabel('Log(Number of Earthquakes)')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_file = os.path.join(base_path, "grafico_terremotos_filtrado_ajuste.png")
plt.savefig(output_file)
plt.show()