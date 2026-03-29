import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
from collections import Counter
from scipy.stats import skew, kurtosis

# 1. Download SNAP citation network dataset
url = "https://snap.stanford.edu/data/cit-HepPh.txt.gz"
file_name = "cit-HepPh.txt"

if not os.path.exists(file_name):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, file_name)

print("Processing citation network...")
df = pd.read_csv(file_name, sep='\t', comment='#', header=None, names=['FromNodeId', 'ToNodeId'])
in_degrees = df['ToNodeId'].value_counts().values

# --- STATISTICAL CHARACTERIZATION ---
N_nodes = len(in_degrees)
mean_deg = np.mean(in_degrees)
var_deg = np.var(in_degrees)
skew_deg = skew(in_degrees)
kurt_deg = kurtosis(in_degrees)

# --- 1. DATA PREP FOR LINEAR PLOT (Raw Counts) ---
degree_counts = Counter(in_degrees)
degrees_lin = np.array(list(degree_counts.keys()))
freq_lin = np.array(list(degree_counts.values()))

sort_idx = np.argsort(degrees_lin)
degrees_lin = degrees_lin[sort_idx]
freq_lin = freq_lin[sort_idx]

# --- 2. DATA PREP FOR LOG-LOG PLOT (Log-Binning) ---
min_deg = max(1, np.min(in_degrees))
max_deg = np.max(in_degrees)

bins_log = np.logspace(np.log10(min_deg), np.log10(max_deg), num=30)
counts_log, edges_log = np.histogram(in_degrees, bins=bins_log)

centers_log = np.sqrt(edges_log[:-1] * edges_log[1:])
widths_log = np.diff(edges_log)
density = counts_log / widths_log

valid_idx = density > 0
x_fit = centers_log[valid_idx]
y_fit = density[valid_idx]

# --- LINEAR REGRESSION (Binned Data) ---
mask_fit = (x_fit >= 10) & (x_fit <= max_deg * 0.5)
log_x = np.log10(x_fit[mask_fit])
log_y = np.log10(y_fit[mask_fit])

m, c = np.polyfit(log_x, log_y, 1)
gamma_value = -m
theoretical_density = 10**(m * np.log10(x_fit) + c)

print("\n--- DATA CHARACTERIZATION (SCALE-FREE NETWORK) ---")
print(f"Total cited papers (N): {N_nodes}")
print(f"Mean citations (In-degree): {mean_deg:.4f}")
print(f"Variance (2nd moment): {var_deg:.4f}")
print(f"Skewness (3rd moment): {skew_deg:.4f}")
print(f"Kurtosis (4th moment): {kurt_deg:.4f}")
print(f"Degree Exponent (gamma): {gamma_value:.4f}")
print("--------------------------------------------------\n")

print("Generating plots...")
# 3. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Raw bars, no fit)
mask_lin_plot = degrees_lin <= 50
ax1.bar(degrees_lin[mask_lin_plot], freq_lin[mask_lin_plot], width=0.8, color='skyblue', edgecolor='black', label='Real Data')
ax1.set_title('In-Degree Distribution (0-50 citations)')
ax1.set_xlabel('Number of Citations ($k$)')
ax1.set_ylabel('Number of Papers')
ax1.set_ylim(0, 4500)
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Log-Log Scale (Log-Binning with Fit)
ax2.scatter(x_fit, y_fit, color='red', s=30, alpha=0.8, label='Log-Binned Data', zorder=5)
ax2.plot(x_fit, theoretical_density, color='green', linewidth=2.5, label=f'Regression ($\gamma$={gamma_value:.2f})')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Scale-Free Citation Network (Log-Log Density)')
ax2.set_xlabel('Number of Citations ($k$)')
ax2.set_ylabel('Probability Density $P(k)$')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "network.png" 
plt.savefig(output_path)
plt.show()