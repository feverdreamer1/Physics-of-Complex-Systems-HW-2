import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis

# 1. Load the Robbins Lunar Crater Database
file_path = "lunar_crater_database_robbins_2018.csv" 
print(f"Loading database from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)
diameter_col = 'DIAM_CIRC_IMG' 

# 2. Filter by completeness limit (>= 2.5 km)
diameters = pd.to_numeric(df[diameter_col], errors='coerce').dropna().values
diameters = diameters[diameters >= 2.5]

# --- STATISTICAL CHARACTERIZATION ---
N_craters = len(diameters)
mean_diam = np.mean(diameters)
var_diam = np.var(diameters)
skew_diam = skew(diameters)
kurt_diam = kurtosis(diameters)

# --- LOGARITHMIC BINNING FOR LOG-LOG PLOT ---
max_d = diameters.max()
min_d = diameters.min()

bins_log = np.logspace(np.log10(min_d), np.log10(max_d), num=50)
counts_log, edges_log = np.histogram(diameters, bins=bins_log)

centers_log = np.sqrt(edges_log[:-1] * edges_log[1:]) 
widths_log = np.diff(edges_log)
density = counts_log / widths_log

valid_idx = density > 0
x_fit = centers_log[valid_idx]
y_fit = density[valid_idx]

# --- LINEAR REGRESSION ---
mask_fit = x_fit < (max_d * 0.2)
log_x = np.log10(x_fit[mask_fit])
log_y = np.log10(y_fit[mask_fit])

m, c = np.polyfit(log_x, log_y, 1)
theoretical_density = 10**(m * np.log10(x_fit) + c)
gamma_value = -m 

print("\n--- DATA CHARACTERIZATION (LUNAR CRATERS) ---")
print(f"Total craters (N): {N_craters}")
print(f"Mean Diameter: {mean_diam:.2f} km")
print(f"Variance (2nd moment): {var_diam:.2f} km^2")
print(f"Skewness (3rd moment): {skew_diam:.4f}")
print(f"Kurtosis (4th moment): {kurt_diam:.4f}")
print(f"Size-Frequency Exponent (gamma): {gamma_value:.4f}")
print("---------------------------------------------\n")

# --- STANDARD COUNT FOR LINEAR PLOT ---
bins_linear = np.linspace(min_d, 20, 40) 
counts_lin, edges_lin = np.histogram(diameters, bins=bins_linear)
centers_lin = (edges_lin[:-1] + edges_lin[1:]) / 2
width_lin = bins_linear[1] - bins_linear[0]

print("Generating plots...")
# 3. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Fit removed for clarity)
ax1.bar(centers_lin, counts_lin, width=width_lin*0.85, color='skyblue', edgecolor='black', label='Real Craters')
ax1.set_title('Crater Size Frequency (Diameters < 20 km)')
ax1.set_xlabel('Crater Diameter (km)')
ax1.set_ylabel('Number of Craters')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Log-Log Scale (Power Law Fit)
ax2.scatter(x_fit, y_fit, color='red', s=30, alpha=0.8, label='Log-Binned Data', zorder=5)
ax2.plot(x_fit, theoretical_density, color='green', linewidth=2.5, label=f'Regression (Slope={m:.2f})')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Lunar Crater Distribution (Robbins 2018)')
ax2.set_xlabel('Crater Diameter (km)')
ax2.set_ylabel('Density of Craters P(D)')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "robbins_lunar_craters_plot.png" 
plt.savefig(output_path)
plt.show()