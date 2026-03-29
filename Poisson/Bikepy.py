import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, skew, kurtosis

# 1. Load the bike sharing dataset
file_path = "hour.csv"
print(f"Reading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Extract and filter wind speed (exclude 0s for dynamic fluid behavior)
windspeed = df['windspeed'].values
windspeed = windspeed[windspeed > 0]

# --- STATISTICAL CHARACTERIZATION ---
N_wind = len(windspeed)
mean_wind = np.mean(windspeed)
var_wind = np.var(windspeed)
skew_wind = skew(windspeed)
kurt_wind = kurtosis(windspeed)

# 3. Theoretical Rayleigh Fit
loc, sigma_rayleigh = rayleigh.fit(windspeed, floc=0)

print("\n--- DATA CHARACTERIZATION (RAYLEIGH - WIND SPEED) ---")
print(f"Total data points (N): {N_wind}")
print(f"Mean Wind Speed: {mean_wind:.4f}")
print(f"Variance (2nd moment): {var_wind:.4f}")
print(f"Skewness (3rd moment): {skew_wind:.4f}")
print(f"Kurtosis (4th moment): {kurt_wind:.4f}")
print(f"Rayleigh parameter (sigma): {sigma_rayleigh:.4f}")
print("-----------------------------------------------------\n")

# --- THEORETICAL RAYLEIGH CURVE ---
x_min = 0
x_max = np.max(windspeed) * 1.1
x_fit = np.linspace(x_min, x_max, 200)
pdf_theoretical = rayleigh.pdf(x_fit, loc=0, scale=sigma_rayleigh)

print("Generating plots...")
# 4. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale
ax1.hist(windspeed, bins=30, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='Real Data (Wind Speed)')
ax1.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label=f'Rayleigh Fit ($\sigma$={sigma_rayleigh:.3f})')
ax1.set_title('Wind Speed Distribution (Linear Scale)')
ax1.set_xlabel('Normalized Wind Speed')
ax1.set_ylabel('Probability Density')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Semi-Log Scale (Log-Y)
ax2.hist(windspeed, bins=30, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='Real Data (Wind Speed)')
ax2.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label='Theoretical Curve')
ax2.set_yscale('log')
ax2.set_title('Wind Speed Distribution (Log-Y Scale)')
ax2.set_xlabel('Normalized Wind Speed')
ax2.set_ylabel('Log Probability Density')
ax2.set_ylim(bottom=1e-3) 
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "viento_rayleigh.png"
plt.savefig(output_path)
plt.show()