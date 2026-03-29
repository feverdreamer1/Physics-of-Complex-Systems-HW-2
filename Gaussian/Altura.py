import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
import os

# 1. Load CSV file
file_path = "weight-height.csv"
print(f"Reading data from {file_path}...")

try:
    df = pd.read_csv(file_path)
    # Extract height column
    heights = pd.to_numeric(df['Height'], errors='coerce').dropna().values
except FileNotFoundError:
    print("Error: CSV file not found. Generating fallback data.")
    np.random.seed(42)
    heights = np.random.normal(loc=168.0, scale=7.0, size=10000)

# --- STATISTICAL CHARACTERIZATION ---
N_people = len(heights)
mean_height = np.mean(heights)
std_height = np.std(heights)
var_height = np.var(heights)
skew_height = skew(heights)
kurt_height = kurtosis(heights) # Excess kurtosis

print("\n--- DATA CHARACTERIZATION (NORMAL - HEIGHT) ---")
print(f"Total analyzed (N): {N_people}")
print(f"Mean Height (mu): {mean_height:.2f} cm")
print(f"Variance (2nd moment): {var_height:.2f} cm^2")
print(f"Skewness (3rd moment): {skew_height:.4f}")
print(f"Kurtosis (4th moment): {kurt_height:.4f}")
print("-----------------------------------------------\n")

# --- THEORETICAL GAUSSIAN CURVE ---
x_min = np.min(heights) - 5
x_max = np.max(heights) + 5
x_fit = np.linspace(x_min, x_max, 200)
pdf_theoretical = norm.pdf(x_fit, loc=mean_height, scale=std_height)

print("Generating plots...")
# 2. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Gaussian Bell Curve)
ax1.hist(heights, bins=50, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='Real Data')
ax1.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label=f'Gaussian ($\mu$={mean_height:.1f}, $\sigma$={std_height:.1f})')
ax1.set_title('Human Height Distribution (Linear Scale)')
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Probability Density')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Semi-Log Scale (Log-Y)
ax2.hist(heights, bins=50, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='Real Data')
ax2.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label='Theoretical Parabola')
ax2.set_yscale('log')
ax2.set_title('Human Height Distribution (Log-Y Scale)')
ax2.set_xlabel('Height (cm)')
ax2.set_ylabel('Log Probability Density')
ax2.set_ylim(bottom=1e-5) 
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "estatura_dataset_continuo.png"
plt.savefig(output_path)
plt.show()