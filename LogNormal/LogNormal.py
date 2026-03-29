import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, skew, kurtosis

# 1. Load Global Gini Index data (World Bank)
try:
    df = pd.read_csv("API_SI.POV.GINI_DS2_en_csv_v2_47.csv", skiprows=4)
    gini_values = df.loc[:, '1960':'2023'].values.flatten()
    gini_values = gini_values[~np.isnan(gini_values)]
except FileNotFoundError:
    print("Error: CSV file not found. Ensure the filename matches.")
    exit()

# --- STATISTICAL CHARACTERIZATION ---
N_data = len(gini_values)
mean_gini = np.mean(gini_values)
var_gini = np.var(gini_values)
skew_gini = skew(gini_values)
kurt_gini = kurtosis(gini_values)

# 2. Theoretical Log-Normal Fit
# floc=0 ensures the distribution is bounded at 0 (Gini cannot be negative)
shape, loc, scale = lognorm.fit(gini_values, floc=0)

print("\n--- DATA CHARACTERIZATION (LOG-NORMAL - GINI) ---")
print(f"Total Gini data points (N): {N_data}")
print(f"Mean Gini Index: {mean_gini:.4f}")
print(f"Variance (2nd moment): {var_gini:.4f}")
print(f"Skewness (3rd moment): {skew_gini:.4f}")
print(f"Kurtosis (4th moment): {kurt_gini:.4f}")
print(f"Log-Normal parameter (Shape/Sigma): {shape:.4f}")
print(f"Log-Normal parameter (Scale/exp(mu)): {scale:.4f}")
print("-------------------------------------------------\n")

# --- THEORETICAL LOG-NORMAL CURVE ---
x_min = np.min(gini_values) * 0.8
x_max = np.max(gini_values) * 1.1
x_fit = np.linspace(x_min, x_max, 200)
pdf_theoretical = lognorm.pdf(x_fit, s=shape, loc=loc, scale=scale)

print("Generating plots...")
# 3. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale
ax1.hist(gini_values, bins=40, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='World Bank Gini Data')
ax1.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label=f'Log-Normal Fit (Shape={shape:.2f})')
ax1.set_title('Global Gini Index Distribution (Linear Scale)')
ax1.set_xlabel('Gini Index')
ax1.set_ylabel('Probability Density')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Semi-Log Scale (Log-Y)
ax2.hist(gini_values, bins=40, density=True, color='lightcoral', edgecolor='black', alpha=0.7, label='World Bank Gini Data')
ax2.plot(x_fit, pdf_theoretical, color='green', linewidth=2.5, label='Theoretical Curve')
ax2.set_yscale('log')
ax2.set_title('Global Gini Index (Log-Y Scale)')
ax2.set_xlabel('Gini Index')
ax2.set_ylabel('Log Probability Density')
ax2.set_ylim(bottom=1e-4) 
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "gini_lognormal.png"
plt.savefig(output_path)
plt.show()