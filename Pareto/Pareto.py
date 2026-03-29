import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os

# 1. Load Real Forbes Billionaires Dataset
file_path = "forbes_billionaires.csv"
print(f"Loading database from {file_path}...")

try:
    df = pd.read_csv(file_path)
    # The wealth column might be named 'NetWorth' or 'finalWorth' depending on the specific Kaggle CSV
    wealth_col = 'NetWorth' if 'NetWorth' in df.columns else 'finalWorth'
    
    # Extract wealth in billions and drop NaNs
    wealth = pd.to_numeric(df[wealth_col], errors='coerce').dropna().values
except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Please download a Forbes Billionaires dataset from Kaggle.")
    exit()

# Sort from richest to poorest
wealth_sorted = np.sort(wealth)[::-1]

# --- STATISTICAL CHARACTERIZATION ---
N_billionaires = len(wealth_sorted)
mean_wealth = np.mean(wealth_sorted)
var_wealth = np.var(wealth_sorted)
skew_wealth = skew(wealth_sorted)
kurt_wealth = kurtosis(wealth_sorted)

ranks = np.arange(1, N_billionaires + 1)

# --- LINEAR REGRESSION (THEORETICAL CURVE) ---
log_ranks = np.log10(ranks)
log_wealth = np.log10(wealth_sorted)

# Fit excluding extreme head noise (the top 10 richest individuals often deviate due to the 'King Effect')
mask_fit = (ranks >= 10) & (ranks <= N_billionaires * 0.8)
m, c = np.polyfit(log_ranks[mask_fit], log_wealth[mask_fit], 1)
theoretical_wealth = 10**(m * np.log10(ranks) + c)
b_value = -m 

print("\n--- DATA CHARACTERIZATION (PARETO - FORBES BILLIONAIRES) ---")
print(f"Total billionaires (N): {N_billionaires}")
print(f"Mean Wealth: {mean_wealth:.2f} Billion USD")
print(f"Variance (2nd moment): {var_wealth:.2f}")
print(f"Skewness (3rd moment): {skew_wealth:.4f}")
print(f"Kurtosis (4th moment): {kurt_wealth:.4f}")
print(f"Rank-Size Exponent (b-value): {b_value:.4f}")
print("-----------------------------------------------------------\n")

print("Generating plots...")
# 2. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Top 50 Billionaires)
top_n = 50
ax1.bar(ranks[:top_n], wealth_sorted[:top_n], color='skyblue', edgecolor='black', label='Forbes Data')
ax1.set_title(f'Wealth Distribution (Top {top_n} Billionaires)')
ax1.set_xlabel('Rank (1 = Richest)')
ax1.set_ylabel('Net Worth (Billions USD)')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Log-Log Scale (Pareto Principle)
ax2.scatter(ranks, wealth_sorted, color='red', s=15, alpha=0.8, label='Empirical Data', zorder=5)
ax2.plot(ranks, theoretical_wealth, color='green', linewidth=2.5, label=f'Regression (Slope={m:.2f})')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Pareto Principle in Global Wealth (Log-Log Scale)')
ax2.set_xlabel('Billionaire Rank')
ax2.set_ylabel('Net Worth (Billions USD)')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "forbes_pareto_plot.png" 
plt.savefig(output_path)
print(f"Plot saved at: {output_path}")
plt.show()