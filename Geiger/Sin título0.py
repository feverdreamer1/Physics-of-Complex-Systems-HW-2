import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, skew, kurtosis

# Number of particles detected in each interval (k)
k_values = np.arange(0, 15)

# Empirical data from Rutherford and Geiger (1910)
real_frequencies = np.array([57, 203, 383, 525, 532, 408, 273, 139, 45, 27, 10, 4, 0, 1, 1])
N_intervals = np.sum(real_frequencies) 

# Reconstruct raw data array for statistical moments
raw_data = np.repeat(k_values, real_frequencies)

# --- STATISTICAL CHARACTERIZATION ---
mean_det = np.mean(raw_data)
var_det = np.var(raw_data)
skew_det = skew(raw_data)
kurt_det = kurtosis(raw_data)

print("\n--- DATA CHARACTERIZATION (POISSON - RUTHERFORD 1910) ---")
print(f"Total time intervals (N): {N_intervals}")
print(f"Mean detections per 7.5s (lambda): {mean_det:.4f}")
print(f"Variance (2nd moment): {var_det:.4f}")
print(f"Skewness (3rd moment): {skew_det:.4f}")
print(f"Kurtosis (4th moment): {kurt_det:.4f}")
print("---------------------------------------------------------\n")

# --- THEORETICAL POISSON PMF ---
pmf_theoretical = poisson.pmf(k_values, mu=mean_det)

print("Generating plots...")
# Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

real_probabilities = real_frequencies / N_intervals

# Plot 1: Linear Scale
ax1.bar(k_values, real_probabilities, width=0.8, color='lightcoral', edgecolor='black', alpha=0.7, label='Rutherford Data (1910)')
ax1.plot(k_values, pmf_theoretical, color='green', marker='o', linestyle='-', linewidth=2, markersize=6, label=f'Poisson PMF ($\lambda$={mean_det:.2f})')
ax1.set_title('Alpha Particle Emissions (Linear Scale)')
ax1.set_xlabel('Number of Particles Detected ($k$)')
ax1.set_ylabel('Probability $P(k)$')
ax1.grid(axis='y', alpha=0.75)
ax1.set_xticks(k_values)
ax1.legend()

# Plot 2: Semi-Log Scale (Log-Y)
ax2.bar(k_values, real_probabilities, width=0.8, color='lightcoral', edgecolor='black', alpha=0.7, label='Empirical Data')
ax2.plot(k_values, pmf_theoretical, color='green', marker='o', linestyle='-', linewidth=2, markersize=6, label='Theoretical PMF')
ax2.set_yscale('log')
ax2.set_title('Alpha Particle Emissions (Log-Y Scale)')
ax2.set_xlabel('Number of Particles Detected ($k$)')
ax2.set_ylabel('Log Probability $P(k)$')
ax2.set_ylim(bottom=1e-5) 
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.set_xticks(k_values)
ax2.legend()

plt.tight_layout()
output_path = "rutherford_poisson.png"
plt.savefig(output_path)
print(f"Plot saved at: {output_path}")
plt.show()