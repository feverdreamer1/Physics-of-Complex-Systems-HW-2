import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# 1. Download Don Quixote text
url = "https://www.gutenberg.org/files/2000/2000-0.txt"
print("Downloading the book...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    text = response.read().decode('utf-8')

print("Processing the text...")
# 2. Tokenization 
words = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())
count = Counter(words)
frequencies = sorted(list(count.values()), reverse=True)
ranks = np.arange(1, len(frequencies) + 1)

# --- STATISTICAL CHARACTERIZATION ---
freq_array = np.array(frequencies)
N_words = len(freq_array)
mean_freq = np.mean(freq_array)
var_freq = np.var(freq_array)
skew_freq = skew(freq_array)
kurt_freq = kurtosis(freq_array)

# --- LINEAR REGRESSION (Zipf's Exponent) ---
fit_limit = min(10000, len(ranks))
ranks_fit = ranks[:fit_limit]
freq_fit = frequencies[:fit_limit]

log_ranks = np.log10(ranks_fit)
log_freq = np.log10(freq_fit)
m, c = np.polyfit(log_ranks, log_freq, 1)

theoretical_freq = 10**(m * np.log10(ranks) + c)
b_value = -m 

print("\n--- DATA CHARACTERIZATION (ZIPF'S LAW) ---")
print(f"Total unique words (N): {N_words}")
print(f"Mean frequency: {mean_freq:.4f}")
print(f"Variance (2nd moment): {var_freq:.4f}")
print(f"Skewness (3rd moment): {skew_freq:.4f}")
print(f"Kurtosis (4th moment): {kurt_freq:.4f}")
print(f"Power Law Exponent (b-value): {b_value:.4f}")
print("-------------------------------------------\n")

print("Generating plots...")
# 3. Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear Scale (Fit removed for clarity)
top_n = 50
ax1.bar(ranks[:top_n], frequencies[:top_n], color='skyblue', edgecolor='black', label='Data')
ax1.set_title(f'Frequency vs Rank (Top {top_n} words)')
ax1.set_ylim(0, 25000) 
ax1.set_xlabel('Word rank (1 = most frequent)')
ax1.set_ylabel('Absolute frequency')
ax1.grid(axis='y', alpha=0.75)
ax1.legend()

# Plot 2: Log-Log Scale (Zipf's Law)
ax2.scatter(ranks, frequencies, color='red', s=10, alpha=0.5, label='Data', zorder=5)
ax2.plot(ranks, theoretical_freq, color='green', linewidth=2.5, label=f'Regression (b={b_value:.2f})')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("Zipf's Law (Log-Log Scale)")
ax2.set_xlabel('Word rank')
ax2.set_ylabel('Absolute frequency')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_path = "zipf_quixote_plot.png" 
plt.savefig(output_path)
plt.show()