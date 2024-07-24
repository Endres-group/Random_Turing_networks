import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# CSV data from simulation from figure8_HS_v3.py
output_dir = './'
csv_path = os.path.join(output_dir, 'heatmap_simulation.csv')

# Load the CSV file
data = pd.read_csv(csv_path)

# Filter rows where log10(Dy) is approximately equal to -log10(Dx)
data['log10_Dx'] = np.log10(data['Dx'])
data['neg_log10_Dx'] = -data['log10_Dx']
filtered_data = data[np.isclose(np.log10(data['Dy']), data['neg_log10_Dx'])]

# Define new bins for log10(Dx)
bins_log = np.linspace(filtered_data['log10_Dx'].min(), filtered_data['log10_Dx'].max(), num=20)

# Assign bins to the data
filtered_data['log10_Dx_bin'] = pd.cut(filtered_data['log10_Dx'], bins_log, labels=bins_log[:-1], include_lowest=True)

# Calculate the mean and standard error of Percentage_Turing for each bin of log10(Dx) grouped by N
summary_stats_logDx_bin = filtered_data.groupby(['N', 'log10_Dx_bin'])['Percentage_Turing'].agg(['mean', 'sem']).reset_index()
summary_stats_logDx_bin['log10_Dx_bin'] = summary_stats_logDx_bin['log10_Dx_bin'].astype(float)

# Calculate the mean and standard error of Percentage_Turing_1 for each bin of log10(Dx) grouped by N
summary_stats_logDx_bin_1 = filtered_data.groupby(['N', 'log10_Dx_bin'])['Percentage_Turing_1'].agg(['mean', 'sem']).reset_index()
summary_stats_logDx_bin_1['log10_Dx_bin'] = summary_stats_logDx_bin_1['log10_Dx_bin'].astype(float)

# Plot the data for Percentage_Turing and Percentage_Turing_1
plt.figure(figsize=(14, 10))

# Plot for Percentage Turing Instability
plt.subplot(2, 1, 1)
for n in summary_stats_logDx_bin['N'].unique():
    subset = summary_stats_logDx_bin[summary_stats_logDx_bin['N'] == n]
    plt.errorbar(subset['log10_Dx_bin'], subset['mean'], yerr=subset['sem'], fmt='o-', capsize=5, capthick=2, label=f'N = {n}')
plt.xlabel(r'$\log_{10}(D_1)$')
plt.ylabel(r'Percentage Turing Instability')
plt.title(r'Turing Instability at $\log_{10}(D_2)=-\log_{10}(D_1)$')
plt.xticks(ticks=np.arange(filtered_data['log10_Dx'].min(), filtered_data['log10_Dx'].max() + 1, 1))
plt.legend()
plt.grid(True, which="both", ls="--")

# Plot for Percentage Turing Instability Type 1
plt.subplot(2, 1, 2)
for n in summary_stats_logDx_bin_1['N'].unique():
    subset = summary_stats_logDx_bin_1[summary_stats_logDx_bin_1['N'] == n]
    plt.errorbar(subset['log10_Dx_bin'], subset['mean'], yerr=subset['sem'], fmt='o-', capsize=5, capthick=2, label=f'N = {n}')
plt.xlabel(r'$\log_{10}(D_1)$')
plt.ylabel(r'Percentage Turing Instability Type 1')
plt.title(r'Turing Instability Type 1 at $\log_{10}(D_2)=-\log_{10}(D_1)$')
plt.xticks(ticks=np.arange(filtered_data['log10_Dx'].min(), filtered_data['log10_Dx'].max() + 1, 1))
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('profile_plot.png')
plt.show()