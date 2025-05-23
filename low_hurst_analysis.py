import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the low Hurst dataset
low_hurst_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\low_hurst_ids_distances_with_properties.tsv"
low_hurst_data = pd.read_csv(low_hurst_file_path, sep='\t')

# Load the regular Hurst dataset
reg_hurst_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\reg_hurst_ids_distances_with_properties.tsv"
reg_hurst_data = pd.read_csv(reg_hurst_file_path, sep='\t')

# Select only the relevant columns: Protein ID and time points for both datasets
low_hurst_time_point_columns = [col for col in low_hurst_data.columns if "Time Point" in col]
reg_hurst_time_point_columns = [col for col in reg_hurst_data.columns if "Time Point" in col]

# Melt both DataFrames to long format for easier plotting
low_hurst_plot_data = low_hurst_data[['Protein ID'] + low_hurst_time_point_columns]
low_hurst_plot_data_melted = low_hurst_plot_data.melt(id_vars='Protein ID', var_name='Time Point', value_name='Distance from Centroid')

reg_hurst_plot_data = reg_hurst_data[['Protein ID'] + reg_hurst_time_point_columns]
reg_hurst_plot_data_melted = reg_hurst_plot_data.melt(id_vars='Protein ID', var_name='Time Point', value_name='Distance from Centroid')

# Plotting low Hurst proteins
plt.figure(figsize=(12, 6))
sns.lineplot(data=low_hurst_plot_data_melted, x='Time Point', y='Distance from Centroid', hue='Protein ID', markers=True, color='red', legend=False)
plt.title('Distance from Centroid Over Time for Low Hurst Proteins (<0.1)')
plt.xlabel('Time Point')
plt.ylabel('Distance from Centroid')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting regular Hurst proteins
plt.figure(figsize=(12, 6))
sns.lineplot(data=reg_hurst_plot_data_melted, x='Time Point', y='Distance from Centroid', hue='Protein ID', markers=True, color='gray', legend=False)
plt.title('Distance from Centroid Over Time for Regular Hurst Proteins (>0.1)')
plt.xlabel('Time Point')
plt.ylabel('Distance from Centroid')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
