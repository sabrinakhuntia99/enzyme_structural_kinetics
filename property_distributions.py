import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the TSV file
file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\dynamic_hull_021025_N_backbone.tsv" # Update with the correct path to your TSV file
df = pd.read_csv(file_path, sep='\t')
# Remove rows where 'density' is 0
df = df[df['density'] != 0]
df = df[df['hull_presence'] != 0]

# List of variables to plot histograms for
variables = [
    'density', 'radius_of_gyration', 'surface_area_to_volume_ratio', 'sphericity',
     'inradius', 'circumradius', 'hydrodynamic_radius',
    'sequence_length', 'avg_plddt', 'hull_presence'
]

# Set up the plot
plt.figure(figsize=(15, 10))

# Plot each histogram
for i, var in enumerate(variables, start=1):
    plt.subplot(3, 4, i)  # Grid of 3 rows, 4 columns
    sns.histplot(df[var], kde=True, bins=20)  # KDE overlayed on histogram
    plt.title(f"Histogram of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")

# Adjust layout
plt.tight_layout()
plt.show()
