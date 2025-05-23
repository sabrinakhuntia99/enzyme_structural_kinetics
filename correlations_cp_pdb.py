import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
import re

# Read the data file
df = pd.read_csv(
    r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\LA_all_timepoints_fixed_10_hull_layers_NCA_backbone_with_predictions.tsv',
    sep='\t'
)

# List of known disordered protein UniProt IDs
disordered_ids = ['P04637', 'P37840', 'P10636', 'P46527', 'P38398']

# Scan the uniprot_id column for the known disordered proteins
df_disordered = df[df['uniprot_id'].isin(disordered_ids)]

# Check if any rows were found
if df_disordered.empty:
    print("No known disordered proteins were found in the dataset.")
else:
    # Print descriptive statistics for the variables of the filtered proteins
    print("Descriptive statistics for known disordered proteins:")
    print(df_disordered.describe())

# Properties to calculate correlations for
properties = [
    'density', 'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
    'inradius', 'circumradius', 'hydrodynamic_radius',
    'sequence_length', 'hull_presence', 'predicted_disorder_content',     'avg_plddt',
    'hull_presence_tryps_0010min',
    'hull_presence_tryps_0015min',
    'hull_presence_tryps_0020min',
    'hull_presence_tryps_0030min',
    'hull_presence_tryps_0040min',
    'hull_presence_tryps_0050min',
    'hull_presence_tryps_0060min',
    'hull_presence_tryps_0120min',
    'hull_presence_tryps_0180min',
    'hull_presence_tryps_0240min',
    'hull_presence_tryps_1440min',
    'hull_presence_tryps_leftover'
]

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
df[properties] = scaler.fit_transform(df[properties])

# Print number of rows before removing zeroes
print(f"Number of rows before filtering: {len(df)}")

# Remove rows where density = 0
df_filtered = df[df['density'] != 0]
#df_filtered = df_filtered[df_filtered['predicted_disorder_content'] <= 0.3]

# Print number of rows after removing zeroes
print(f"Number of rows after filtering {len(df_filtered)}")

# Check if the filtered DataFrame is empty
if df_filtered.empty:
    raise ValueError("All rows were removed because density = 0. Check the data.")

# Function to compute kernel correlation matrix
def kernel_correlation_matrix(data, gamma=1.0):
    """
    Compute the kernel correlation matrix using the RBF kernel.

    Parameters:
        data (pd.DataFrame): Input data with features as columns.
        gamma (float): Parameter for the RBF kernel.

    Returns:
        kernel_corr (np.ndarray): Kernel correlation matrix.
    """
    # Compute the RBF kernel matrix
    kernel_matrix = rbf_kernel(data, gamma=gamma)

    # Center the kernel matrix
    n = kernel_matrix.shape[0]
    one_n = np.ones((n, n)) / n
    kernel_matrix_centered = kernel_matrix - one_n @ kernel_matrix - kernel_matrix @ one_n + one_n @ kernel_matrix @ one_n

    # Compute the kernel correlation matrix
    kernel_corr = np.corrcoef(kernel_matrix_centered)

    return kernel_corr


# Compute the kernel correlation matrix
kernel_corr = kernel_correlation_matrix(df_filtered[properties], gamma=1.0)

# Print the kernel correlation values in a readable format
print("Kernel Correlation Values:")
for i, prop1 in enumerate(properties):
    for j, prop2 in enumerate(properties):
        if i < j:  # Avoid duplicate pairs and self-correlations
            print(f"{prop1} and {prop2} = {kernel_corr[i, j]:.4f}")

plt.figure(figsize=(20, 18))
sns.set(font_scale=0.8)

# Cluster similar properties together
g = sns.clustermap(kernel_corr,
                   cmap='coolwarm',
                   center=0,
                   annot=False,
                   figsize=(20, 18),
                   dendrogram_ratio=0.1,
                   cbar_pos=(0.02, 0.8, 0.03, 0.18),
                   vmin=-1, vmax=1)

# Rotate labels and adjust layout
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
g.ax_heatmap.set_title("Kernel Correlation Matrix (Clustered)", pad=40)
plt.tight_layout()
plt.show()


from sklearn.decomposition import PCA

# Reduce to 3D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_filtered[properties])

# Interactive 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2],
                c=df_filtered['predicted_disorder_content'],
                cmap='plasma',
                s=50,
                alpha=0.7)

plt.colorbar(sc, label='Disorder Content')
ax.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
ax.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
ax.set_zlabel('PC3 ({:.1f}%)'.format(pca.explained_variance_ratio_[2]*100))
plt.title("3D PCA Projection Colored by Molecular Disorder Ratio")
plt.show()