import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.setrecursionlimit(5000)

# Load and prepare data
df = pd.read_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\LA_all_timepoints_dynamic_hull_id_040125_NCA_backbone_with_predictions.tsv", sep="\t")
df = df.set_index('uniprot_id')

# Select structural property columns
structural_cols = [
    'density', 'radius_of_gyration', 'surface_area_to_volume_ratio',
    'sphericity', 'inradius', 'circumradius',
    'hydrodynamic_radius', 'avg_plddt', 'predicted_disorder_content', 'hull_presence'
]

# Combine into one heatmap input and fill missing values
heatmap_df = df[structural_cols].fillna(0)

# Filter out rows where density = 0
heatmap_df = heatmap_df[heatmap_df['density'] != 0]

# Sort by hull_presence in descending order (highest values at the top)
heatmap_df = heatmap_df.sort_values('hull_presence', ascending=False)

# Normalize data (Min-Max scaling between 0 and 1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(heatmap_df)
normalized_df = pd.DataFrame(normalized_data, columns=heatmap_df.columns, index=heatmap_df.index)

# Plot heatmap
sns.set(style="white")
plt.figure(figsize=(16, 10))
g = sns.heatmap(
    normalized_df,
    cmap="YlGnBu",
    xticklabels=True,
    yticklabels=False,
    cbar_kws={"label": "Normalized Value", "location": "left", "fraction": 0.18}
)

plt.title("Protein Hull Presence & Structural Properties\n(Sorted by Hull Presence, Highest at Top)", y=1.03)
plt.tight_layout()
plt.show()
