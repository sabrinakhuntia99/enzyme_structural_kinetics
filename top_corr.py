import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import rbf_kernel

# Load data
df = pd.read_csv(r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\combined_data.tsv', sep='\t')

# Filter out rows where hull_presence equals 0
df = df[df['hull_presence'] != 0]


# Prepare numeric data
numeric_df = df.drop(columns=['uniprot_id'])
target_var = 'hull_presence'

# Calculate correlations with hull_presence
correlation_results = pd.DataFrame(index=numeric_df.columns, columns=['Pearson', 'Spearman', 'Kernel'])

for var in numeric_df.columns:
    if var != target_var:
        # Pearson
        correlation_results.at[var, 'Pearson'] = pearsonr(numeric_df[target_var], numeric_df[var])[0]
        # Spearman
        correlation_results.at[var, 'Spearman'] = spearmanr(numeric_df[target_var], numeric_df[var])[0]

# Kernel correlation (needs standardization)
df_std = (numeric_df - numeric_df.mean()) / numeric_df.std()
kernel_sim = rbf_kernel(df_std.T)
np.fill_diagonal(kernel_sim, 1)
kernel_sim = pd.DataFrame(kernel_sim, index=numeric_df.columns, columns=numeric_df.columns)
correlation_results['Kernel'] = kernel_sim[target_var]

# Drop hull_presence row (self-correlation)
correlation_results = correlation_results.drop(target_var)

# Print results
print("="*80)
print("CORRELATIONS WITH HULL_PRESENCE".center(80))
print("="*80)
print(correlation_results.sort_values('Pearson', ascending=False).round(2).to_string())

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_results.sort_values('Pearson', ascending=False),
            annot=True, fmt=".2f", cmap='coolwarm',
            center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation'})
plt.title(f'Correlations with {target_var}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()