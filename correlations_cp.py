import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
from sklearn.metrics import pairwise_kernels

# Read the data file
df = pd.read_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\combined_data_with_disorder_content.tsv", sep='\t')
df = df.dropna(subset=['hull_presence'])
df = df[df['sequence_length'] != 0]

# Calculate and display descriptive statistics for hull_presence
print("\nDescriptive Statistics for hull_presence:")
print(df['hull_presence'].describe())

# Properties to calculate correlations for
properties = [
    'density', 'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
    'inradius', 'circumradius', 'hydrodynamic_radius', 'sequence_length', 'hull_presence', 'disorder_content'
]

# Function to perform Min-Max scaling
def min_max_scaling(feature):
    min_val = min(feature)
    max_val = max(feature)
    scaled_feature = [(x - min_val) / (max_val - min_val) for x in feature]
    return scaled_feature

# Normalize the features
for prop in properties:
    df[prop] = min_max_scaling(df[prop])

# Remove outliers using z-score
z_scores = zscore(df[properties])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]



# Calculate and display Kernel Correlations for all properties
def kernel_correlation_matrix(df, kernel='rbf'):
    # Reshape the data to compute kernel correlations
    corr_matrix = pairwise_kernels(df, metric=kernel)
    return corr_matrix

# Calculate the Kernel Correlation Matrix
print("\nKernel Correlation Matrix:")
kernel_corr_matrix = kernel_correlation_matrix(df[properties], kernel='rbf')

# Plot the Kernel Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(kernel_corr_matrix, annot=True, cmap='coolwarm', xticklabels=properties, yticklabels=properties, center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title("Kernel Correlation Matrix (RBF Kernel)")
plt.show()

# Calculate and display Pearson Correlation Matrix for all properties
print("\nPearson Correlation Matrix:")
pearson_corr_matrix = df[properties].corr()

# Plot the Pearson Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', xticklabels=properties, yticklabels=properties, center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title("Pearson Correlation Matrix")
plt.show()

# Calculate and display Kendall's Tau Correlation Matrix for all properties
print("\nKendall's Tau Correlation Matrix:")
kendall_corr_matrix = df[properties].corr(method='kendall')

# Plot the Kendall's Tau Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(kendall_corr_matrix, annot=True, cmap='coolwarm', xticklabels=properties, yticklabels=properties, center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title("Kendall's Tau Correlation Matrix")
plt.show()

# Calculate and display Spearman Correlation Matrix for all properties
print("\nSpearman Correlation Matrix:")
spearman_corr_matrix = df[properties].corr(method='spearman')

# Plot the Spearman Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', xticklabels=properties, yticklabels=properties, center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title("Spearman Correlation Matrix")
plt.show()

'''
# Plot box plots to check the variance between each property and hull_presence
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df, x='hull_presence', y=prop)
    plt.title(f'Variance of {prop.capitalize()} by hull_presence')
    plt.xlabel('hull_presence')
    plt.ylabel(prop.capitalize())

plt.tight_layout()
plt.show()

# Plot scatter plots with trend lines
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    sns.regplot(data=df, x=prop, y='hull_presence', line_kws={"color":"red"})
    plt.title(f'{prop.capitalize()} vs hull_presence')

plt.tight_layout()
plt.show()

# Plot scatter plots with monotonic trend lines
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    lowess_sm = sm.nonparametric.lowess(df['hull_presence'], df[prop])
    plt.plot(lowess_sm[:, 0], lowess_sm[:, 1], color='red')
    plt.scatter(df[prop], df['hull_presence'], alpha=0.5)
    plt.title(f'{prop.capitalize()} vs hull_presence')
    plt.xlabel(prop.capitalize())
    plt.ylabel('hull_presence')

plt.tight_layout()
plt.show()

# Plot scatter plots with Lagrange polynomial interpolation
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    x = df[prop].values
    y = df['hull_presence'].values

    # Lagrange interpolation
    poly = Polynomial.fit(x, y, deg=3)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    plt.scatter(x, y, alpha=0.5, label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Lagrange Polynomial Fit')
    plt.title(f'{prop.capitalize()} vs hull_presence')
    plt.xlabel(prop.capitalize())
    plt.ylabel('hull_presence')
    plt.legend()

plt.tight_layout()
plt.show()



# Plot distribution of each variable
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties + ['hull_presence'], 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[prop], kde=True)
    plt.title(f'Distribution of {prop.capitalize()}')
    plt.xlabel(prop.capitalize())
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
'''
