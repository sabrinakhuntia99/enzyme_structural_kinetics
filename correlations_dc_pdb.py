import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import statsmodels.api as sm
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_kernels

# Read the data file

df = pd.read_csv(r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\lp_disorder_predictions_output_pdb_model.tsv', sep='\t')


# Properties to calculate correlations for
properties = [
    'density', 'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
    'inradius', 'circumradius', 'hydrodynamic_radius',
    'sequence_length'
]

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
df[properties] = scaler.fit_transform(df[properties])

# Remove outliers based on Z-score
threshold = 3  # Define a threshold for identifying outliers
z_scores = np.abs(zscore(df[properties]))
df = df[(z_scores < threshold).all(axis=1)]

# Plot the distribution of disorder_content
plt.figure(figsize=(10, 6))
sns.histplot(df['disorder_content'], bins=30, kde=True)
plt.title('Distribution of Disorder Content for Human Proteins (PDB Model Predictions)', fontsize=16)
plt.xlabel('Disorder Content', fontsize=20)  # Double the default size
plt.ylabel('Frequency', fontsize=20)          # Double the default size
plt.grid()
plt.show()

# Calculate and display the Pearson correlations
print("Pearson Correlations:")
for prop in properties:
    corr = df[prop].corr(df['disorder_content'])
    print(f"{prop.capitalize()} and disorder_content: {corr}")

# Calculate and display the Spearman correlations
print("\nSpearman Correlations:")
for prop in properties:
    corr_spearman = df[prop].corr(df['disorder_content'], method='spearman')
    print(f"{prop.capitalize()} and disorder_content: {corr_spearman}")

# Calculate and display Kernel Correlations
def kernel_correlation(x, y, kernel='linear'):
    # Reshape x and y to 2D arrays for kernel calculation
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    # Calculate the kernel matrix
    K_x = pairwise_kernels(x, metric=kernel)
    K_y = pairwise_kernels(y, metric=kernel)

    # Calculate the kernel correlation
    numerator = np.mean(K_x * K_y)
    denominator = np.sqrt(np.mean(K_x) * np.mean(K_y))
    return numerator / denominator

print("\nKernel Correlations:")
for prop in properties:
    corr_kernel = kernel_correlation(df[prop], df['disorder_content'], kernel='rbf')
    print(f"{prop.capitalize()} and disorder_content: {corr_kernel}")


# Uncomment the following blocks to create additional plots with adjusted axis titles

'''
# Plot scatter plots with trend lines
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    sns.regplot(data=df, x=prop, y='disorder_content', line_kws={"color":"red"})
    plt.title(f'{prop.capitalize()} vs disorder_content', fontsize=16)
    plt.xlabel(prop.capitalize(), fontsize=14)  # Double the default size
    plt.ylabel('Disorder Content', fontsize=14)  # Double the default size

plt.tight_layout()
plt.show()

# Plot scatter plots with monotonic trend lines
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    lowess_sm = sm.nonparametric.lowess(df['disorder_content'], df[prop])
    plt.plot(lowess_sm[:, 0], lowess_sm[:, 1], color='red')
    plt.scatter(df[prop], df['disorder_content'], alpha=0.5)
    plt.title(f'{prop.capitalize()} vs disorder_content', fontsize=16)
    plt.xlabel(prop.capitalize(), fontsize=14)  # Double the default size
    plt.ylabel('Disorder Content', fontsize=14)  # Double the default size

plt.tight_layout()
plt.show()

# Plot scatter plots with Lagrange polynomial interpolation
plt.figure(figsize=(18, 20))
for i, prop in enumerate(properties, 1):
    plt.subplot(4, 3, i)
    x = df[prop].values
    y = df['disorder_content'].values

    # Lagrange interpolation
    poly = Polynomial.fit(x, y, deg=3)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    plt.scatter(x, y, alpha=0.5, label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Lagrange Polynomial Fit')
    plt.title(f'{prop.capitalize()} vs disorder_content', fontsize=16)
    plt.xlabel(prop.capitalize(), fontsize=14)  # Double the default size
    plt.ylabel('Disorder Content', fontsize=14)  # Double the default size
    plt.legend()

plt.tight_layout()
plt.show()
'''
