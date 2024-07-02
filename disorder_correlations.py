import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import statsmodels.api as sm

# Read the data file
df = pd.read_csv('predicted_disorder_content.tsv', sep='\t')

# Properties to calculate correlations for
properties = ['density', 'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
              'euler_characteristic', 'inradius', 'circumradius', 'hydrodynamic_radius', 'sequence_length']

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

# Calculate and display the Pearson correlations
print("Pearson Correlations:")
for prop in properties:
    corr = df[prop].corr(df['Predicted_Disorder_Content_GB'])
    print(f"Correlation between {prop.capitalize()} and Predicted_Disorder_Content_GB: {corr}")

# Calculate and display the Spearman correlations
print("\nSpearman Correlations:")
for prop in properties:
    corr_spearman = df[prop].corr(df['Predicted_Disorder_Content_GB'], method='spearman')
    print(f"Spearman Correlation between {prop.capitalize()} and Predicted_Disorder_Content_GB: {corr_spearman}")

# Plot scatter plots with trend lines
plt.figure(figsize=(18, 15))
for i, prop in enumerate(properties, 1):
    plt.subplot(3, 3, i)
    sns.regplot(data=df, x=prop, y='Predicted_Disorder_Content_GB', line_kws={"color":"red"})
    plt.title(f'{prop.capitalize()} vs Predicted_Disorder_Content_GB')

plt.tight_layout()
plt.show()

# Plot scatter plots with monotonic trend lines
plt.figure(figsize=(18, 15))
for i, prop in enumerate(properties, 1):
    plt.subplot(3, 3, i)
    lowess_sm = sm.nonparametric.lowess(df['Predicted_Disorder_Content_GB'], df[prop])
    plt.plot(lowess_sm[:, 0], lowess_sm[:, 1], color='red')
    plt.scatter(df[prop], df['Predicted_Disorder_Content_GB'], alpha=0.5)
    plt.title(f'{prop.capitalize()} vs Predicted_Disorder_Content_GB')
    plt.xlabel(prop.capitalize())
    plt.ylabel('Predicted_Disorder_Content_GB')

plt.tight_layout()
plt.show()