import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Paths for datasets
pdb_data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_data_pdb.tsv"
af_data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_data_with_measures_for_model.tsv"

# Load both datasets
pdb_df = pd.read_csv(pdb_data_path, sep='\t')
af_df = pd.read_csv(af_data_path, sep='\t')

# Drop unnecessary columns
drop_columns = ['uniprot_id', 'hydrophobicity_full', 'euler_characteristic']
pdb_df.drop(columns=drop_columns, inplace=True)

# Plotting distribution of disorder content for both datasets
plt.figure(figsize=(14, 6))

# PDB dataset plot
plt.subplot(1, 2, 1)
sns.histplot(pdb_df['disorder_content'], bins=30, kde=True)
plt.title('Distribution of Disorder Content (PDB)')
plt.xlabel('Disorder Content')
plt.ylabel('Frequency')
plt.tight_layout()

# AlphaFold dataset plot
plt.subplot(1, 2, 2)
sns.histplot(af_df['disorder_content'], bins=30, kde=True)
plt.title('Distribution of Disorder Content (AlphaFold)')
plt.xlabel('Disorder Content')
plt.ylabel('Frequency')
plt.tight_layout()

plt.show()

# Separating the features (x) from the output: disorder_content (y) for both datasets
pdb_x = pdb_df.iloc[:, :-1]
pdb_y = pdb_df.iloc[:, -1]

af_x = af_df.iloc[:, :-1]
af_y = af_df.iloc[:, -1]

# Function to train Gradient Boosting Regressor and evaluate the model
def train_and_evaluate_gb_model(x, y, dataset_name="Dataset"):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    regressor = GradientBoostingRegressor(n_estimators=150, max_depth=9, learning_rate=0.1)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print(f"{dataset_name} Gradient Boosting:")
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    return regressor

# Train and evaluate models for both datasets
pdb_gb_model = train_and_evaluate_gb_model(pdb_x, pdb_y, "PDB")
af_gb_model = train_and_evaluate_gb_model(af_x, af_y, "AlphaFold")

# Function to evaluate model with a dropped column
def evaluate_model_with_dropped_column(df, y, column_to_drop):
    df_modified = df.drop(columns=[column_to_drop])
    x_train, x_test, y_train, y_test = train_test_split(df_modified, y, test_size=0.2, random_state=42)
    regressor = GradientBoostingRegressor(n_estimators=150, max_depth=9, learning_rate=0.1)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    return r2_score(y_test, y_pred)

# Evaluate each feature for both datasets
def evaluate_features(df, y, dataset_name="Dataset"):
    results = {}
    for column in df.columns:
        r2_score_dropped = evaluate_model_with_dropped_column(df, y, column)
        results[column] = r2_score_dropped
        print(f"R^2 Score of GB model ({dataset_name}) after dropping '{column}': {r2_score_dropped:.4f}")
    return results

# Get R² scores for feature evaluation
pdb_results = evaluate_features(pdb_x, pdb_y, "PDB")
af_results = evaluate_features(af_x, af_y, "AlphaFold")

# Visualization of R² Scores when dropping each feature for both datasets
# Convert the results dictionaries to DataFrames
pdb_results_df = pd.DataFrame(list(pdb_results.items()), columns=['Feature', 'PDB'])
af_results_df = pd.DataFrame(list(af_results.items()), columns=['Feature', 'AlphaFold'])

# Merge the two DataFrames for side-by-side comparison
comparison_df = pd.merge(pdb_results_df, af_results_df, on='Feature')

# Plot the R² scores as a grouped vertical bar chart
plt.figure(figsize=(16, 10))
comparison_df_melted = pd.melt(comparison_df, id_vars=['Feature'], value_vars=['PDB', 'AlphaFold'],
                               var_name='Dataset', value_name='R² Score')

# Define custom colors for each dataset
palette = {'PDB': 'green', 'AlphaFold': 'purple'}

# Use sns.barplot with vertical orientation and custom colors
sns.barplot(x='Feature', y='R² Score', hue='Dataset', data=comparison_df_melted, palette=palette)
plt.title('Impact of Feature Removal on R² Scores (Gradient Boosting Model): PDB vs. AlphaFold', fontsize=16)
plt.xlabel('Structural Feature', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate feature labels for better readability
plt.legend(title='Structures')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('comparison_feature_importance_r2_scores_vertical.png', dpi=300)
plt.show()
