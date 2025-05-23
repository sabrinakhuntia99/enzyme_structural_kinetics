import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the training dataset
pdb_data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_data_pdb.tsv"

# Load the training dataset
pdb_df = pd.read_csv(pdb_data_path, sep='\t')

# Drop unnecessary columns
drop_columns = ['uniprot_id', 'hydrophobicity_full', 'euler_characteristic']
pdb_df.drop(columns=drop_columns, inplace=True)

# Plotting distribution of disorder content for the training dataset
plt.figure(figsize=(14, 6))
sns.histplot(pdb_df['disorder_content'], bins=30, kde=True)
plt.title('Distribution of Disorder Content (PDB Structures of DisProt Proteins)')
plt.xlabel('Disorder Content')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Separating the features (x) from the output: disorder_content (y)
pdb_x = pdb_df.iloc[:, :-1]  # All columns except the last one
pdb_y = pdb_df.iloc[:, -1]   # Last column (disorder_content)

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

# Train and evaluate the Gradient Boosting model
pdb_gb_model = train_and_evaluate_gb_model(pdb_x, pdb_y, "PDB")

# ----- SHAP Explanation Code -----
import shap
#shap.initjs()

# Select a representative sample from the training data for the explanation
sample_data = pdb_x.sample(100, random_state=42)

# Create a TreeExplainer for the Gradient Boosting model
explainer = shap.TreeExplainer(pdb_gb_model)

# Compute SHAP values for the sample data
shap_values = explainer.shap_values(sample_data)

# Plot the SHAP summary plot to display feature importance
shap.summary_plot(shap_values, sample_data, feature_names=pdb_x.columns)
# ----------------------------------

