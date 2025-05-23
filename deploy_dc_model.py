import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Path to the training dataset
pdb_data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_data_pdb.tsv"

# Load the training dataset
pdb_df = pd.read_csv(pdb_data_path, sep='\t')

# Drop unnecessary columns
drop_columns = ['uniprot_id', 'hydrophobicity_full', 'euler_characteristic']
pdb_df.drop(columns=drop_columns, inplace=True)

# Correlation matrix heatmap
plt.figure(figsize=(14, 12))
corr_matrix = pdb_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap of All Features and Disorder Content", fontsize=16)
plt.tight_layout()
plt.show()

# Plot distribution of disorder content
plt.figure(figsize=(14, 6))
sns.histplot(pdb_df['disorder_content'], bins=30, kde=True)
plt.title('Distribution of Disorder Content (PDB Structures of DisProt Proteins)')
plt.xlabel('Disorder Content')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Separate features and target
pdb_x = pdb_df.iloc[:, :-1]
pdb_y = pdb_df.iloc[:, -1]

# Cross-validation training and evaluation
def train_and_evaluate_gb_model_cv(x, y, dataset_name="Dataset", cv_splits=5):
    regressor = GradientBoostingRegressor(n_estimators=150, max_depth=9, learning_rate=0.1)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    r2_scores = cross_val_score(regressor, x, y, cv=cv, scoring='r2')
    print(f"{dataset_name} Gradient Boosting CV R^2 scores: {r2_scores}")
    print(f"Mean R^2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    y_pred_cv = cross_val_predict(regressor, x, y, cv=cv)
    mse = mean_squared_error(y, y_pred_cv)
    print(f"{dataset_name} Gradient Boosting CV Mean Squared Error: {mse:.4f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(y, y_pred_cv, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual disorder_content')
    plt.ylabel('Predicted disorder_content (CV)')
    plt.title(f'{dataset_name} Gradient Boosting: Actual vs Predicted (CV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    regressor.fit(x, y)
    return regressor

# Train model with CV
pdb_gb_model = train_and_evaluate_gb_model_cv(pdb_x, pdb_y, "PDB")

# Load and clean new dataset
new_data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\LA_all_timepoints_dynamic_hull_id_040125_NCA_backbone.tsv"
new_df = pd.read_csv(new_data_path, sep='\t')
new_df = new_df[new_df['density'] != 0]

new_df.drop(columns=[
    'euler_characteristic', 'hull_presence', 'avg_plddt',
    'hull_presence_tryps_0010min', 'hull_presence_tryps_0015min',
    'hull_presence_tryps_0020min', 'hull_presence_tryps_0030min',
    'hull_presence_tryps_0040min', 'hull_presence_tryps_0050min',
    'hull_presence_tryps_0060min', 'hull_presence_tryps_0120min',
    'hull_presence_tryps_0180min', 'hull_presence_tryps_0240min',
    'hull_presence_tryps_1440min', 'hull_presence_tryps_leftover'
], inplace=True)

# Match new dataset columns with training features
new_df = new_df[pdb_x.columns]

# Predict and clip
new_predictions = pdb_gb_model.predict(new_df)
new_predictions = np.clip(new_predictions, 0, 1)
new_df['predicted_disorder_content'] = new_predictions

# Show sample predictions
print(new_df.head())

# SHAP analysis
explainer = shap.TreeExplainer(pdb_gb_model)
shap_values = explainer.shap_values(pdb_x)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, pdb_x, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP Values)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, pdb_x, show=False)
plt.title("Feature Impact on Disorder Content")
plt.tight_layout()
plt.show()
