import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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

# Function to train Polynomial Regression and evaluate the model
def train_and_evaluate_polynomial_model(x, y, degree=2, dataset_name="Dataset"):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    # Train linear regression on polynomial features
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_test_poly)
    print(f"{dataset_name} Polynomial Regression (Degree {degree}):")
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    return model, poly

# Train and evaluate the Polynomial Regression model
pdb_poly_model, poly_transformer = train_and_evaluate_polynomial_model(pdb_x, pdb_y, degree=2, dataset_name="PDB")
