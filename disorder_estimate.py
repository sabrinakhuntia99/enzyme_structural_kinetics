import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load your original dataset (df) used for training
data_path = 'updated_disorder_data_060324.tsv'
df = pd.read_csv(data_path, sep='\t')

# Define a function to convert 'hydrophobicity_full' to a list of floats
def convert_to_float_list(x):
    try:
        return [float(item) for item in x.strip('[]').split()]
    except ValueError:
        return None  # Handle cases where conversion fails

# Convert 'hydrophobicity_full' column in df
df['hydrophobicity_full'] = df['hydrophobicity_full'].apply(convert_to_float_list)

# Drop 'hydrophobicity_full' column from df
df.drop(columns=['hydrophobicity_full'], inplace=True)

# Separate features (x) and output (y) from the original dataset (df)
x = df.drop(columns=['disorder_content'])
y = df['disorder_content']

# Split data into training and testing sets (not necessary for prediction on exp_data)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# DECISION TREE REGRESSOR
regressor1 = DecisionTreeRegressor(max_depth=14)
regressor1.fit(x, y)

# RANDOM FOREST REGRESSOR
regressor2 = RandomForestRegressor(n_estimators=150, max_depth=20)
regressor2.fit(x, y)

# GRADIENT BOOSTING REGRESSOR
regressor3 = GradientBoostingRegressor(n_estimators=150, max_depth=9, learning_rate=0.1)
regressor3.fit(x, y)

# Now load and preprocess the new dataset (exp_data)
exp_data_path = 'pep_cleave_measures.tsv'
exp_data = pd.read_csv(exp_data_path, sep='\t')

# Convert 'hydrophobicity_full' column in exp_data
exp_data['hydrophobicity_full'] = exp_data['hydrophobicity_full'].apply(convert_to_float_list)

# Drop 'hydrophobicity_full' column from exp_data (if present)
exp_data.drop(columns=['hydrophobicity_full'], inplace=True, errors='ignore')

# Use the trained models to predict 'disorder_content' in exp_data
y_pred1 = regressor1.predict(exp_data)
y_pred2 = regressor2.predict(exp_data)
y_pred3 = regressor3.predict(exp_data)

exp_data['Predicted_Disorder_Content_DT'] = y_pred1
exp_data['Predicted_Disorder_Content_RF'] = y_pred2
exp_data['Predicted_Disorder_Content_GB'] = y_pred3



# Save the predicted data to a new TSV file
exp_data.to_csv('predicted_disorder_content.tsv', sep='\t', index=False)

print("Predictions saved to predicted_disorder_content.tsv")
