import numpy as np
import pandas as pd
import ast  # Library for literal_eval function


# Read the TSV file
df = pd.read_csv("processed_disprot_new.tsv", sep="\t")

# Define a function to calculate min, max, sum, and average of an array
def calculate_stats(arr):
    return np.min(arr), np.max(arr), np.sum(arr), np.mean(arr)


# Convert string representations of arrays into actual arrays of numerical values
df['hydrophobicity_full'] = df['hydrophobicity_full'].apply(lambda x: np.array(ast.literal_eval(x)))
df['hydrophobicity_region'] = df['hydrophobicity_region'].apply(lambda x: np.array(ast.literal_eval(x)))

# Apply the operation to hydrophobicity_full column
df['hydrophobicity_full'] = df['hydrophobicity_full'].apply(lambda arr: np.array(calculate_stats(arr)))

# Apply the operation to hydrophobicity_region column
df['hydrophobicity_region'] = df['hydrophobicity_region'].apply(lambda arr: np.array(calculate_stats(arr)))

# Write the modified DataFrame to a new TSV file
df.to_csv("disorder_data_60324.tsv", sep="\t", index=False)

print("File 'disorder_data_60324.tsv' has been created.")