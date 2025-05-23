import numpy as np
import pandas as pd
import ast  # Library for literal_eval function

# Read the TSV file
df = pd.read_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\lp_data_hurst_exponent.tsv", sep="\t")

# Drop the specified columns
df = df.drop(columns=['index', 'euler_characteristic'])

# Write the modified DataFrame to a new TSV file
df.to_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\lp_data_he_for_model.tsv", sep="\t", index=False)

print("File has been created.")
