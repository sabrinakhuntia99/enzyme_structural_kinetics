import numpy as np
import pandas as pd
import ast  # Library for literal_eval function

# Read the TSV file
df = pd.read_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_with_measures.tsv", sep="\t")

# Drop the specified columns
df = df.drop(columns=['uniprot_id', 'euler_characteristic'])

# Function to convert string representation of lists to actual lists
def convert_to_list(string):
    try:
        # Safely evaluate the string representation of the list
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return []  # Return an empty list if conversion fails

# Apply the conversion function to the hydrophobicity_full column
df['hydrophobicity_full'] = df['hydrophobicity_full'].apply(convert_to_list)

# Write the modified DataFrame to a new TSV file
df.to_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_data_with_measures_for_model.tsv", sep="\t", index=False)

print("File has been created.")
