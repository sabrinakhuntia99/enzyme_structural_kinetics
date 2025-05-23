from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from molecular_extraction_functions import conv_array_text
from tqdm import tqdm

# Initialize PDB parser
parse = PDBParser()

# Load the dataset
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0
)
data_df = data_df.applymap(conv_array_text)

# Function to get PDB file paths
def get_pdb_file_paths(folder_path):
    pdb_paths = {}
    pattern = re.compile(r"AF-(\w+)-F\d+-model_v4.pdb")

    for subdir, _, files in tqdm(os.walk(folder_path)):
        match = pattern.match(os.path.basename(subdir))
        if match:
            uniprot_id = match.group(1)
            pdb_files = [f for f in files if f.endswith('.pdb')]
            if pdb_files:
                pdb_paths[uniprot_id] = os.path.join(subdir, pdb_files[0])

    return pdb_paths

# Get PDB file paths
pdb_paths_dict = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")

# Lists to store lysine and arginine content
lys_arg_content = []

# Loop through each protein to calculate lysine and arginine content
for idx, uniprot_id in enumerate(data_df.index):
    try:
        # Get corresponding PDB file path
        pdb_file_path = pdb_paths_dict.get(uniprot_id)

        # Check if the PDB file exists
        if not os.path.isfile(pdb_file_path):
            print(f"PDB file not found for UniProt ID {uniprot_id}. Skipping...")
            lys_arg_content.append(np.nan)  # Append NaN for missing PDBs
            continue

        # Parse the PDB file
        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Initialize counts
        lys_count = 0
        arg_count = 0
        total_residues = 0

        # Extract sequences and count lysine/arginine
        for model in structure:
            for chain in model:
                for res in chain:
                    res_name = res.get_resname()
                    total_residues += 1
                    if res_name == 'LYS':
                        lys_count += 1
                    elif res_name == 'ARG':
                        arg_count += 1

        # Calculate combined lysine and arginine content
        combined_content = (lys_count + arg_count) / total_residues * 100 if total_residues > 0 else np.nan
        lys_arg_content.append(combined_content)

    except Exception as e:
        print(f"Error processing UniProt ID {uniprot_id}: {e}")
        lys_arg_content.append(np.nan)  # Append NaN for errors

# Convert to DataFrame for easier handling
lys_arg_df = pd.DataFrame(lys_arg_content, columns=['lys_arg_content'])
lys_arg_df['has_values'] = lys_arg_df['lys_arg_content'].notna()

# Plotting density histograms
plt.figure(figsize=(12, 6))

# Histogram for proteins with values
plt.subplot(1, 2, 1)
plt.hist(lys_arg_df[lys_arg_df['has_values']]['lys_arg_content'], bins=30, density=True, alpha=0.7, color='blue')
plt.title('Density Histogram of Lysine and Arginine Content (With Values)')
plt.xlabel('Lysine + Arginine Content (%)')
plt.ylabel('Density')
plt.grid(axis='y')

# Histogram for proteins with 0/NA values
plt.subplot(1, 2, 2)
plt.hist(lys_arg_df[~lys_arg_df['has_values']]['lys_arg_content'].fillna(0), bins=30, density=True, alpha=0.7, color='red')
plt.title('Density Histogram of Lysine and Arginine Content (0/NA Values)')
plt.xlabel('Lysine + Arginine Content (%)')
plt.ylabel('Density')
plt.grid(axis='y')

plt.tight_layout()
plt.show()
