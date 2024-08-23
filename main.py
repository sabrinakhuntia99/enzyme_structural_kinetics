from __future__ import division
from Bio.PDB import *
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import spearmanr, linregress

from util_func import conv_array_text, extract_backbone_atoms
from tqdm import tqdm
from calc_functions import calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio, \
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius, calculate_circumradius, \
    calculate_hydrodynamic_radius

parse = PDBParser()

# Atomic masses in Dalton (g/mol)
atomic_masses = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'P': 30.974,
    'S': 32.06,
    'SE': 78.96,
}

# Kyte-Doolittle hydrophobicity scale
kyte_doolittle_scale = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9,
    'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3,
    'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5,
    'K': -3.9, 'R': -4.5
}

# Extract all peptide cleavage XYZ coordinates (based on time point)
data_df = pd.read_csv("pep_cleave_coordinates_10292023.csv", index_col=0)
data_df = data_df.applymap(conv_array_text)

# Preallocate arrays for storing disorder_properties, CPI, sequence_length, and hydrophobicity_full
num_proteins = len(data_df.index)
disorder_properties = {
    'Density': np.zeros(num_proteins),
    'Radius_of_Gyration': np.zeros(num_proteins),
    'Surface_Area_to_Volume_Ratio': np.zeros(num_proteins),
    'Sphericity': np.zeros(num_proteins),
    'CPI': np.zeros(num_proteins),
    'Euler_Characteristic': np.zeros(num_proteins),
    'Inradius': np.zeros(num_proteins),
    'Circumradius': np.zeros(num_proteins),
    'Hydrodynamic_Radius': np.zeros(num_proteins),
    'Sequence_Length': np.zeros(num_proteins),
    'Hydrophobicity_Full': [''] * num_proteins  # Store as strings to avoid mixed-type issues
}

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

# Function to calculate CPI using the slope of the regression line
def calculate_cpi(dist_to_centroid):
    slopes = []
    for dist_list in dist_to_centroid:
        if len(dist_list) > 1:
            time_points = np.arange(len(dist_list))
            slope, _, _, _, _ = linregress(time_points, dist_list)
            slopes.append(slope)
    return slopes

# Get dictionary of PDB file paths
pdb_paths_dict = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")

# Loop through each protein to calculate disorder_properties and CPI
for idx, uniprot_id in enumerate(data_df.index):
    try:
        print(uniprot_id)
        # Get corresponding PDB file path
        pdb_file_path = pdb_paths_dict.get(uniprot_id)

        # Check if the PDB file exists
        if not os.path.isfile(pdb_file_path):
            print(f"PDB file not found for UniProt ID {uniprot_id}. Skipping...")
            continue

        # Parse PDB file
        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Extract sequence
        ppb = PPBuilder()
        sequences = [pp.get_sequence() for pp in ppb.build_peptides(structure)]
        sequence = ''.join(str(seq) for seq in sequences)
        sequence_length = len(sequence)

        # Calculate hydrophobicity_full using Kyte-Doolittle scale
        hydrophobicity_values = [kyte_doolittle_scale.get(aa, np.nan) for aa in sequence]
        hydrophobicity_full = ','.join(map(str, hydrophobicity_values))

        # Calculate disorder_properties
        density = calculate_density(structure)
        radius_of_gyration = calculate_radius_of_gyration(structure)
        surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
        sphericity = calculate_sphericity(structure)
        euler_characteristic = calculate_euler_characteristic(structure)
        inradius = calculate_inradius(structure)
        circumradius = calculate_circumradius(structure)
        hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

        # Ensure valid values
        disorder_properties['Density'][idx] = np.nan if np.isnan(density) or np.isinf(density) else density
        disorder_properties['Radius_of_Gyration'][idx] = np.nan if np.isnan(radius_of_gyration) or np.isinf(radius_of_gyration) else radius_of_gyration
        disorder_properties['Surface_Area_to_Volume_Ratio'][idx] = np.nan if np.isnan(surface_area_to_volume_ratio) or np.isinf(surface_area_to_volume_ratio) else surface_area_to_volume_ratio
        disorder_properties['Sphericity'][idx] = np.nan if np.isnan(sphericity) or np.isinf(sphericity) else sphericity
        disorder_properties['Euler_Characteristic'][idx] = np.nan if np.isnan(euler_characteristic) or np.isinf(euler_characteristic) else euler_characteristic
        disorder_properties['Inradius'][idx] = np.nan if np.isnan(inradius) or np.isinf(inradius) else inradius
        disorder_properties['Circumradius'][idx] = np.nan if np.isnan(circumradius) or np.isinf(circumradius) else circumradius
        disorder_properties['Hydrodynamic_Radius'][idx] = np.nan if np.isnan(hydrodynamic_radius) or np.isinf(hydrodynamic_radius) else hydrodynamic_radius

        # Calculate centroid of the protein structure
        points = extract_backbone_atoms(structure)
        centroid = np.mean(points, axis=0)

        dist_to_centroid = []
        # Calculate distance between each data coordinate and centroid
        for coord_array in data_df.loc[uniprot_id, data_df.columns[1:]]:
            if coord_array:
                dists_at_coord = [np.linalg.norm(coord - centroid) for coord in coord_array]
                dist_to_centroid.append(dists_at_coord)

        # Calculate CPI using the slope of regression line
        if dist_to_centroid:
            cpi_values = calculate_cpi(dist_to_centroid)
            disorder_properties['CPI'][idx] = np.nan if len(cpi_values) == 0 else np.mean(cpi_values)

        # Store sequence length and hydrophobicity full
        disorder_properties['Sequence_Length'][idx] = sequence_length
        disorder_properties['Hydrophobicity_Full'][idx] = hydrophobicity_full

    except Exception as e:
        print(f"An error occurred for UniProt ID {uniprot_id}: {e}")
        continue
# Convert disorder_properties dictionary to DataFrame
disorder_properties_df = pd.DataFrame(disorder_properties, index=data_df.index)

# Handle NaN or infinite values before calculating correlations
# Keep track of valid indices in disorder_properties_df
valid_indices = disorder_properties_df.replace([np.inf, -np.inf], np.nan).dropna().index

# Filter both disorder_properties_df and data_df to include only valid rows
disorder_properties_df_clean = disorder_properties_df.loc[valid_indices]
data_df_clean = data_df.loc[valid_indices]

# Calculate correlations between all properties using Spearman's rank correlation
correlations = {}
for prop1 in disorder_properties_df_clean.columns:
    for prop2 in disorder_properties_df_clean.columns:
        if prop1 != prop2:
            corr, _ = spearmanr(disorder_properties_df_clean[prop1], disorder_properties_df_clean[prop2])
            correlations[(prop1, prop2)] = corr

print("Spearman's rank correlations between properties:")
for (prop1, prop2), corr_value in correlations.items():
    print(f"{prop1} vs {prop2}: {corr_value}")

# Export the final data to a TSV file
disorder_properties_df_clean.to_csv("exp_data.tsv", sep='\t')
