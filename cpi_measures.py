from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
from util_func import conv_array_text, extract_backbone_atoms
from calc_functions import calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio, calculate_sphericity, calculate_euler_characteristic, calculate_inradius, calculate_circumradius, calculate_hydrodynamic_radius
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Extract all peptide cleavage XYZ coordinates (based on time point)
data_df = pd.read_csv("pep_cleave_coordinates_10292023.csv", index_col=0)
data_df = data_df.applymap(conv_array_text)

# Preallocate arrays for storing disorder_properties and CPI
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
    'Hydrodynamic_Radius': np.zeros(num_proteins)
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

# Get dictionary of PDB file paths
pdb_paths_dict = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")

def calculate_cpi(distances):
    cpi_values = []
    for i in range(len(distances) - 1):
        if distances[i] < distances[i + 1]:
            cpi_values.append(-1)  # Distance increases over time
        elif distances[i] > distances[i + 1]:
            cpi_values.append(1)  # Distance decreases over time
        else:
            cpi_values.append(0)  # Distance remains constant
    return cpi_values

# Loop through each protein to calculate disorder_properties and CPI
for idx, uniprot_id in enumerate(data_df.index):
    try:
        # Get corresponding PDB file path
        pdb_file_path = pdb_paths_dict.get(uniprot_id)

        # Check if the PDB file exists
        if not os.path.isfile(pdb_file_path):
            print(f"PDB file not found for UniProt ID {uniprot_id}. Skipping...")
            continue

        # Parse PDB file
        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Calculate disorder_properties
        disorder_properties['Density'][idx] = calculate_density(structure)
        disorder_properties['Radius_of_Gyration'][idx] = calculate_radius_of_gyration(structure)
        disorder_properties['Surface_Area_to_Volume_Ratio'][idx] = calculate_surface_area_to_volume_ratio(structure)
        disorder_properties['Sphericity'][idx] = calculate_sphericity(structure)

        disorder_properties['Euler_Characteristic'][idx] = calculate_euler_characteristic(structure)
        disorder_properties['Inradius'][idx] = calculate_inradius(structure)
        disorder_properties['Circumradius'][idx] = calculate_circumradius(structure)
        disorder_properties['Hydrodynamic_Radius'][idx] = calculate_hydrodynamic_radius(structure)



        # Calculate centroid of the protein structure
        points = extract_backbone_atoms(structure)
        centroid = np.mean(points, axis=0)

        dist_to_centroid = []
        # Calculate distance between each data coordinate and centroid
        for coord_array in data_df.loc[uniprot_id, data_df.columns[1:]]:
            if coord_array:
                dists_at_coord = [np.linalg.norm(coord - centroid) for coord in coord_array]
                dist_to_centroid.append(np.average(dists_at_coord))

        # Calculate CPI
        cpi_values = calculate_cpi(dist_to_centroid)
        disorder_properties['CPI'][idx] = sum(cpi_values)

    except Exception as e:
        # print(f"An error occurred for UniProt ID {uniprot_id}: {e}")
        continue

# Convert disorder_properties dictionary to DataFrame
disorder_properties_df = pd.DataFrame(disorder_properties)

# Calculate correlation between CPI and each property
correlations = disorder_properties_df.corr()['CPI']

print("Correlation between CPI and each property:")
print(correlations)
# Add full matrix

