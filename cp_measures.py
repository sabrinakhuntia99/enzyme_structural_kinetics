from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
from molecular_extraction_functions import conv_array_text, extract_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, calculate_average_plddt
)
from tqdm import tqdm

parse = PDBParser()

# Extract all peptide cleavage XYZ coordinates (based on time point)
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
data_df = data_df.applymap(conv_array_text)

# Preallocate arrays for storing structural properties
num_proteins = len(data_df.index)
structural_properties = {
    'hurst_exponent': np.zeros(num_proteins),
    'density': np.zeros(num_proteins),
    'radius_of_gyration': np.zeros(num_proteins),
    'surface_area_to_volume_ratio': np.zeros(num_proteins),
    'sphericity': np.zeros(num_proteins),
    'euler_characteristic': np.zeros(num_proteins),
    'inradius': np.zeros(num_proteins),
    'circumradius': np.zeros(num_proteins),
    'hydrodynamic_radius': np.zeros(num_proteins),
    'sequence_length': np.zeros(num_proteins),
    'avg_plddt': np.zeros(num_proteins)
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

def calculate_hurst_exponent(distances):
    """Calculate the Hurst exponent using the Rescaled Range method."""
    n = len(distances)
    if n < 2:
        return np.nan  # Not enough data

    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    deviations = distances - mean_distance
    cumulative_dev = np.cumsum(deviations)
    R = np.max(cumulative_dev) - np.min(cumulative_dev)
    S = np.std(distances)

    if S == 0:
        return np.nan  # Avoid division by zero

    R_S = R / S
    hurst_exponent = np.log(R_S) / np.log(n)

    return hurst_exponent

# Loop through each protein to calculate structural properties and Hurst exponent
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

        # Extract full sequence from the structure
        full_sequence = ''.join(residue.get_resname() for model in structure for chain in model for residue in chain if
                                residue.id[0] == ' ')  # Exclude heteroatoms

        # Calculate structural properties
        structural_properties['density'][idx] = calculate_density(structure)
        structural_properties['radius_of_gyration'][idx] = calculate_radius_of_gyration(structure)
        structural_properties['surface_area_to_volume_ratio'][idx] = calculate_surface_area_to_volume_ratio(structure)
        structural_properties['sphericity'][idx] = calculate_sphericity(structure)
        structural_properties['euler_characteristic'][idx] = calculate_euler_characteristic(structure)
        structural_properties['inradius'][idx] = calculate_inradius(structure)
        structural_properties['circumradius'][idx] = calculate_circumradius(structure)
        structural_properties['hydrodynamic_radius'][idx] = calculate_hydrodynamic_radius(structure)
        structural_properties['sequence_length'][idx] = len(full_sequence)
        structural_properties['avg_plddt'][idx] = calculate_average_plddt(structure)

        # Calculate centroid of the protein structure
        points = extract_backbone_atoms(structure)
        centroid = np.mean(points, axis=0)

        # Calculate distances from centroid
        dist_to_centroid = []
        for coord_array in data_df.loc[uniprot_id, data_df.columns[1:]]:
            if coord_array:
                dists_at_coord = [np.linalg.norm(coord - centroid) for coord in coord_array]
                dist_to_centroid.append(np.average(dists_at_coord))

        # Calculate Hurst exponent
        if dist_to_centroid:
            hurst_exponent_value = calculate_hurst_exponent(dist_to_centroid)
            structural_properties['hurst_exponent'][idx] = hurst_exponent_value

    except Exception as e:
        continue

# Convert structural_properties dictionary to DataFrame
structural_properties_df = pd.DataFrame(structural_properties)

# Export to TSV file
output_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\lp_data_he_101124.tsv"
structural_properties_df.to_csv(output_file_path, sep='\t', index=False)
