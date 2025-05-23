from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import requests
from molecular_extraction_functions import conv_array_text, extract_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, sequence_to_hydrophobicity_array
)
from tqdm import tqdm
from scipy.stats import linregress
import time

parse = PDBParser()

# Define the path for the PDB files
pdb_folder_path = 'C:/Users/Sabrina/Documents/GitHub/protein_structural_kinetics/pdb_list'

# Extract all peptide cleavage XYZ coordinates (based on time point)
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
data_df = data_df.applymap(conv_array_text)
data_df = data_df[0:100]

# Preallocate arrays for storing structural properties
num_proteins = len(data_df.index)
structural_properties = {
    'density': np.zeros(num_proteins),
    'radius_of_gyration': np.zeros(num_proteins),
    'surface_area_to_volume_ratio': np.zeros(num_proteins),
    'sphericity': np.zeros(num_proteins),
    'hurst_exponent': np.zeros(num_proteins),
    'euler_characteristic': np.zeros(num_proteins),
    'inradius': np.zeros(num_proteins),
    'circumradius': np.zeros(num_proteins),
    'hydrodynamic_radius': np.zeros(num_proteins),
    'hydrophobicity_full': np.zeros(num_proteins)
}


def uniprot_to_pdb(uniprot_id, retries=3):
    """Fetch PDB IDs associated with a UniProt ID with retry mechanism."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"

    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            pdb_ids = []
            for line in response.text.splitlines():
                if line.startswith("DR   PDB;"):
                    pdb_id = line.split(";")[1].strip()
                    pdb_ids.append(pdb_id)

            return pdb_ids

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying

    return []  # Return empty if all attempts fail


def get_pdb_file_paths(uniprot_ids):
    pdb_paths = {}
    for uniprot_id in tqdm(uniprot_ids):
        pdb_ids = uniprot_to_pdb(uniprot_id)
        for pdb_id in pdb_ids:
            pdb_file_path = os.path.join(pdb_folder_path, f"{pdb_id}.pdb")
            if os.path.isfile(pdb_file_path):
                pdb_paths[uniprot_id] = pdb_file_path
                break  # Take the first available PDB file for this UniProt ID
        time.sleep(1)  # Add a delay between requests
    return pdb_paths


# Get dictionary of PDB file paths
pdb_paths_dict = get_pdb_file_paths(data_df.index)


def calculate_hurst_exponent(distances):
    """Calculate the Hurst exponent for a given set of distances."""
    if len(distances) < 2:
        return np.nan  # Not enough data to calculate slope

    # Log-log plot for Hurst exponent calculation
    x_values = np.arange(1, len(distances) + 1)
    log_x = np.log(x_values)
    log_y = np.log(distances)

    slope, _, _, _, _ = linregress(log_x, log_y)
    return slope


# Loop through each protein to calculate structural properties and Hurst exponent
for idx, uniprot_id in enumerate(data_df.index):
    try:
        # Get corresponding PDB file path
        pdb_file_path = pdb_paths_dict.get(uniprot_id)

        # Check if the PDB file exists
        if not pdb_file_path:
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
        structural_properties['hydrophobicity_full'][idx] = sequence_to_hydrophobicity_array(full_sequence)

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
        print(f"An error occurred for UniProt ID {uniprot_id}: {e}")
        continue

# Convert structural_properties dictionary to DataFrame
structural_properties_df = pd.DataFrame(structural_properties)

# Write the data to a TSV file
structural_properties_df.to_csv('lp_data_pdb_he.tsv', sep='\t', index=True)
