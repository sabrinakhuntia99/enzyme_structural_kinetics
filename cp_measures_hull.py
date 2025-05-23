from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, calculate_average_plddt
)
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

parse = PDBParser()

# Approximate atomic radii (in angstroms) for amino acids
atomic_radii = {
    "ALA": 1.80, "ARG": 2.50, "ASN": 2.10, "ASP": 2.05, "CYS": 2.00,
    "GLN": 2.20, "GLU": 2.30, "GLY": 1.75, "HIS": 2.35, "ILE": 2.40,
    "LEU": 2.40, "LYS": 2.50, "MET": 2.45, "PHE": 2.50, "PRO": 2.00,
    "SER": 2.00, "THR": 2.15, "TRP": 2.65, "TYR": 2.50, "VAL": 2.30
}

# Extract all peptide cleavage XYZ coordinates
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0
)
data_df = data_df.applymap(conv_array_text)

# Preallocate arrays for storing structural properties
num_proteins = len(data_df.index)
structural_properties = {
    'density': np.zeros(num_proteins),
    'radius_of_gyration': np.zeros(num_proteins),
    'surface_area_to_volume_ratio': np.zeros(num_proteins),
    'sphericity': np.zeros(num_proteins),
    'euler_characteristic': np.zeros(num_proteins),
    'inradius': np.zeros(num_proteins),
    'circumradius': np.zeros(num_proteins),
    'hydrodynamic_radius': np.zeros(num_proteins),
    'sequence_length': np.zeros(num_proteins),
    'avg_plddt': np.zeros(num_proteins),
    'hull_presence': np.zeros(num_proteins),
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

all_hull_radii = []

# Loop through each protein to calculate structural properties
for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if pdb_file_path is None or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Compute mean atomic radius for this protein
        residue_radii = [atomic_radii.get(res.get_resname(), 2.2) for model in structure
                         for chain in model for res in chain if res.id[0] == ' ']
        avg_atomic_radius = np.mean(residue_radii) if residue_radii else 2.2  # Default to 2.2 Å

        structural_properties['density'][idx] = calculate_density(structure)
        structural_properties['radius_of_gyration'][idx] = calculate_radius_of_gyration(structure)
        structural_properties['surface_area_to_volume_ratio'][idx] = calculate_surface_area_to_volume_ratio(structure)
        structural_properties['sphericity'][idx] = calculate_sphericity(structure)
        structural_properties['euler_characteristic'][idx] = calculate_euler_characteristic(structure)
        structural_properties['inradius'][idx] = calculate_inradius(structure)
        structural_properties['circumradius'][idx] = calculate_circumradius(structure)
        structural_properties['hydrodynamic_radius'][idx] = calculate_hydrodynamic_radius(structure)
        structural_properties['sequence_length'][idx] = len(residue_radii)
        structural_properties['avg_plddt'][idx] = calculate_average_plddt(structure)

        points = extract_N_and_CA_backbone_atoms(structure)
        hull = ConvexHull(points)
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]
        max_hull_radius = max(hull_distances)
        all_hull_radii.append(max_hull_radius)

    except Exception:
        continue

all_hull_radii = np.array(all_hull_radii)
mean_hull_radius = np.mean(all_hull_radii)
std_hull_radius = np.std(all_hull_radii)

def determine_num_layers(hull_radius, mean_radius, std_radius, avg_atomic_radius):
    scale_factor = avg_atomic_radius / 2.2  # Normalize by typical atomic radius (2.2 Å)
    adjusted_mean = mean_radius * scale_factor
    adjusted_std = std_radius * scale_factor

    thresholds = [
        adjusted_mean - 2 * adjusted_std,
        adjusted_mean - 1.5 * adjusted_std,
        adjusted_mean - adjusted_std,
        adjusted_mean - 0.5 * adjusted_std,
        adjusted_mean,
        adjusted_mean + 0.5 * adjusted_std,
        adjusted_mean + adjusted_std,
        adjusted_mean + 1.5 * adjusted_std,
        adjusted_mean + 2 * adjusted_std,
    ]

    for layer, threshold in enumerate(thresholds, start=1):
        if hull_radius < threshold:
            return layer
    return 10

for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if pdb_file_path is None or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)
        residue_radii = [atomic_radii.get(res.get_resname(), 2.2) for model in structure
                         for chain in model for res in chain if res.id[0] == ' ']
        avg_atomic_radius = np.mean(residue_radii) if residue_radii else 2.2

        points = extract_N_and_CA_backbone_atoms(structure)
        hull = ConvexHull(points)
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]
        max_hull_radius = max(hull_distances)

        num_layers = determine_num_layers(max_hull_radius, mean_hull_radius, std_hull_radius, avg_atomic_radius)

        structural_properties['hull_presence'][idx] = num_layers

    except Exception:
        continue

structural_properties_df = pd.DataFrame(structural_properties)
output_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\dynamic_hull_atomic_radius.tsv"
structural_properties_df.to_csv(output_file_path, sep='\t', index=False)
