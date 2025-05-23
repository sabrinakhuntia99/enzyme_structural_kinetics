from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
import requests
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, calculate_average_plddt
)
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Initialize PDB parser
parse = PDBParser()

# Load cleavage coordinate data
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
data_df = data_df.applymap(conv_array_text)

# Preallocate structural properties arrays
num_proteins = len(data_df.index)
timepoint_columns = list(data_df.columns)
structural_properties = {
    'density': np.full(num_proteins, np.nan),
    'radius_of_gyration': np.full(num_proteins, np.nan),
    'surface_area_to_volume_ratio': np.full(num_proteins, np.nan),
    'sphericity': np.full(num_proteins, np.nan),
    'euler_characteristic': np.full(num_proteins, np.nan),
    'inradius': np.full(num_proteins, np.nan),
    'circumradius': np.full(num_proteins, np.nan),
    'hydrodynamic_radius': np.full(num_proteins, np.nan),
    'sequence_length': np.full(num_proteins, np.nan),
    'avg_plddt': np.full(num_proteins, np.nan),
    'hull_presence': np.zeros(num_proteins),
}

# Add per-timepoint hull presence columns
for col in timepoint_columns:
    structural_properties[f'hull_presence_{col}'] = np.zeros(num_proteins)

# Path configuration
pdb_folder_path = 'C:/Users/Sabrina/Documents/GitHub/protein_structural_kinetics/pdb_list'


def uniprot_to_pdb(uniprot_id):
    """Fetch experimental PDB IDs from UniProt"""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return [line.split(";")[1].strip()
                    for line in response.text.splitlines()
                    if line.startswith("DR   PDB;")]
    except Exception as e:
        print(f"Error fetching PDBs for {uniprot_id}: {str(e)}")
    return []


# First pass: Collect hull statistics from experimental structures
all_hull_radii = []
pdb_structure_cache = {}

print("Collecting hull statistics from experimental PDBs...")
for idx, uniprot_id in enumerate(tqdm(data_df.index)):
    pdb_ids = uniprot_to_pdb(uniprot_id)

    for pdb_id in pdb_ids:
        pdb_path = os.path.join(pdb_folder_path, f"{pdb_id}.pdb")
        if os.path.exists(pdb_path):
            try:
                structure = parse.get_structure(pdb_id, pdb_path)
                points = extract_N_and_CA_backbone_atoms(structure)
                hull = ConvexHull(points)
                centroid = np.mean(points, axis=0)
                hull_distances = [np.linalg.norm(p - centroid) for p in points[hull.vertices]]
                all_hull_radii.append(max(hull_distances))
                pdb_structure_cache[(uniprot_id, pdb_id)] = (structure, points)
            except Exception as e:
                continue

# Calculate global hull parameters
mean_hull_radius = np.mean(all_hull_radii)
std_hull_radius = np.std(all_hull_radii)


def determine_num_layers(hull_radius):
    """Dynamic layer calculation based on global distribution"""
    thresholds = [
        mean_hull_radius - 2 * std_hull_radius,
        mean_hull_radius - 1.5 * std_hull_radius,
        mean_hull_radius - std_hull_radius,
        mean_hull_radius - 0.5 * std_hull_radius,
        mean_hull_radius,
        mean_hull_radius + 0.5 * std_hull_radius,
        mean_hull_radius + std_hull_radius,
        mean_hull_radius + 1.5 * std_hull_radius,
        mean_hull_radius + 2 * std_hull_radius,
    ]
    for layer, threshold in enumerate(thresholds, 1):
        if hull_radius < threshold:
            return layer
    return 10


# Second pass: Calculate properties and hull presence
print("\nCalculating structural properties and hull presence...")
for idx, uniprot_id in enumerate(tqdm(data_df.index)):
    pdb_ids = uniprot_to_pdb(uniprot_id)
    if not pdb_ids:
        continue

    # Initialize tracking variables
    max_global_presence = 0
    timepoint_max = {col: 0 for col in timepoint_columns}
    prop_accumulator = {k: [] for k in structural_properties.keys()
                        if k not in ['hull_presence'] + [f'hull_presence_{c}' for c in timepoint_columns]}

    for pdb_id in pdb_ids:
        cache_key = (uniprot_id, pdb_id)
        if cache_key not in pdb_structure_cache:
            continue

        structure, points = pdb_structure_cache[cache_key]

        try:
            # Calculate standard properties
            prop_accumulator['density'].append(calculate_density(structure))
            prop_accumulator['radius_of_gyration'].append(calculate_radius_of_gyration(structure))
            prop_accumulator['surface_area_to_volume_ratio'].append(
                calculate_surface_area_to_volume_ratio(structure))
            prop_accumulator['sphericity'].append(calculate_sphericity(structure))
            prop_accumulator['euler_characteristic'].append(calculate_euler_characteristic(structure))
            prop_accumulator['inradius'].append(calculate_inradius(structure))
            prop_accumulator['circumradius'].append(calculate_circumradius(structure))
            prop_accumulator['hydrodynamic_radius'].append(calculate_hydrodynamic_radius(structure))
            prop_accumulator['sequence_length'].append(len([res for res in structure.get_residues()]))
            prop_accumulator['avg_plddt'].append(calculate_average_plddt(structure))

            # Calculate hull properties
            hull = ConvexHull(points)
            centroid = np.mean(points, axis=0)
            hull_distances = [np.linalg.norm(p - centroid) for p in points[hull.vertices]]
            max_hull_radius = max(hull_distances)
            num_layers = determine_num_layers(max_hull_radius)
            hull_layers = [np.percentile(hull_distances, (i + 1) * (100 / num_layers))
                           for i in range(num_layers)]

            # Process timepoints for current PDB
            current_pdb_timepoint_max = {col: 0 for col in timepoint_columns}
            for col in timepoint_columns:
                coords = data_df.loc[uniprot_id, col]
                if coords:
                    try:
                        avg_dist = np.mean([np.linalg.norm(c - centroid) for c in coords])
                        layer = next((i + 1 for i, t in enumerate(hull_layers) if avg_dist < t), 0)
                        current_pdb_timepoint_max[col] = layer
                    except:
                        continue

            # Update global maxima
            for col in timepoint_columns:
                if current_pdb_timepoint_max[col] > timepoint_max[col]:
                    timepoint_max[col] = current_pdb_timepoint_max[col]

            current_global_max = max(current_pdb_timepoint_max.values())
            if current_global_max > max_global_presence:
                max_global_presence = current_global_max

        except Exception as e:
            continue

    # Store results after processing all PDBs for this protein
    if any(len(v) > 0 for v in prop_accumulator.values()):
        # Average properties across PDBs
        for k in prop_accumulator:
            structural_properties[k][idx] = np.nanmean(prop_accumulator[k]) if prop_accumulator[k] else np.nan

        # Store hull presence values
        structural_properties['hull_presence'][idx] = max_global_presence
        for col in timepoint_columns:
            structural_properties[f'hull_presence_{col}'][idx] = timepoint_max[col]

# Create final DataFrame
structural_properties_df = pd.DataFrame(structural_properties)
structural_properties_df.insert(0, 'uniprot_id', data_df.index)

# Save results
output_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\PDB_all_timepoints_dynamic_hull_id_030625_NCA_backbone.tsv"
structural_properties_df.to_csv(output_path, sep='\t', index=False)

# Generate hull presence visualization
plt.figure(figsize=(12, 6))
plt.hist(structural_properties_df['hull_presence'].replace(0, np.nan).dropna(),
         bins=np.arange(0.5, 11.5, 1), edgecolor='black', align='mid')
plt.title('Maximum Hull Presence Distribution (Experimental Structures Only)')
plt.xlabel('Hull Layer')
plt.ylabel('Frequency')
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(output_path), 'hull_presence_distribution.png'))
plt.show()

print(f"\nProcessing complete! Results saved to {output_path}")