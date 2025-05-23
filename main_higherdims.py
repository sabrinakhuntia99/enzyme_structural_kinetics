from __future__ import division
import pyvista as pv
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from ripser import ripser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Constants
MAX_POINTS = 200  # Max points to sample for topology

# Load and preprocess coordinate data
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
data_df = data_df.applymap(conv_array_text)
data_df = data_df[0:1000]

# Output structural dictionary
num_proteins = len(data_df.index)
structural_properties = {
    'betti_0': np.zeros(num_proteins),
    'betti_1': np.zeros(num_proteins),
    'hull_presence': np.zeros(num_proteins),
}
for i in range(1, 11):
    structural_properties[f'lys_arg_layer_{i}'] = np.zeros(num_proteins)

# Add per-timepoint hull presence
timepoint_columns = list(data_df.columns[1:])
for col in timepoint_columns:
    structural_properties['hull_presence_' + col] = np.zeros(num_proteins)

# Load PDB paths
def get_pdb_file_paths(folder_path):
    pdb_paths = {}
    pattern = re.compile(r"AF-(\w+)-F\d+-model_v4.pdb")
    for subdir, _, files in os.walk(folder_path):
        match = pattern.match(os.path.basename(subdir))
        if match:
            uniprot_id = match.group(1)
            pdb_files = [f for f in files if f.endswith('.pdb')]
            if pdb_files:
                pdb_paths[uniprot_id] = os.path.join(subdir, pdb_files[0])
    return pdb_paths

pdb_paths_dict = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")
parse = PDBParser()

# Calculate overall distribution of hull radii to define dynamic layers
all_hull_radii = []

# Loop through proteins to calculate all hull radii for global statistics
for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if pdb_file_path is None or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)
        points = extract_N_and_CA_backbone_atoms(structure)
        hull = ConvexHull(points)
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]
        all_hull_radii.extend(hull_distances)
    except Exception as e:
        print(f"Error processing {uniprot_id}: {e}")

# Global hull radius statistics
all_hull_radii = np.array(all_hull_radii)
mean_hull_radius = np.mean(all_hull_radii)
std_hull_radius = np.std(all_hull_radii)

# Function to determine the number of dynamic layers
def determine_num_layers(hull_radius, mean_radius, std_radius):
    thresholds = [
        mean_radius - 2 * std_radius,
        mean_radius - 1.5 * std_radius,
        mean_radius - std_radius,
        mean_radius - 0.5 * std_radius,
        mean_radius,
        mean_radius + 0.5 * std_radius,
        mean_radius + std_radius,
        mean_radius + 1.5 * std_radius,
        mean_radius + 2 * std_radius,
    ]
    for layer, threshold in enumerate(thresholds, start=1):
        if hull_radius < threshold:
            return layer
    return 10  # Default to layer 10 if above all thresholds

# Process each protein
for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if pdb_file_path is None or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)
        points = extract_N_and_CA_backbone_atoms(structure)
        hull = ConvexHull(points)
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]
        max_hull_radius = max(hull_distances)

        # Determine dynamic number of layers and calculate the percentile thresholds
        num_layers = determine_num_layers(max_hull_radius, mean_hull_radius, std_hull_radius)
        hull_layers = [np.percentile(hull_distances, (i + 1) * (100 / num_layers)) for i in range(num_layers)]

        # Compute per-timepoint hull presence and track the maximum
        max_hull_presence = 0
        for col in timepoint_columns:
            coord_array = data_df.loc[uniprot_id, col]
            if coord_array:
                dists_at_coord = [np.linalg.norm(coord - centroid) for coord in coord_array]
                avg_distance = np.average(dists_at_coord)
            else:
                avg_distance = np.nan

            timepoint_hull_presence = 0  # Default if none of the thresholds are met
            for layer_idx in range(len(hull_layers)):
                if avg_distance < hull_layers[layer_idx]:
                    timepoint_hull_presence = layer_idx + 1
                    break

            # Store the per-timepoint hull presence
            structural_properties['hull_presence_' + col][idx] = timepoint_hull_presence

            # Update the maximum hull presence encountered
            if timepoint_hull_presence > max_hull_presence:
                max_hull_presence = timepoint_hull_presence

        # Assign the maximum hull presence across all timepoints
        structural_properties['hull_presence'][idx] = max_hull_presence

        # Compute topological properties (Betti 0 and 1)
        diagrams = ripser(points, maxdim=1, do_cocycles=False)['dgms']
        betti_0 = len(diagrams[0])  # Betti-0 (connected components)
        betti_1 = len(diagrams[1])  # Betti-1 (loops)

        # Output topological values
        structural_properties['betti_0'][idx] = betti_0
        structural_properties['betti_1'][idx] = betti_1

        # Lys/Arg counts per layer
        num_layers = 10
        layer_thresholds = [np.percentile(hull_distances, (i + 1) * 10) for i in range(num_layers)]
        lys_arg_counts = [0] * num_layers
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] != ' ' or res.get_resname() not in ['LYS', 'ARG']:
                        continue
                    if not res.has_id('CA'):
                        continue
                    dist = np.linalg.norm(res['CA'].get_coord() - centroid)
                    for i, t in enumerate(layer_thresholds):
                        if dist < t:
                            lys_arg_counts[i] += 1
                            break

        # Output Lys/Arg counts for each layer
        for i in range(1, 11):
            structural_properties[f'lys_arg_layer_{i}'][idx] = lys_arg_counts[i - 1]

    except Exception as e:
        print(f"Error processing {uniprot_id}: {e}")

# Save output
structural_properties_df = pd.DataFrame(structural_properties)
structural_properties_df.insert(0, 'uniprot_id', data_df.index)
structural_properties_df.to_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\041025_topology_only.tsv",
    sep='\t', index=False)

# Optional: Plot global hull presence distribution
plt.figure(figsize=(10, 6))
plt.hist(structural_properties_df['hull_presence'], bins=np.arange(0.5, 11.5, 1), edgecolor='black', align='mid')
plt.xticks(np.arange(1, 11))
plt.xlabel('Hull Presence Layer')
plt.ylabel('Frequency')
plt.title('Distribution of Global Hull Presence')
plt.grid(axis='y', linestyle='--')
plt.show()

