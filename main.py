from __future__ import division
from Bio.PDB import *
import pandas as pd
import os
import re
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, calculate_average_plddt
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import networkx as nx

parse = PDBParser()

# Extract all peptide cleavage XYZ coordinates (based on time point)
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
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

# Preallocate arrays for lysine+arginine counts for each hull layer (1 through 10)
for i in range(1, 11):
    structural_properties[f'lys_arg_layer_{i}'] = np.zeros(num_proteins)

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

# List to store all hull radii
all_hull_radii = []

# First loop: Calculate various properties and accumulate hull radii for layer calculation
for idx, uniprot_id in enumerate(data_df.index):
    try:
        # Get corresponding PDB file path
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if pdb_file_path is None or not os.path.isfile(pdb_file_path):
            continue

        # Parse PDB file
        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Extract full sequence from the structure (exclude heteroatoms)
        full_sequence = ''.join(
            residue.get_resname()
            for model in structure
            for chain in model
            for residue in chain if residue.id[0] == ' '
        )

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

        # Calculate points for the convex hull
        points = extract_N_and_CA_backbone_atoms(structure)

        # Create the convex hull
        hull = ConvexHull(points)

        # Calculate distances from centroid to hull vertices
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]
        max_hull_radius = max(hull_distances)
        all_hull_radii.append(max_hull_radius)

    except Exception as e:
        # If any error occurs, skip the protein.
        continue

# Calculate overall distribution of hull radii to define dynamic layers
all_hull_radii = np.array(all_hull_radii)
mean_hull_radius = np.mean(all_hull_radii)
std_hull_radius = np.std(all_hull_radii)

def determine_num_layers(hull_radius, mean_radius, std_radius):
    # Define custom thresholds for layers
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
    # Determine the layer based on thresholds
    for layer, threshold in enumerate(thresholds, start=1):
        if hull_radius < threshold:
            return layer
    return 10  # Default to layer 10 if above all thresholds

# Get timepoint column names (all columns except the index/UniProt ID)
timepoint_columns = list(data_df.columns[1:])
# Preallocate arrays for per-timepoint hull presence
for col in timepoint_columns:
    structural_properties['hull_presence_' + col] = np.zeros(num_proteins)
# Second loop: Calculate hull presence and Lys/Arg counts for each protein and for each timepoint
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
        max_hull_presence = 0  # Initialize maximum hull presence for this protein
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

        # Count Lysine and Arginine residues per hull layer.
        # Initialize count list for each dynamic layer
        lys_arg_counts = [0] * num_layers

        # Loop over all residues (exclude heteroatoms) and check for LYS and ARG
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    if residue.get_resname() not in ['LYS', 'ARG']:
                        continue
                    # Use the CA atom as representative if present
                    if not residue.has_id('CA'):
                        continue
                    res_coord = residue['CA'].get_coord()
                    dist = np.linalg.norm(res_coord - centroid)
                    # Determine which hull layer this residue falls into
                    for layer_idx, threshold in enumerate(hull_layers):
                        if dist < threshold:
                            lys_arg_counts[layer_idx] += 1
                            break

        # Save the counts into fixed 10 columns: if the protein has fewer than 10 layers, remaining layers get 0.
        for i in range(1, 11):
            if i <= num_layers:
                structural_properties[f'lys_arg_layer_{i}'][idx] = lys_arg_counts[i-1]
            else:
                structural_properties[f'lys_arg_layer_{i}'][idx] = 0

        # Generate peptide bond network graph with cleavage points
        try:
            # Extract residues in order from the first chain of the first model
            model = structure[0]
            chain = next(model.get_chains())  # Get the first chain
            residues = [res for res in chain if res.id[0] == ' ']
            if not residues:
                continue

            residue_ids = [res.id[1] for res in residues]

            # Create a networkx graph
            G = nx.Graph()
            G.add_nodes_from(residue_ids)
            # Add edges between consecutive residues
            edges = [(residue_ids[i], residue_ids[i+1]) for i in range(len(residue_ids)-1)]
            G.add_edges_from(edges)

            # Determine positions for plotting (linear layout)
            pos = {res_id: (i, 0) for i, res_id in enumerate(residue_ids)}

            # Create a plot
            plt.figure(figsize=(20, 2))
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
            nx.draw_networkx_edges(G, pos, width=1, edge_color='black')

            # Process cleavage points for each timepoint
            colors = plt.cm.get_cmap('tab10', len(timepoint_columns))
            for t_idx, col in enumerate(timepoint_columns):
                cleavage_coords = data_df.loc[uniprot_id, col]
                if not isinstance(cleavage_coords, np.ndarray) or cleavage_coords.size == 0:
                    continue
                cleaved_residues = set()
                for coord in cleavage_coords:
                    min_dist = float('inf')
                    closest_res = None
                    for res in residues:
                        if not res.has_id('CA'):
                            continue
                        ca_coord = res['CA'].get_coord()
                        dist = np.linalg.norm(coord - ca_coord)
                        if dist < min_dist:
                            min_dist = dist
                            closest_res = res
                    if closest_res:
                        cleaved_residues.add(closest_res.id[1])
                # Plot cleaved residues for this timepoint
                if cleaved_residues:
                    nx.draw_networkx_nodes(G, pos, nodelist=list(cleaved_residues),
                                           node_color=[colors(t_idx)],
                                           node_size=100,
                                           edgecolors='black',
                                           label=f'{col}')

            plt.title(f'Peptide Bond Network for {uniprot_id} with Cleavage Points')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis('off')
            plt.savefig(f'peptide_network_{uniprot_id}.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating graph for {uniprot_id}: {e}")
            continue

    except Exception as e:
        continue

# Convert structural properties dictionary to DataFrame
structural_properties_df = pd.DataFrame(structural_properties)
# Insert the UniProt ID as the first column
structural_properties_df.insert(0, 'uniprot_id', data_df.index)

# Export the DataFrame to a TSV file
output_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\040825_full_data.tsv"
structural_properties_df.to_csv(output_file_path, sep='\t', index=False)

# Plot distribution of the global hull presence
plt.figure(figsize=(10, 6))
plt.hist(structural_properties_df['hull_presence'], bins=np.arange(0.5, 11.5, 1), edgecolor='black', align='mid')
plt.xticks(np.arange(1, 11))
plt.xlabel('Hull Presence Layer')
plt.ylabel('Frequency')
plt.title('Distribution of Global Hull Presence Using N + CA Backbone')
plt.grid(axis='y', linestyle='--')
plt.show()