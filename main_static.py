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

# Get timepoint column names (all columns except the index/UniProt ID)
timepoint_columns = list(data_df.columns[1:])
# Preallocate arrays for per-timepoint hull presence
for col in timepoint_columns:
    structural_properties['hull_presence_' + col] = np.zeros(num_proteins)

# Process each protein to calculate structural properties and hull layers
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
        centroid = np.mean(points, axis=0)
        hull_distances = [np.linalg.norm(point - centroid) for point in points[hull.vertices]]

        # Determine 10 hull layers based on percentiles of hull_distances
        hull_layers = np.percentile(hull_distances, np.linspace(10, 100, 10))

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
            for layer_idx in range(10):
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

        # Count Lysine and Arginine residues per hull layer (10 layers)
        lys_arg_counts = [0] * 10  # Initialize counts for 10 layers

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
                    for layer_idx in range(10):
                        if dist < hull_layers[layer_idx]:
                            lys_arg_counts[layer_idx] += 1
                            break

        # Save the counts into layers 1 through 10
        for i in range(1, 11):
            structural_properties[f'lys_arg_layer_{i}'][idx] = lys_arg_counts[i-1]

    except Exception as e:
        continue

# Convert structural properties dictionary to DataFrame
structural_properties_df = pd.DataFrame(structural_properties)
# Insert the UniProt ID as the first column
structural_properties_df.insert(0, 'uniprot_id', data_df.index)

# Export the DataFrame to a TSV file
output_file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\LA_all_timepoints_fixed_10_hull_layers_NCA_backbone.tsv"
structural_properties_df.to_csv(output_file_path, sep='\t', index=False)

# Plot distribution of the global hull presence
plt.figure(figsize=(10, 6))
plt.hist(structural_properties_df['hull_presence'], bins=np.arange(0.5, 11.5, 1), edgecolor='black', align='mid')
plt.xticks(np.arange(1, 11))
plt.xlabel('Hull Presence Layer')
plt.ylabel('Frequency')
plt.title('Distribution of Global Hull Presence Using N + CA Backbone (10 Layers)')
plt.grid(axis='y', linestyle='--')
plt.show()


'''
VISUALIZATION
'''
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def plot_convex_hull_layers_interactive(points):
    """
    Recursively compute convex hull layers from the given points and create an interactive 3D plot.
    Each convex hull layer is plotted as a mesh (filled surface) with a unique color and legend entry.

    Parameters:
        points (np.ndarray): An array of shape (n_points, 3) containing the 3D coordinates.
    """
    # Copy points so that we can remove outer hull vertices iteratively
    remaining_points = points.copy()
    layers = []  # will store a tuple (points used for hull, ConvexHull object)

    # Compute convex hull layers iteratively until not enough points remain for a 3D hull.
    while remaining_points.shape[0] >= 4:
        try:
            hull = ConvexHull(remaining_points)
        except Exception as e:
            break
        layers.append((remaining_points.copy(), hull))
        # Remove points that are part of the current hull
        mask = np.ones(remaining_points.shape[0], dtype=bool)
        mask[hull.vertices] = False
        remaining_points = remaining_points[mask]

    # Create a Plotly figure
    fig = go.Figure()

    # Define a colormap for layers (you can use any colors you like)
    layer_colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
        '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
        '#FF97FF', '#FECB52'
    ]

    # Plot each convex hull layer as a mesh.
    for i, (pts, hull) in enumerate(layers):
        # The hull.simplices array contains indices of pts for each triangle face.
        simplices = hull.simplices
        # Extract the x, y, and z coordinates from pts.
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        # The indices for the mesh faces.
        i_indices = simplices[:, 0]
        j_indices = simplices[:, 1]
        k_indices = simplices[:, 2]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_indices, j=j_indices, k=k_indices,
            color=layer_colors[i % len(layer_colors)],
            opacity=0.5,
            name=f'Layer {i + 1}',
            showlegend=True
        ))

    # Calculate overall centroid of all backbone points.
    centroid = np.mean(points, axis=0)

    # Add scatter trace for backbone coordinates (blue)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(color='blue', size=3),
        name='Backbone Coordinates'
    ))

    # Add scatter trace for centroid (red)
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Centroid'
    ))

    # Update layout for interactive features and axes titles
    fig.update_layout(
        title="Interactive 3D Convex Hull Layers of a Protein",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(
            itemsizing='constant'
        )
    )

    fig.show()


# ---------------------------------------
# Example usage for a single protein
# ---------------------------------------
# Replace 'uniprot_id_example' with a valid key from your data.
uniprot_id_example = "O60361"  # selecting the first available protein for demonstration
pdb_file_path = pdb_paths_dict.get(uniprot_id_example)

if pdb_file_path is not None and os.path.isfile(pdb_file_path):
    structure = parse.get_structure(uniprot_id_example, pdb_file_path)
    # Extract backbone coordinates (N + CA) using your provided function.
    points = extract_N_and_CA_backbone_atoms(structure)

    # Plot the interactive 3D convex hull layers as meshes.
    plot_convex_hull_layers_interactive(points)
else:
    print("PDB file for the selected protein not found.")
