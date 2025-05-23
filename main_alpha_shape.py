from __future__ import division
from Bio.PDB import *
import pandas as pd
import numpy as np
import os
import re
import alphashape  # NEW
from shapely.geometry import Point  # NEW
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius, calculate_average_plddt
)
from tqdm import tqdm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

parse = PDBParser()

# Load and preprocess data
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0)
data_df = data_df.applymap(conv_array_text)
data_df = data_df[0:10]

# Initialize structural properties storage
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
    'max_depth': np.zeros(num_proteins),  # NEW: Renamed from max_radius
    'hull_presence': np.zeros(num_proteins),
}

# Initialize layer counters
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


pdb_paths_dict = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")

# First pass: Calculate basic properties and spatial parameters
for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if not pdb_file_path or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)

        # Extract structural properties
        structural_properties['density'][idx] = calculate_density(structure)
        structural_properties['radius_of_gyration'][idx] = calculate_radius_of_gyration(structure)
        structural_properties['surface_area_to_volume_ratio'][idx] = calculate_surface_area_to_volume_ratio(structure)
        structural_properties['sphericity'][idx] = calculate_sphericity(structure)
        structural_properties['euler_characteristic'][idx] = calculate_euler_characteristic(structure)
        structural_properties['inradius'][idx] = calculate_inradius(structure)
        structural_properties['circumradius'][idx] = calculate_circumradius(structure)
        structural_properties['hydrodynamic_radius'][idx] = calculate_hydrodynamic_radius(structure)
        structural_properties['sequence_length'][idx] = len([
            residue for model in structure
            for chain in model
            for residue in chain if residue.id[0] == ' '
        ])
        structural_properties['avg_plddt'][idx] = calculate_average_plddt(structure)

        # Calculate spatial parameters using concave hull
        points = extract_N_and_CA_backbone_atoms(structure)
        if len(points) < 4:  # Need at least 4 points for 3D alpha shape
            continue

        try:  # NEW: Compute concave hull
            concave_hull = alphashape.alphashape(points, 0.5)
            structural_properties['max_depth'][idx] = concave_hull.area  # Store hull area
        except:
            structural_properties['max_depth'][idx] = 0

    except Exception as e:
        continue

# Second pass: Concave hull-based layer analysis
timepoint_columns = list(data_df.columns[1:])
for col in timepoint_columns:
    structural_properties[f'hull_presence_{col}'] = np.zeros(num_proteins)

for idx, uniprot_id in enumerate(data_df.index):
    try:
        pdb_file_path = pdb_paths_dict.get(uniprot_id)
        if not pdb_file_path or not os.path.isfile(pdb_file_path):
            continue

        structure = parse.get_structure(uniprot_id, pdb_file_path)
        points = extract_N_and_CA_backbone_atoms(structure)

        if len(points) < 4:
            continue

        # NEW: Compute concave hull and distances
        concave_hull = alphashape.alphashape(points, 0.5)
        if not concave_hull.is_valid:
            continue

        # Calculate distances to hull surface
        distances = []
        for point in points:
            distances.append(concave_hull.distance(Point(point)))
        max_depth = np.max(distances)
        structural_properties['max_depth'][idx] = max_depth

        # NEW: Define depth-based layers
        layer_boundaries = np.linspace(0, max_depth, 11) if max_depth > 0 else []

        # Process timepoints
        max_hull_presence = 0
        for col in timepoint_columns:
            coords = data_df.loc[uniprot_id, col]
            if not coords:
                structural_properties[f'hull_presence_{col}'][idx] = 0
                continue

            # Calculate average depth from surface
            avg_depth = np.mean([concave_hull.distance(Point(coord)) for coord in coords])

            # Assign layer (1=outermost near surface, 10=innermost)
            layer = next((i for i in range(1, 11)
                          if layer_boundaries[i - 1] <= avg_depth < layer_boundaries[i]), 10)
            structural_properties[f'hull_presence_{col}'][idx] = layer
            max_hull_presence = max(max_hull_presence, layer)

        structural_properties['hull_presence'][idx] = max_hull_presence

        # Count lysine/arginine in layers
        lys_arg_counts = [0] * 10
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ' or residue.get_resname() not in ['LYS', 'ARG']:
                        continue
                    if not residue.has_id('CA'):
                        continue
                    res_coord = residue['CA'].get_coord()
                    depth = concave_hull.distance(Point(res_coord))
                    for layer in range(1, 11):
                        if layer_boundaries[layer - 1] <= depth < layer_boundaries[layer]:
                            lys_arg_counts[layer - 1] += 1
                            break

        for i in range(10):
            structural_properties[f'lys_arg_layer_{i + 1}'][idx] = lys_arg_counts[i]

    except Exception as e:
        continue


# Updated visualization function
def plot_protein_layers(uniprot_id, pdb_file_path, save_path=None):
    """Visualize protein layers based on concave hull depth"""
    parser = PDBParser()
    structure = parser.get_structure(uniprot_id, pdb_file_path)
    points = extract_N_and_CA_backbone_atoms(structure)

    if len(points) < 4:
        return

    concave_hull = alphashape.alphashape(points, 0.5)
    if not concave_hull.is_valid:
        return

    # Calculate distances and layer boundaries
    distances = [concave_hull.distance(Point(p)) for p in points]
    max_depth = np.max(distances)
    layer_boundaries = np.linspace(0, max_depth, 11)
    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='lightgray', alpha=0.3, s=5, label='All atoms')

    # Plot layers
    for layer in range(10):
        mask = (distances >= layer_boundaries[layer]) & (distances < layer_boundaries[layer + 1])
        layer_points = points[mask]
        ax.scatter(layer_points[:, 0], layer_points[:, 1], layer_points[:, 2],
                   c=colors[layer], s=20,
                   label=f'Layer {layer + 1} ({layer_boundaries[layer]:.1f}-{layer_boundaries[layer + 1]:.1f}Å)')

    ax.set_title(f'{uniprot_id} Concave Hull Layers\n(Layer 1=Surface, Layer 10=Core)')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Generate results
structural_properties_df = pd.DataFrame(structural_properties)
structural_properties_df.insert(0, 'uniprot_id', data_df.index)

# Updated histogram
plt.figure(figsize=(10, 6))
plt.hist(structural_properties_df['hull_presence'],
         bins=np.arange(0.5, 11.5, 1),
         edgecolor='black')
plt.xticks(range(1, 11))
plt.xlabel('Innermost Layer Reached (10=Core, 1=Surface)')
plt.ylabel('Frequency')
plt.title('Protein Cleavage Depth Distribution (Concave Hull)')
plt.show()

# Example visualization
uniprot_id = "O60361"
pdb_file_path = pdb_paths_dict[uniprot_id]
plot_protein_layers(uniprot_id, pdb_file_path, save_path="concave_hull_layers.png")