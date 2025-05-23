from __future__ import division
from Bio.PDB import PDBParser
import pandas as pd
import os
import re
import numpy as np
from molecular_extraction_functions import conv_array_text, extract_N_and_CA_backbone_atoms
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_surface_area_to_volume_ratio,
    calculate_sphericity, calculate_euler_characteristic, calculate_inradius,
    calculate_circumradius, calculate_hydrodynamic_radius
)
from tqdm import tqdm
from scipy.spatial import ConvexHull

# Initialize PDB parser
parser = PDBParser()

# Load peptide cleavage coordinates (per time point)
data_df = pd.read_csv(
    r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\pep_cleave_coordinates_10292023.csv",
    index_col=0
)
# Convert text arrays to numpy arrays
data_df = data_df.applymap(conv_array_text)

# Identify timepoint columns (all 'tryps_*')
time_cols = [col for col in data_df.columns if col.startswith('tryps_')]

# Preallocate storage for structural properties + new distance/breakpoint columns
num_proteins = len(data_df.index)
struct_props = {
    'density': np.zeros(num_proteins),
    'radius_of_gyration': np.zeros(num_proteins),
    'surface_area_to_volume_ratio': np.zeros(num_proteins),
    'sphericity': np.zeros(num_proteins),
    'euler_characteristic': np.zeros(num_proteins),
    'inradius': np.zeros(num_proteins),
    'circumradius': np.zeros(num_proteins),
    'hydrodynamic_radius': np.zeros(num_proteins)
}
# distance-to-centroid per time point
for col in time_cols:
# breaking point time in minutes    struct_props[f'distance_{col}'] = np.zeros(num_proteins)
struct_props['breaking_time'] = [None] * num_proteins

# Helper: parse breaking minute from column name
def parse_minutes(colname):
    m = re.search(r"_(\d+)min", colname)
    return int(m.group(1)) if m else None

# Collect PDB file paths by UniProt ID
def get_pdb_file_paths(folder_path):
    pdb_paths = {}
    pattern = re.compile(r"AF-(\w+)-F\d+-model_v4\.pdb")
    for subdir, _, files in tqdm(os.walk(folder_path)):
        match = pattern.match(os.path.basename(subdir))
        if match:
            uid = match.group(1)
            pdbs = [f for f in files if f.endswith('.pdb')]
            if pdbs:
                pdb_paths[uid] = os.path.join(subdir, pdbs[0])
    return pdb_paths

pdb_paths = get_pdb_file_paths(r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human")

# Loop over proteins
for idx, uniprot_id in enumerate(data_df.index):
    pdb_file = pdb_paths.get(uniprot_id)
    if not pdb_file or not os.path.isfile(pdb_file):
        continue

    # Parse structure
    structure = parser.get_structure(uniprot_id, pdb_file)

    # Compute structural properties
    struct_props['density'][idx] = calculate_density(structure)
    struct_props['radius_of_gyration'][idx] = calculate_radius_of_gyration(structure)
    struct_props['surface_area_to_volume_ratio'][idx] = calculate_surface_area_to_volume_ratio(structure)
    struct_props['sphericity'][idx] = calculate_sphericity(structure)
    struct_props['euler_characteristic'][idx] = calculate_euler_characteristic(structure)
    struct_props['inradius'][idx] = calculate_inradius(structure)
    struct_props['circumradius'][idx] = calculate_circumradius(structure)
    struct_props['hydrodynamic_radius'][idx] = calculate_hydrodynamic_radius(structure)

    # Calculate convex hull centroid from backbone points
    points = extract_N_and_CA_backbone_atoms(structure)
    hull = ConvexHull(points)
    centroid = points[hull.vertices].mean(axis=0)

    # Compute distances at each time point and detect breaking time
    distances = []
    for col in time_cols:
        coord = data_df.at[uniprot_id, col]
        dist = np.nan
        if coord is not None and len(coord) > 0:
            dist = np.linalg.norm(coord - centroid)
        struct_props[f'distance_{col}'][idx] = dist
        distances.append(dist)

    # Find global minimum index, but only count it as a break if there is a rise immediately afterwards
    break_time = None
    if np.nansum(~np.isnan(distances)) >= 2:               # ensure at least two valid points
        # index of the global minimum distance
        min_idx = int(np.nanargmin(distances))
        # check that it's not the last timepoint, and that the very next point is higher
        if min_idx < len(distances) - 1 and distances[min_idx + 1] > distances[min_idx]:
            break_time = parse_minutes(time_cols[min_idx])
    struct_props['breaking_time'][idx] = break_time


# Compile and save results
results_df = pd.DataFrame(struct_props, index=data_df.index)
results_df.to_csv(r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\structural_properties_with_distances.csv", index_label='uniprot_id')
