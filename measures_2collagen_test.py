import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull

# Import calculation functions
from calc_functions import (
    calculate_density, calculate_radius_of_gyration,
    calculate_surface_area_to_volume_ratio,
    calculate_circumradius, calculate_hydrodynamic_radius
)

# Function to calculate metrics for a given PDB file
def calculate_metrics(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)

    metrics = {
        'density': calculate_density(structure),
        'radius_of_gyration': calculate_radius_of_gyration(structure),
        'surface_area_to_volume_ratio': calculate_surface_area_to_volume_ratio(structure),
        'circumradius': calculate_circumradius(structure),
        'hydrodynamic_radius': calculate_hydrodynamic_radius(structure),
    }

    return metrics, structure

# Function to extract nitrogen coordinates from the structure
def extract_nitrogen_coordinates(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('N'):  # Only get nitrogen atoms
                    coords.append(residue['N'].get_coord())
    return np.array(coords)

# Function to extract all atom coordinates from the structure
def extract_all_coordinates(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:  # Iterate over all atoms in the residue
                    coords.append(atom.get_coord())
    return np.array(coords)

# Function to plot 3D structure with nitrogen backbone line and convex hull
def plot_3d_structure(all_coords, nitrogen_coords, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the background and axis colors to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Set the pane colors to black (background)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

    # Remove grid lines for a cleaner look
    ax.grid(False)

    # Plot the nitrogen backbone with a line
    if len(nitrogen_coords) > 1:
        ax.plot(nitrogen_coords[:, 0], nitrogen_coords[:, 1], nitrogen_coords[:, 2], color='cyan', linewidth=2, label='Nitrogen Backbone')

    # Calculate and plot the convex hull using nitrogen atoms only
    if len(nitrogen_coords) >= 4:  # Convex hull requires at least 4 points
        hull = ConvexHull(nitrogen_coords)
        for simplex in hull.simplices:
            ax.plot3D(nitrogen_coords[simplex, 0], nitrogen_coords[simplex, 1], nitrogen_coords[simplex, 2], 'r-', alpha=0.5)

    # Set axis labels and title with white color for visibility
    ax.set_title(title, color='white', fontsize=48)  # Title font size increased to 48
    ax.set_title(title, color='white')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')

    # Set tick colors to white for visibility
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')


    plt.show()

# Define file paths for the two collagen structures
collagen_type_2_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/collagen_exp/collagen 2A vWC domain (5NIR).pdb'
collagen_type_3_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/collagen_exp/type III collagen triple helix (4GYX).pdb'

# Calculate metrics for both types of collagen
metrics_type_2, structure_type_2 = calculate_metrics(collagen_type_2_path)
metrics_type_3, structure_type_3 = calculate_metrics(collagen_type_3_path)

# Prepare data for plotting
labels = ['Density', 'Radius of Gyration', 'Surface Area to Volume Ratio', 'Circumradius', 'Hydrodynamic Radius']
metrics_values_type_2 = [metrics_type_2['density'], metrics_type_2['radius_of_gyration'],
                         metrics_type_2['surface_area_to_volume_ratio'],
                         metrics_type_2['circumradius'], metrics_type_2['hydrodynamic_radius']]
metrics_values_type_3 = [metrics_type_3['density'], metrics_type_3['radius_of_gyration'],
                         metrics_type_3['surface_area_to_volume_ratio'],
                         metrics_type_3['circumradius'], metrics_type_3['hydrodynamic_radius']]

# Set up the bar plot
x = np.arange(len(labels))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for Type 2 and Type 3 collagen
bars_type_2 = ax.bar(x - width / 2, metrics_values_type_2, width, label='Type 2 Collagen')
bars_type_3 = ax.bar(x + width / 2, metrics_values_type_3, width, label='Type 3 Collagen')

# Labeling the plot
ax.set_ylabel('Values')
ax.set_title('Comparison of Structural Properties of Collagen Types')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to add value labels on top of the bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Apply labels
autolabel(bars_type_2)
autolabel(bars_type_3)

plt.tight_layout()
plt.show()

# Extract and plot coordinates
coords_type_2 = extract_all_coordinates(structure_type_2)
coords_type_3 = extract_all_coordinates(structure_type_3)

nitrogen_coords_type_2 = extract_nitrogen_coordinates(structure_type_2)
nitrogen_coords_type_3 = extract_nitrogen_coordinates(structure_type_3)

plot_3d_structure(coords_type_2, nitrogen_coords_type_2, 'Nitrogen Backbone Convex Hull of collagen 2A vWC domain (5NIR)')
plot_3d_structure(coords_type_3, nitrogen_coords_type_3, 'Nitrogen Backbone Convex Hull of type III collagen triple helix (4GYX)')
