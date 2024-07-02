import csv
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from scipy.spatial import ConvexHull, distance
import os
import gzip
import numpy as np
import requests
import pandas as pd

# Atomic masses in Dalton (g/mol)
atomic_masses = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'P': 30.974,
    'S': 32.06,
    'SE': 78.96,
}

# Hydrophobicity values for amino acids
hydrophobicity_values = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

# Function to calculate density
def calculate_density(structure):
    atoms = list(structure.get_atoms())
    total_mass = sum(atomic_masses.get(atom.element, 0) for atom in atoms)

    atoms_coords = np.array([atom.coord for atom in atoms])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume

    volume_cm3 = volume * 1e-24
    mass_g = total_mass * 1.66054e-24
    density = mass_g / volume_cm3
    return density

# Function to calculate radius of gyration
def calculate_radius_of_gyration(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    center_of_mass = np.mean(atoms_coords, axis=0)
    distances = distance.cdist(atoms_coords, [center_of_mass])
    radius_of_gyration = np.sqrt(np.mean(distances**2))
    return radius_of_gyration

# Function to calculate sphericity
def calculate_sphericity(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    equivalent_radius = (3 * volume / (4 * np.pi))**(1/3)
    sphere_surface_area = 4 * np.pi * (equivalent_radius**2)
    sphericity = sphere_surface_area / surface_area
    return sphericity

# Function to calculate surface area to volume ratio
def calculate_surface_area_to_volume_ratio(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    return surface_area / volume

# Function to calculate Euler characteristic
def calculate_euler_characteristic(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    num_vertices = len(hull.vertices)
    num_edges = len(hull.simplices) * 3
    num_faces = len(hull.simplices)
    euler_characteristic = num_vertices - num_edges + num_faces
    return euler_characteristic

# Function to calculate inradius
def calculate_inradius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    centroid = np.mean(atoms_coords[hull.vertices], axis=0)
    distances = np.linalg.norm(atoms_coords[hull.vertices] - centroid, axis=1)
    inradius = np.min(distances)
    return inradius

# Function to calculate circumradius
def calculate_circumradius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    centroid = np.mean(atoms_coords[hull.vertices], axis=0)
    distances = np.linalg.norm(centroid - atoms_coords[hull.vertices], axis=1)
    circumradius = np.max(distances)
    return circumradius

# Function to calculate hydrodynamic radius
def calculate_hydrodynamic_radius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume

    # Calculate the axial ratio
    a = (3 * volume / (4 * np.pi))**(1/3)  # Major axis
    b = np.sqrt(3 * volume / (4 * np.pi * a))  # Minor axis

    form_factor = a / b  # Axial ratio (a/b)

    radius = form_factor * (3 * volume / (4 * np.pi))**(1/3)  # Hydrodynamic radius formula
    return radius

# Function to search for PDB file based on UniProt ID
def search_pdb_file(uniprot_id):
    uniprot_id = uniprot_id.split('-')[0]  # Extract UniProt ID without isoform information
    return r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human\AF-{}-F1-model_v4.pdb\AF-{}-F1-model_v4.pdb".format(uniprot_id, uniprot_id)

# Function to extract amino acid sequence from PDB structure
def extract_sequence(structure):
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        return str(pp.get_sequence())

# Function to calculate the length of the amino acid sequence or hydrophobicity array
def calculate_sequence_length(sequence):
    return len(sequence)

# Read DisProt data
df = pd.read_csv('pep_cleave.csv', index_col=0)

# Initialize lists to store metrics and disorder content
data = []

# Loop through each UniProt ID in DisProt data
for uni_id in df.index:
    if df.loc[uni_id].notna().any():
        try:
            pdb_file_path = search_pdb_file(uni_id)
            if os.path.isfile(pdb_file_path):
                parser = PDBParser()
                if pdb_file_path.endswith('.pdb'):
                    structure = parser.get_structure("structure", pdb_file_path)
                elif pdb_file_path.endswith('.pdb.gz'):
                    with gzip.open(pdb_file_path, 'rt') as gz_file:
                        structure = parser.get_structure("structure", gz_file)

                # Calculate metrics
                density = calculate_density(structure)
                radius_of_gyration = calculate_radius_of_gyration(structure)
                sphericity = calculate_sphericity(structure)
                surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
                euler_characteristic = calculate_euler_characteristic(structure)
                inradius = calculate_inradius(structure)
                circumradius = calculate_circumradius(structure)
                hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

                # Extract amino acid sequence
                region_sequence = extract_sequence(structure)

                # Calculate sequence length
                sequence_length = calculate_sequence_length(region_sequence)

                # Append data to the list
                data.append([uni_id, region_sequence, density, radius_of_gyration, sphericity,
                             surface_area_to_volume_ratio, euler_characteristic, inradius,
                             circumradius, hydrodynamic_radius, sequence_length])
            else:
                print(f"PDB file not found for UniProt ID {uni_id}. Skipping...")
        except Exception as e:
            print(f"Error processing UniProt ID {uni_id}: {e}")

# Write the processed data to a TSV file
output_file = "processed_proteolysis.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(['uniprot_id', 'region_sequence', 'density', 'radius_of_gyration', 'sphericity',
                     'surface_area_to_volume_ratio', 'euler_characteristic', 'inradius',
                     'circumradius', 'hydrodynamic_radius', 'sequence_length'])
    writer.writerows(data)
