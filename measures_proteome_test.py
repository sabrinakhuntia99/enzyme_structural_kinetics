import os
import gzip
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import distance



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

def calculate_density(structure):
    atoms = list(structure.get_atoms())
    total_mass = sum(atomic_masses.get(atom.element, 0) for atom in atoms)

    atoms_coords = [atom.coord for atom in atoms]
    hull = ConvexHull(atoms_coords)
    volume = hull.volume  # in cubic Angstroms

    # Convert volume to cubic centimeters (1 Å³ = 1e-24 cm³)
    volume_cm3 = volume * 1e-24

    # Convert mass from Daltons to grams (1 Da = 1.66054e-24 g)
    mass_g = total_mass * 1.66054e-24

    # Calculate density in g/cm³
    density = mass_g / volume_cm3
    return density

def calculate_radius_of_gyration(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    center_of_mass = np.mean(atoms_coords, axis=0)
    distances = distance.cdist(atoms_coords, [center_of_mass])
    radius_of_gyration = np.sqrt(np.mean(distances**2))
    return radius_of_gyration

def calculate_surface_area_to_volume_ratio(structure):
    atoms_coords = [atom.coord for atom in structure.get_atoms()]
    hull = ConvexHull(atoms_coords)
    surface_area = hull.area
    volume = hull.volume
    return surface_area / volume

def calculate_sphericity(structure):
    atoms_coords = [atom.coord for atom in structure.get_atoms()]
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    equivalent_radius = (3 * volume / (4 * np.pi))**(1/3)
    return (np.pi**(1/3)) * ((6 * equivalent_radius)**(2/3)) / surface_area


def calculate_metrics(folder_path):
    surface_area_to_volume_ratios = []
    sphericities = []

    densities = []
    radii_of_gyration = []

    for subdir, _, files in os.walk(folder_path):
        pdb_files = [f for f in files if f.endswith('.pdb') or f.endswith('.pdb.gz')]
        if pdb_files:
            for pdb_file in pdb_files:
                full_pdb_path = os.path.join(subdir, pdb_file)

                parser = PDBParser()
                if pdb_file.endswith('.pdb'):
                    structure = parser.get_structure("structure", full_pdb_path)
                elif pdb_file.endswith('.pdb.gz'):
                    with gzip.open(full_pdb_path, 'rt') as gz_file:
                        structure = parser.get_structure("structure", gz_file)

                surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
                sphericity = calculate_sphericity(structure)
                density = calculate_density(structure)
                radius_of_gyration = calculate_radius_of_gyration(structure)

                surface_area_to_volume_ratios.append(surface_area_to_volume_ratio)
                sphericities.append(sphericity)
                densities.append(density)
                radii_of_gyration.append(radius_of_gyration)

    return surface_area_to_volume_ratios, sphericities, densities, radii_of_gyration

# Define folder paths for tuberculosis, human, antibodies, and zebrafish proteomes
#human_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_human'
#mtb_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_mtb'
#antibodies_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/antibodies_human'
zebrafish_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_zebrafish'

# Calculate metrics for tuberculosis proteome
#tb_sphericities, tb_surface_area_to_volume_ratios, tb_radii_of_gyration, tb_densities = calculate_metrics(mtb_folder_path)

# Calculate metrics for human proteome
#human_sphericities, human_surface_area_to_volume_ratios, human_radii_of_gyration, human_densities = calculate_metrics(human_folder_path)

# Calculate metrics for antibodies
#antibodies_sphericities, antibodies_surface_area_to_volume_ratios,antibodies_radii_of_gyration, antibodies_densities = calculate_metrics(antibodies_folder_path)

# Calculate metrics for zebrafish proteome
zebrafish_sphericities, zebrafish_surface_area_to_volume_ratios, zebrafish_radii_of_gyration, zebrafish_densities = calculate_metrics(zebrafish_folder_path)

plt.figure(figsize=(12, 10))

# Plot histograms for surface area to volume ratio
plt.subplot(2, 2, 1)
#plt.hist(tb_surface_area_to_volume_ratios, bins=40, color='orange', alpha=0.5, label='Tuberculosis')
#plt.hist(human_surface_area_to_volume_ratios, bins=40, color='blue', alpha=0.5, label='Human')
#plt.hist(antibodies_surface_area_to_volume_ratios, bins=40, color='green', alpha=0.5, label='Antibodies')
plt.hist(zebrafish_surface_area_to_volume_ratios, bins=40, color='red', alpha=0.5, label='Zebrafish')
plt.xlabel('Surface Area to Volume Ratio')
plt.ylabel('Frequency')
plt.legend()

# Plot histograms for sphericity
plt.subplot(2, 2, 2)
#plt.hist(tb_sphericities, bins=40, color='orange', alpha=0.5, label='Tuberculosis')
#plt.hist(human_sphericities, bins=40, color='blue', alpha=0.5, label='Human')
#plt.hist(antibodies_sphericities, bins=40, color='green', alpha=0.5, label='Antibodies')
plt.hist(zebrafish_sphericities, bins=40, color='red', alpha=0.5, label='Zebrafish')
plt.xlabel('Sphericity')
plt.ylabel('Frequency')
plt.legend()

# Plot histograms for density
plt.subplot(2, 2, 3)
#plt.hist(tb_densities, bins=40, color='orange', alpha=0.5, label='Tuberculosis')
#plt.hist(human_densities, bins=40, color='blue', alpha=0.5, label='Human')
#plt.hist(antibodies_densities, bins=40, color='green', alpha=0.5, label='Antibodies')
plt.hist(zebrafish_densities, bins=40, color='red', alpha=0.5, label='Zebrafish')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.legend()

# Plot histograms for radius of gyration
plt.subplot(2, 2, 4)
#plt.hist(tb_radii_of_gyration, bins=40, color='orange', alpha=0.5, label='Tuberculosis')
#plt.hist(human_radii_of_gyration, bins=40, color='blue', alpha=0.5, label='Human')
#plt.hist(antibodies_radii_of_gyration, bins=40, color='green', alpha=0.5, label='Antibodies')
plt.hist(zebrafish_radii_of_gyration, bins=40, color='red', alpha=0.5, label='Zebrafish')
plt.xlabel('Radius of Gyration')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
