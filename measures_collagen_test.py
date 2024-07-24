import os
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

def calculate_sphericity(structure):
    atoms_coords = [atom.coord for atom in structure.get_atoms()]
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    equivalent_radius = (3 * volume / (4 * np.pi))**(1/3)
    return (np.pi**(1/3)) * ((6 * equivalent_radius)**(2/3)) / surface_area

def calculate_surface_area_to_volume_ratio(structure):
    atoms_coords = [atom.coord for atom in structure.get_atoms()]
    hull = ConvexHull(atoms_coords)
    surface_area = hull.area
    volume = hull.volume
    return surface_area / volume

def calculate_isoperimetric_quotient(structure):
    atoms_coords = [atom.coord for atom in structure.get_atoms()]
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    equivalent_radius = ((3 * volume) / (4 * np.pi))**(1/3)
    equivalent_sphere_volume = (4/3) * np.pi * equivalent_radius**3
    return volume / equivalent_sphere_volume

def calculate_radius_of_gyration(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    center_of_mass = np.mean(atoms_coords, axis=0)
    distances = distance.cdist(atoms_coords, [center_of_mass])
    radius_of_gyration = np.sqrt(np.mean(distances**2))
    return radius_of_gyration

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

def calculate_metrics(folder_path):
    sphericities = []
    surface_area_to_volume_ratios = []
    isoperimetric_quotients = []
    radii_of_gyration = []
    densities = []

    for subdir, _, files in os.walk(folder_path):
        pdb_files = [f for f in files if f.endswith('.pdb')]
        if pdb_files:
            for pdb_file in pdb_files:
                full_pdb_path = os.path.join(subdir, pdb_file)

                parser = PDBParser()
                structure = parser.get_structure("structure", full_pdb_path)

                sphericity = calculate_sphericity(structure)
                surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
                isoperimetric_quotient = calculate_isoperimetric_quotient(structure)
                radius_of_gyration = calculate_radius_of_gyration(structure)
                density = calculate_density(structure)

                sphericities.append(sphericity)
                surface_area_to_volume_ratios.append(surface_area_to_volume_ratio)
                isoperimetric_quotients.append(isoperimetric_quotient)
                radii_of_gyration.append(radius_of_gyration)
                densities.append(density)

    return sphericities, surface_area_to_volume_ratios, isoperimetric_quotients, radii_of_gyration, densities

# Define folder paths for regular collagen and collagen triple helix
regular_collagen_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/collagen_exp/collagen 2A vWC domain (5NIR).pdb'
collagen_triple_helix_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/collagen_exp/type III collagen triple helix (4GYX).pdb'

parser = PDBParser()

# Parse regular collagen structure
regular_structure = parser.get_structure("structure", regular_collagen_path)

# Calculate metrics for regular collagen
regular_sphericity = calculate_sphericity(regular_structure)
regular_surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(regular_structure)
regular_isoperimetric_quotient = calculate_isoperimetric_quotient(regular_structure)
regular_radius_of_gyration = calculate_radius_of_gyration(regular_structure)
regular_density = calculate_density(regular_structure)

# Parse collagen triple helix structure
helix_structure = parser.get_structure("structure", collagen_triple_helix_path)

# Calculate metrics for collagen triple helix
helix_sphericity = calculate_sphericity(helix_structure)
helix_surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(helix_structure)
helix_isoperimetric_quotient = calculate_isoperimetric_quotient(helix_structure)
helix_radius_of_gyration = calculate_radius_of_gyration(helix_structure)
helix_density = calculate_density(helix_structure)

# Plot results
labels = ['Sphericity', 'Surface Area to Volume Ratio', 'Isoperimetric Quotient', 'Radius of Gyration', 'Density']
regular_metrics = [regular_sphericity, regular_surface_area_to_volume_ratio, regular_isoperimetric_quotient, regular_radius_of_gyration, regular_density]
helix_metrics = [helix_sphericity, helix_surface_area_to_volume_ratio, helix_isoperimetric_quotient, helix_radius_of_gyration, helix_density]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(x - width/2, regular_metrics, width, label='Regular Collagen')
bar2 = ax.bar(x + width/2, helix_metrics, width, label='Collagen Triple Helix')

ax.set_ylabel('Values')
ax.set_title('Comparison of Collagen Structures')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)

plt.tight_layout()
plt.show()
