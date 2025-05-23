import numpy as np
from scipy.spatial import ConvexHull, distance
from Bio.SeqUtils import IsoelectricPoint
import ast  # Library for literal_eval function

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

def calculate_radius_of_gyration(structure):

    # Extract coordinates of all atoms
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])

    # Compute the center of mass
    center_of_mass = np.mean(atoms_coords, axis=0)

    # Compute the squared distances from the center of mass
    squared_distances = np.sum((atoms_coords - center_of_mass) ** 2, axis=1)

    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(np.mean(squared_distances))

    return radius_of_gyration

def calculate_sphericity(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    equivalent_radius = (3 * volume / (4 * np.pi))**(1/3)
    sphere_surface_area = 4 * np.pi * (equivalent_radius**2)
    sphericity = sphere_surface_area / surface_area
    return sphericity

def calculate_surface_area_to_volume_ratio(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume
    surface_area = hull.area
    return surface_area / volume


def calculate_euler_characteristic(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)

    num_vertices = len(hull.vertices)
    num_faces = len(hull.simplices)

    # Create a set to track unique edges
    edges = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)

    num_edges = len(edges)

    euler_characteristic = num_vertices - num_edges + num_faces
    return euler_characteristic
def calculate_inradius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    centroid = np.mean(atoms_coords[hull.vertices], axis=0)
    distances = np.linalg.norm(atoms_coords[hull.vertices] - centroid, axis=1)
    inradius = np.min(distances)
    return inradius

def calculate_circumradius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    centroid = np.mean(atoms_coords[hull.vertices], axis=0)
    distances = np.linalg.norm(centroid - atoms_coords[hull.vertices], axis=1)
    circumradius = np.max(distances)
    return circumradius

def calculate_hydrodynamic_radius(structure):
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])
    hull = ConvexHull(atoms_coords)
    volume = hull.volume

    a = (3 * volume / (4 * np.pi))**(1/3)  # Major axis
    b = np.sqrt(3 * volume / (4 * np.pi * a))  # Minor axis

    form_factor = a / b

    radius = form_factor * (3 * volume / (4 * np.pi))**(1/3)
    return radius


def sequence_to_hydrophobicity_array(sequence):
    hydrophobicity_sequence = ','.join([f'{hydrophobicity_values.get(aa, 0)}' for aa in sequence])
    hydrophobicity_sequence = f"({hydrophobicity_sequence})"
    arr = np.array(ast.literal_eval(hydrophobicity_sequence))
    return np.min(arr), np.max(arr), np.sum(arr), np.mean(arr)


def calculate_pI(sequence):
    pI = IsoelectricPoint(sequence).pi
    return pI


def calculate_cubic_spline_rmse(structure):
    import numpy as np
    from scipy.interpolate import CubicSpline
    from sklearn.metrics import mean_squared_error

    # Get the x, y, z coordinates of atoms
    atoms_coords = np.array([atom.coord for atom in structure.get_atoms()])

    # Filter out rows containing NaN values
    atoms_coords = atoms_coords[~np.isnan(atoms_coords).any(axis=1)]

    # Project onto XY plane (z=0)
    atoms_coords_2d = atoms_coords[:, :2]

    # Check if the number of data points is sufficient for interpolation
    if len(atoms_coords_2d) < 2:
        print("Insufficient data points for interpolation.")
        return np.nan

    # Sort the coordinates based on x-values
    sorted_indices = np.sort(atoms_coords_2d[:, 0])
    sorted_coords = atoms_coords_2d[sorted_indices]

    # Check if x-values are in increasing order
    if not np.all(np.diff(sorted_coords[:, 0]) > 0):
        print("X-values are not in increasing order.")
        return np.nan

    # Check for duplicate x-values
    unique_indices = np.unique(sorted_coords[:, 0], return_index=True)[1]
    if len(unique_indices) < len(sorted_coords):
        # Add a small perturbation to duplicate x-values
        perturbation = 1e-9
        sorted_coords[unique_indices[:-1], 0] += perturbation

    # Perform cubic spline interpolation
    try:
        cubic_spline = CubicSpline(sorted_coords[:, 0], sorted_coords[:, 1])
    except ValueError as e:
        print(f"Error in cubic spline interpolation: {e}")
        return np.nan

    # Calculate interpolated y values
    interpolated_y = cubic_spline(sorted_coords[:, 0])

    # Check for NaN values in the interpolated data
    if np.isnan(interpolated_y).any():
        print("NaN values encountered in interpolated data.")
        return np.nan

    # Calculate RMSE between original y values and interpolated y values
    rmse = np.sqrt(mean_squared_error(sorted_coords[:, 1], interpolated_y))

    return rmse


def calculate_average_plddt(structure):

    plddt_scores = []

    for atom in structure.get_atoms():
        # The pLDDT score is stored in the B-factor column
        plddt_scores.append(atom.bfactor)

    if plddt_scores:
        avg_plddt = np.mean(plddt_scores)
    else:
        avg_plddt = None  # Return None if no pLDDT scores were found

    return avg_plddt


def calculate_hurst_exponent(distances):
    """Calculate the Hurst exponent using the Rescaled Range method."""
    n = len(distances)
    if n < 2:
        return np.nan  # Not enough data

    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    deviations = distances - mean_distance
    cumulative_dev = np.cumsum(deviations)
    R = np.max(cumulative_dev) - np.min(cumulative_dev)
    S = np.std(distances)

    if S == 0:
        return np.nan  # Avoid division by zero

    R_S = R / S
    hurst_exponent = np.log(R_S) / np.log(n)

    return hurst_exponent