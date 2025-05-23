from Bio.PDB import PDBParser
from calc_functions import calculate_radius_of_gyration, calculate_hydrodynamic_radius

# Define the path to the specific PDB file
pdb_file_path = r"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\1ao6.pdb"  # Update this path if needed

# Initialize PDBParser
parse = PDBParser()

# Load the structure from the PDB file
structure = parse.get_structure("1ao6", pdb_file_path)

# Calculate the radius of gyration
radius_of_gyration = calculate_radius_of_gyration(structure)
hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

# Print the result
print(f"Radius of Gyration for 1AO6.pdb: {radius_of_gyration}")
print(f"Hydrodynamic Radius for 1AO6.pdb: {hydrodynamic_radius}")

