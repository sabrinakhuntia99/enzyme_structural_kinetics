import csv
import os
import gzip
import pandas as pd
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_sphericity,
    calculate_surface_area_to_volume_ratio, calculate_euler_characteristic,
    calculate_inradius, calculate_circumradius, calculate_hydrodynamic_radius
)
from pdb_functions import search_pdb_file
from Bio.PDB import PDBParser

# Define the path for the human proteome and the matrisome dataset
human_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_human'
matrisome_file_path = r'C:/Users/Sabrina/Documents/GitHub/collagen/matrisome.csv'

# Read the matrisome data to get UniProt IDs
matrisome_df = pd.read_csv(matrisome_file_path, encoding='ISO-8859-1')  # or use 'latin1'

# Extract UniProt IDs (assuming the column name is 'UniProt_IDs')
uni_ids = matrisome_df['UniProt_IDs'].dropna().unique().tolist()

data_ecm = []
data_collagens = []

for uni_id in uni_ids:
    # Split multiple UniProt IDs if present
    for uid in uni_id.split(':'):
        uid = uid.strip()  # Remove any leading/trailing whitespace
        try:
            pdb_file_path = search_pdb_file(uid)  # Replace with your function to get the PDB file path

            if os.path.isfile(pdb_file_path):
                parser = PDBParser()
                structure = None

                # Load structure based on file type
                if pdb_file_path.endswith('.pdb'):
                    structure = parser.get_structure("structure", pdb_file_path)
                elif pdb_file_path.endswith('.pdb.gz'):
                    with gzip.open(pdb_file_path, 'rt') as gz_file:
                        structure = parser.get_structure("structure", gz_file)

                # Extract full sequence from the structure
                full_sequence = ''.join(
                    residue.get_resname() for model in structure for chain in model for residue in chain
                    if residue.id[0] == ' '  # Exclude heteroatoms
                )

                # Calculate the required properties
                density = calculate_density(structure)
                radius_of_gyration = calculate_radius_of_gyration(structure)
                sphericity = calculate_sphericity(structure)
                surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
                euler_characteristic = calculate_euler_characteristic(structure)
                inradius = calculate_inradius(structure)
                circumradius = calculate_circumradius(structure)
                hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

                sequence_length = len(full_sequence)

                # Prepare the data row
                data_row = [
                    sequence_length, density,
                    radius_of_gyration, sphericity,
                    surface_area_to_volume_ratio, inradius,
                    circumradius, hydrodynamic_radius
                ]

                # Append to appropriate dataset
                data_ecm.append(data_row)

                # Check if the current entry is a collagen
                if 'Collagens' in matrisome_df.loc[matrisome_df['UniProt_IDs'] == uni_id, 'Matrisome Category'].values:
                    data_collagens.append(data_row)

            else:
                continue

        except Exception as e:
            print(f"Error processing UniProt ID {uid}: {e}")
            continue

# Write output to TSV files
output_file_ecm = "../data/ecm_without_disorder.tsv"
output_file_collagens = "../data/collagens_without_disorder.tsv"

with open(output_file_ecm, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        'sequence_length', 'density',
        'radius_of_gyration', 'sphericity',
        'surface_area_to_volume_ratio', 'inradius',
        'circumradius', 'hydrodynamic_radius'
    ])
    writer.writerows(data_ecm)

with open(output_file_collagens, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        'sequence_length', 'density',
        'radius_of_gyration', 'sphericity',
        'surface_area_to_volume_ratio', 'inradius',
        'circumradius', 'hydrodynamic_radius'
    ])
    writer.writerows(data_collagens)
