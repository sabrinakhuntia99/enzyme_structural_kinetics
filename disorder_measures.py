import csv
import os
import gzip
import requests
import pandas as pd
from calc_functions import (
    calculate_density, calculate_radius_of_gyration, calculate_sphericity,
    calculate_surface_area_to_volume_ratio, calculate_euler_characteristic,
    calculate_inradius, calculate_circumradius, calculate_hydrodynamic_radius,
    sequence_to_hydrophobicity_array
)
from pdb_functions import search_pdb_file
from Bio.PDB import PDBParser  # Ensure you have Biopython installed

# Define the path for the human proteome
human_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_human'

# Read DisProt data
df = pd.read_csv(r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot.tsv', sep='\t')

data = []

for _, row in df.iterrows():
    try:
        uni_id = row['acc']
        url = f"https://www.disprot.org/api/{uni_id}"
        response = requests.get(url)
        data_api = response.json()

        disorder_content = data_api.get("disorder_content")
        full_sequence = data_api.get("sequence")
        pdb_file_path = search_pdb_file(uni_id)

        if os.path.isfile(pdb_file_path):
            parser = PDBParser()
            structure = None

            # Load structure based on file type
            if pdb_file_path.endswith('.pdb'):
                structure = parser.get_structure("structure", pdb_file_path)
            elif pdb_file_path.endswith('.pdb.gz'):
                with gzip.open(pdb_file_path, 'rt') as gz_file:
                    structure = parser.get_structure("structure", gz_file)

            # Calculate the required properties
            density = calculate_density(structure)
            radius_of_gyration = calculate_radius_of_gyration(structure)
            sphericity = calculate_sphericity(structure)
            surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
            euler_characteristic = calculate_euler_characteristic(structure)
            inradius = calculate_inradius(structure)
            circumradius = calculate_circumradius(structure)
            hydrodynamic_radius = calculate_hydrodynamic_radius(structure)
            hydrophobicity_full = sequence_to_hydrophobicity_array(full_sequence)

            sequence_length = len(full_sequence)

            # Append the collected data
            data.append([
                uni_id, sequence_length, hydrophobicity_full, density,
                radius_of_gyration, sphericity, surface_area_to_volume_ratio,
                euler_characteristic, inradius, circumradius,
                hydrodynamic_radius, disorder_content
            ])
        else:
            print(f"PDB file not found for UniProt ID {uni_id}. Skipping...")

    except Exception as e:
        print(f"Error processing UniProt ID {uni_id}: {e}")

# Write output to a TSV file
output_file = "../data/disprot_with_measures.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        'uniprot_id', 'sequence_length', 'hydrophobicity_full', 'density',
        'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
        'euler_characteristic', 'inradius', 'circumradius',
        'hydrodynamic_radius', 'disorder_content'
    ])
    writer.writerows(data)