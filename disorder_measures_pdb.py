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
from Bio.PDB import PDBParser  # Ensure you have Biopython installed

# Define the path for the PDB files
pdb_folder_path = 'C:/Users/Sabrina/Documents/GitHub/protein_structural_kinetics/pdb_list'

# Read DisProt data
df = pd.read_csv(r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot.tsv', sep='\t')

data = []


def uniprot_to_pdb(uniprot_id):
    """Fetch PDB IDs associated with a UniProt ID."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data for UniProt ID: {uniprot_id}")
        return []

    pdb_ids = []
    for line in response.text.splitlines():
        if line.startswith("DR   PDB;"):
            pdb_id = line.split(";")[1].strip()
            pdb_ids.append(pdb_id)

    return pdb_ids


for _, row in df.iterrows():
    try:
        uni_id = row['acc']
        url = f"https://www.disprot.org/api/{uni_id}"
        response = requests.get(url)
        data_api = response.json()

        disorder_content = data_api.get("disorder_content")
        full_sequence = data_api.get("sequence")

        # Convert UniProt ID to PDB IDs
        pdb_ids = uniprot_to_pdb(uni_id)

        for pdb_id in pdb_ids:
            pdb_file_path = os.path.join(pdb_folder_path, f"{pdb_id}.pdb")

            if os.path.isfile(pdb_file_path):
                parser = PDBParser()
                structure = None

                # Load structure based on file type
                structure = parser.get_structure("structure", pdb_file_path)

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
                # Break after processing the first found PDB file
                break
            else:
                print(f"PDB file not found for UniProt ID {uni_id} (PDB ID: {pdb_id}). Skipping...")

    except Exception as e:
        print(f"Error processing UniProt ID {uni_id}: {e}")

# Write output to a TSV file
output_file = "disprot_with_measures_pdb_only.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        'uniprot_id', 'sequence_length', 'hydrophobicity_full', 'density',
        'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
        'euler_characteristic', 'inradius', 'circumradius',
        'hydrodynamic_radius', 'disorder_content'
    ])
    writer.writerows(data)
