import csv
import os
import gzip
import numpy as np
import requests
import pandas as pd
import ast
from calc_functions import calculate_density, calculate_radius_of_gyration, calculate_disorder, calculate_sphericity, calculate_surface_area_to_volume_ratio, calculate_euler_characteristic, calculate_inradius, calculate_circumradius, calculate_hydrodynamic_radius, sequence_to_hydrophobicity_array, calculate_pI, calculate_cubic_spline_rmse
from pdb_functions import extract_sequence_from_pdb, search_pdb_file


human_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_human'

# Read DisProt data
df = pd.read_csv('disprot.tsv', sep='\t')
df = df[0:1]

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
            if pdb_file_path.endswith('.pdb'):
                structure = parser.get_structure("structure", pdb_file_path)

            elif pdb_file_path.endswith('.pdb.gz'):
                with gzip.open(pdb_file_path, 'rt') as gz_file:
                    structure = parser.get_structure("structure", gz_file)

            density = calculate_density(structure)
            radius_of_gyration = calculate_radius_of_gyration(structure)
            sphericity = calculate_sphericity(structure)
            surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
            euler_characteristic = calculate_euler_characteristic(structure)
            inradius = calculate_inradius(structure)
            circumradius = calculate_circumradius(structure)
            hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

            region_sequence = row['region_sequence']
            hydrophobicity_region = sequence_to_hydrophobicity_array(region_sequence)

            term_namespace = map_namespace_to_number(row['term_namespace'])

            sequence_length = len(full_sequence)
            hydrophobicity_full = sequence_to_hydrophobicity_array(full_sequence)

            #pI_full = calculate_pI(full_sequence)
            #pI_region = calculate_pI(region_sequence)
            lagrange_rmse = calculate_cubic_spline_rmse(structure)

            data.append([uni_id, row['start'], row['end'], sequence_length, hydrophobicity_full, hydrophobicity_region, density, radius_of_gyration, sphericity,
                         surface_area_to_volume_ratio, term_namespace, euler_characteristic, inradius,
                         circumradius, hydrodynamic_radius, lagrange_rmse, disorder_content])

        else:
            print(f"PDB file not found for UniProt ID {uni_id}. Skipping...")

    except Exception as e:
        print(f"Error processing UniProt ID {uni_id}: {e}")

output_file = "disprot_060424.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        ['uniprot_id', 'start', 'end', 'sequence_length', 'hydrophobicity_full', 'hydrophobicity_region', 'density',
         'radius_of_gyration', 'sphericity',
         'surface_area_to_volume_ratio', 'term_namespace', 'euler_characteristic', 'inradius',
         'circumradius', 'hydrodynamic_radius', 'lagrange_rmse', 'disorder_content'])

    writer.writerows(data)