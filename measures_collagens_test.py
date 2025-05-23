import csv
import os
import gzip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from calc_functions import (calculate_density, calculate_radius_of_gyration, calculate_sphericity,
                            calculate_surface_area_to_volume_ratio, calculate_euler_characteristic,
                            calculate_inradius, calculate_circumradius, calculate_hydrodynamic_radius)
from pdb_functions import extract_sequence_from_pdb, search_pdb_file

# List of collagen UniProt IDs
collagens = ['Q03692', 'Q5QPC7', 'Q5QPC8', 'C9JMN2', 'H7C381', 'P12107', 'A2AAS7', 'H0Y3B3', 'H0Y3B5',
             'H0Y3M9', 'H0YHM5', 'H0YHM9', 'H0YIS1', 'P13942', 'Q4VXY6', 'D6RGG3', 'H0Y4P7', 'H0Y5N9',
             'H0Y991', 'Q99715', 'E7ES46', 'E7ES47', 'E7ES49', 'E7ES50', 'E7ES51', 'E7ES55', 'E7ES56',
             'E7EX21', 'E9PEG9', 'G5E987', 'H7BYT9', 'H7BZB6', 'Q5TAT6', 'H0YBB2', 'J3QT75', 'J3QT83',
             'Q05707', 'Q4G0W3', 'P39059', 'A6NCT7', 'A6NDR9', 'H7BY97', 'H7BZL8', 'H7C3F0', 'Q07092',
             'A2A2Y8', 'H0Y420', 'Q9UMD9', 'H7BXV5', 'H7C457', 'P39060', 'Q14993', 'Q5JVU1', 'I3L3H7',
             'P02452', 'P08123', 'B7ZBI4', 'B7ZBI5', 'Q9P218', 'A6PVD9', 'F5GZK2', 'H0Y4C9', 'H0YDH6',
             'Q96P44', 'H0YAX7', 'Q8NFW1', 'Q86Y22', 'E9PNK8', 'F8WDM8', 'H0YCZ7', 'Q17RW2', 'A8MWQ5',
             'D6R8Y2', 'E9PNV9', 'H0YAE1', 'Q9BXS0', 'C9JPW4', 'Q96A83', 'H0YD40', 'Q5T1U7', 'Q8IZC6',
             'H7BZU0', 'H7C3P2', 'Q2UY09', 'P02458', 'E7ENY8', 'H7C435', 'P02461', 'F5H5K0', 'P02462',
             'A2A352', 'P08572', 'H7BXM4', 'Q01955', 'J3KNM7', 'P53420', 'H0Y998', 'H0Y9H0', 'H0Y9R8',
             'P29400', 'A8MXH5', 'B4DZ39', 'F5H3Q5', 'F5H851', 'Q14031', 'H7BY82', 'P20908', 'P05997',
             'P25940', 'P12109', 'C9JH44', 'H7C0M5', 'P12110', 'C9JNG9', 'E7ENL6', 'E9PCV6', 'I3L392',
             'P12111', 'A8TX70', 'E9PAL5', 'F8W8G8', 'H0Y393', 'H0Y935', 'H0Y9T2', 'A6NMZ7', 'F8W6Y7',
             'H0Y940', 'H0YA33', 'C9JBL3', 'Q02388', 'C9JTN9', 'P27658', 'E9PP49', 'P25067', 'A6NEQ6',
             'P20849', 'B1AKJ1', 'B1AKJ3', 'H0Y409', 'Q14055', 'Q14050', 'Q4VXW1']


# Function to process PDB files and extract required properties
def process_pdb_files(uniprot_ids):
    data = []
    for uniprot_id in uniprot_ids:
        try:
            # Search for the PDB file using the UniProt ID
            pdb_file_path = search_pdb_file(uniprot_id)

            if os.path.isfile(pdb_file_path):
                parser = PDBParser()

                # Check if the file is in .pdb or .pdb.gz format
                if pdb_file_path.endswith('.pdb'):
                    structure = parser.get_structure("structure", pdb_file_path)
                elif pdb_file_path.endswith('.pdb.gz'):
                    with gzip.open(pdb_file_path, 'rt') as gz_file:
                        structure = parser.get_structure("structure", gz_file)

                # Perform calculations
                density = calculate_density(structure)
                radius_of_gyration = calculate_radius_of_gyration(structure)
                sphericity = calculate_sphericity(structure)
                surface_area_to_volume_ratio = calculate_surface_area_to_volume_ratio(structure)
                euler_characteristic = calculate_euler_characteristic(structure)
                inradius = calculate_inradius(structure)
                circumradius = calculate_circumradius(structure)
                hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

                # Extract sequence from the PDB file and compute its length
                full_sequence = extract_sequence_from_pdb(pdb_file_path)
                sequence_length = len(full_sequence)

                # Append the results to the data list
                data.append([uniprot_id, sequence_length, density, radius_of_gyration, sphericity,
                             surface_area_to_volume_ratio, euler_characteristic, inradius, circumradius,
                             hydrodynamic_radius])

            else:
                print(f"PDB file not found for UniProt ID {uniprot_id}. Skipping...")

        except Exception as e:
            print(f"Error processing UniProt ID {uniprot_id}: {e}")

    return data


# Process collagen data
collagen_data = process_pdb_files(collagens)

# Convert to DataFrame for easier manipulation
columns = ['uniprot_id', 'sequence_length', 'density', 'radius_of_gyration', 'sphericity',
           'surface_area_to_volume_ratio', 'euler_characteristic', 'inradius', 'circumradius',
           'hydrodynamic_radius']
df = pd.DataFrame(collagen_data, columns=columns)

# Save the results to a TSV file
output_file = "collagens_data.tsv"
df.to_csv(output_file, sep='\t', index=False)
print("Data processing complete. Results saved to", output_file)

# Plot distributions for each property
properties = ['density', 'radius_of_gyration', 'sphericity', 'surface_area_to_volume_ratio',
              'euler_characteristic', 'inradius', 'circumradius', 'hydrodynamic_radius']

for prop in properties:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=prop, fill=True)
    plt.title(f"Distribution of {prop} for Collagen Proteins")
    plt.xlabel(prop)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f"{prop}_distribution.png")  # Save each plot as an image file
    plt.show()
