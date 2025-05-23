import csv
import gzip
import requests
from pdb_functions import search_pdb_file
from Bio.PDB import PDBParser
import pandas as pd
import os
from calc_functions import calculate_hydrodynamic_radius, calculate_euler_characteristic, calculate_average_plddt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Path to your local directory (adjust if necessary)
human_folder_path = 'C:/Users/Sabrina/PycharmProjects/intrinsic_disorder/proteome_human'

# Read DisProt data
df = pd.read_csv('C:/Users/Sabrina/Documents/GitHub/protein_structural_kinetics/data/disprot.tsv', sep='\t')

# Lists to store Hydrodynamic Radius, disorder content, Euler characteristic, and average pLDDT
hydrodynamic_radius_list = []
disorder_content_list = []
euler_characteristic_list = []
avg_plddt_list = []  # New list for average pLDDT

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

            # Calculate Hydrodynamic Radius
            hydrodynamic_radius = calculate_hydrodynamic_radius(structure)

            # Calculate Euler characteristic
            euler_characteristic = calculate_euler_characteristic(structure)

            # Calculate average pLDDT (assuming the function is implemented in calc_functions)
            avg_plddt = calculate_average_plddt(structure)

            # Append the results for this protein
            data.append([uni_id, hydrodynamic_radius, disorder_content, euler_characteristic, avg_plddt])

            # Store the Hydrodynamic Radius, disorder content, Euler characteristic, and avg pLDDT in the respective lists
            hydrodynamic_radius_list.append(hydrodynamic_radius)
            disorder_content_list.append(disorder_content)
            euler_characteristic_list.append(euler_characteristic)
            avg_plddt_list.append(avg_plddt)  # Add average pLDDT to the list

        else:
            print(f"PDB file not found for UniProt ID {uni_id}. Skipping...")

    except Exception as e:
        print(f"Error processing UniProt ID {uni_id}: {e}")

# Output results to a TSV file
output_file = "../data/disprot_hydro_radius_disorder_euler_avgplddt.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(['uniprot_id', 'hydrodynamic_radius', 'disorder_content', 'euler_characteristic', 'avg_plddt'])
    writer.writerows(data)

# Calculate Pearson correlation between Hydrodynamic Radius and disorder content
if hydrodynamic_radius_list and disorder_content_list:
    correlation_rg_disorder, p_value_rg_disorder = pearsonr(hydrodynamic_radius_list, disorder_content_list)
    print(f"Pearson correlation between Hydrodynamic Radius and disorder content: {correlation_rg_disorder:.4f}")
    print(f"P-value: {p_value_rg_disorder:.4f}")
else:
    print("Not enough data to calculate Pearson correlation for Hydrodynamic Radius and disorder content.")

# Plot disorder content vs. Hydrodynamic Radius
plt.figure(figsize=(10, 6))
plt.scatter(disorder_content_list, hydrodynamic_radius_list, alpha=0.7, edgecolors='w', s=100)
plt.title('Disorder Content vs. Hydrodynamic Radius')
plt.xlabel('Disorder Content')
plt.ylabel('Hydrodynamic Radius')
plt.grid(True)
plt.show()

# Plot disorder content vs. average pLDDT
plt.figure(figsize=(10, 6))
plt.scatter(disorder_content_list, avg_plddt_list, alpha=0.7, edgecolors='w', s=100, color='r')
plt.title('Disorder Content vs. Average pLDDT')
plt.xlabel('Disorder Content')
plt.ylabel('Average pLDDT')
plt.grid(True)
plt.show()
