import pandas as pd
import numpy as np
import json

def extract_aa_sequence(structure):
    amino_acid_sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue is an amino acid (exclude water molecules, ligands, etc.)
                if residue.get_id()[0] == " " and residue.get_resname() not in ["HOH", "H2O", "WAT"]:
                    amino_acid_sequence += residue.get_resname()

    return amino_acid_sequence
def extract_all_atoms(structure):
    points0 = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    point0 = atom.get_coord()
                    points0.append(point0)
    points0 = np.array(points0)
    return points0

def extract_C_backbone_atoms(structure):
    points = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "C":
                        point = atom.get_coord()
                        points.append(point)
    points = np.array(points)
    return points

def extract_CA_backbone_atoms(structure):
    points = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        point = atom.get_coord()
                        points.append(point)
    points = np.array(points)
    return points


def extract_N_backbone_atoms(structure):
    points = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "N":
                        point = atom.get_coord()
                        points.append(point)
    points = np.array(points)
    return points

def extract_N_and_CA_backbone_atoms(structure):
    points = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "N" or atom.get_name() == "CA":
                        point = atom.get_coord()
                        points.append(point)
    points = np.array(points)
    return points

def conv_array_text(string_from_csv):
    if not pd.isna(string_from_csv):
        # print(string_from_csv)
        converted_list = json.loads(string_from_csv.replace("array(", "").replace(")", "").replace(". ", ".0"))
        return converted_list
    else:
        return []

def conv_array_coord(data):
    coordinates_list = []
    for key, values in data.items():
        for value in values:
            if len(value) == 3:
                coordinates_list.append({'TimePoint': key, 'X': value[0], 'Y': value[1], 'Z': value[2]})
    return coordinates_list

