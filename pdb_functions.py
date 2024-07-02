from Bio.PDB import PDBParser

def extract_sequence_from_pdb(pdb_file_path):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file_path)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    sequence += residue.get_resname()

    return sequence

def search_pdb_file(uniprot_id):
    uniprot_id = uniprot_id.split('-')[0]
    return rf"C:\Users\Sabrina\PycharmProjects\intrinsic_disorder\proteome_human\AF-{uniprot_id}-F1-model_v4.pdb\AF-{uniprot_id}-F1-model_v4.pdb"
