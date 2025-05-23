import pandas as pd
import matplotlib.pyplot as plt
import re
import gseapy as gp

# 1) Load data
FILE_PATH = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\all_timepoints_dynamic_hull_id_032825_NCA_backbone_with_predictions.tsv"
df = pd.read_csv(FILE_PATH, sep='\t')

# 2) Define UniProt ID sets
AXOLOTL_IDS = [
    "Q0P665", "P49019", "Q9Y283", "O14654", "Q9NPH2", "Q9NZM3", "Q7Z628",
    "P29475", "A6NND4", "Q9H1J5", "P98196", "Q8VXB5", "P20472", "Q6NUK1",
    "Q9P013", "Q9NR30", "Q6P9F5", "Q99583", "Q15238", "Q9Y3B7", "P60866",
    "Q9H0M0", "Q8N8S7", "P08590", "P12829", "Q9UBC2", "Q13459", "Q8IVL1",
    "Q96JP0", "Q9BX66", "Q9Y336", "Q8NF91", "Q9H4A6", "Q9Y6X3", "Q5TAT6",
    "P02675", "P02679", "Q9UGM5", "Q70Z53", "Q9BTZ2", "P78348", "Q9NRS4", "Q9HCU0"
]

ALL_SALAMANDER_IDS = [
    "Q0P665", "P49019", "Q9Y283", "O14654", "Q9NPH2", "Q9NZM3", "Q7Z628",
    "P29475", "A6NND4", "Q9H1J5", "P98196", "Q8VXB5", "P20472", "Q6NUK1",
    "Q9P013", "Q9NR30", "Q6P9F5", "Q99583", "Q15238", "Q9Y3B7", "P60866",
    "Q9H0M0", "Q8N8S7", "P08590", "P12829", "Q9UBC2", "Q13459", "Q8IVL1",
    "Q96JP0", "Q9BX66", "Q9Y336", "Q8NF91", "Q9H4A6", "Q9Y6X3", "Q5TAT6",
    "P02675", "P02679", "Q9UGM5", "Q70Z53", "Q9BTZ2", "P78348", "Q9NRS4", "Q9HCU0",
    "P08590", "O14654", "Q9H4A6", "Q8N8S7", "Q9P013", "Q9Y3B7", "P60866",
    "Q9UBC2", "Q9NPH2", "Q13459", "Q8NF91", "Q9NR30"
]


HUMAN_IDS = [
    # Matrix Metalloproteinases (MMPs) & Proteases
    "P03956", "P22894", "P45452", "P09237", "P50281", "P51511", "Q9H239", "Q13443",

    # Growth Factors & Cytokines
    "P01127", "P01137", "P09038", "P05230", "P10600", "P01579", "P10145", "P18510",

    # Extracellular Matrix (ECM) Components
    "P02452", "P05997", "P12107", "P12109", "P02751", "P10451", "P09486",

    # Macrophage & Immune-Related Proteins
    "P04141", "P11532", "P08238", "P19876", "P16035", "P01033", "P05112", "P60568",

    # Coagulation & Fibrinolysis
    "P00734", "P00748", "P00747", "P00751",

    # Additional from original list
    "P14780", "P15692"  # MMPs/Growth factors
]

# 3) Data filtering function
def filter_and_analyze(dataframe, id_list, species_name):
    filtered_df = dataframe[dataframe['uniprot_id'].isin(id_list)].copy()
    filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]

    found_ids = set(filtered_df['uniprot_id'])
    missing_ids = set(id_list) - found_ids

    print(f"\n{species_name} analysis:")
    print(f"Found {len(found_ids)}/{len(id_list)} proteins in dataset")
    print("Found IDs:", found_ids)
    print("Missing IDs:", missing_ids)

    return filtered_df.set_index('uniprot_id')


# 4) Process both species
ax_df = filter_and_analyze(df, AXOLOTL_IDS, "Salamanders")
human_df = filter_and_analyze(df, HUMAN_IDS, "Human")

# 5) Timepoint configuration
TIME_COLS = [
    "hull_presence_tryps_0010min", "hull_presence_tryps_0015min",
    "hull_presence_tryps_0020min", "hull_presence_tryps_0030min",
    "hull_presence_tryps_0040min", "hull_presence_tryps_0050min",
    "hull_presence_tryps_0060min", "hull_presence_tryps_0120min",
    "hull_presence_tryps_0180min", "hull_presence_tryps_0240min",
    "hull_presence_tryps_1440min", "hull_presence_tryps_leftover"
]


# 6) Plotting function
def plot_kinetics(df, title, highlight=None):
    plt.figure(figsize=(12, 6))

    # Generate time labels
    time_labels = [re.search(r'_(\d+min|leftover)', col).group(1) for col in TIME_COLS]

    # Plot all proteins
    for prot_id in df.index:
        series = df.loc[prot_id, TIME_COLS]
        if highlight and prot_id in highlight:
            plt.plot(series.values, lw=2.5, label=f"{prot_id} ({highlight[prot_id]})")
        else:
            plt.plot(series.values, color='gray', alpha=0.4, lw=0.8)

    plt.xticks(range(len(TIME_COLS)), time_labels, rotation=45)
    plt.title(f"Protein Structural Kinetics: {title}")
    plt.ylabel("Hull Presence")
    plt.xlabel("Digestion Time")

    if highlight:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# 7) Generate plots with key highlights
AX_HIGHLIGHT = {
    "Q0P665": "COL13A1",
    "P02675": "Fibrinogen Beta",
    "P02679": "Fibrinogen Gamma"
}

HUMAN_HIGHLIGHT = {
    "P03956": "MMP-1",
    "P01137": "TGF-β1",
    "P01033": "TIMP1"
}

plot_kinetics(ax_df, "Salamanders Limb Regeneration Proteins", AX_HIGHLIGHT)
plot_kinetics(human_df, "Human Wound Debridement Proteins", HUMAN_HIGHLIGHT)


# 8) Pathway enrichment analysis
def run_enrichment(id_list, species):
    print(f"\nRunning pathway enrichment for {species}:")
    try:
        enr = gp.enrichr(
            gene_list=id_list,
            gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],
            organism='human' if species == 'human' else 'other'
        )
        return enr.results.head(10)
    except Exception as e:
        print(f"Enrichment failed: {str(e)}")
        return None


print(run_enrichment(ax_df.index.tolist(), "axolotl"))
print(run_enrichment(human_df.index.tolist(), "human"))

# 9) Comparative analysis
comparison_df = pd.DataFrame({
    'Salamanders': ax_df[TIME_COLS].mean(),
    'Human': human_df[TIME_COLS].mean()
})

plt.figure(figsize=(10, 5))
comparison_df.plot(marker='o')
plt.title("Comparative Structural Kinetics: Salamanders vs Human Proteins")
plt.ylabel("Mean Hull Presence")
plt.xticks(range(len(TIME_COLS)), time_labels, rotation=45)
plt.tight_layout()
plt.show()