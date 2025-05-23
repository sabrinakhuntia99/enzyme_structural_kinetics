import matplotlib.pyplot as plt
import numpy as np

# Data provided for AlphaFold and PDB models
features = [
    'sequence_length', 'density', 'radius_of_gyration', 'sphericity',
    'surface_area_to_volume_ratio', 'inradius', 'circumradius', 'hydrodynamic_radius'
]

# R² Scores for AlphaFold and PDB models
r2_scores_alphafold = [0.9550, 0.9515, 0.9686, 0.9617, 0.9679, 0.9672, 0.9678, 0.9635]
r2_scores_pdb = [0.9560, 0.9783, 0.9729, 0.9616, 0.9749, 0.9652, 0.9651, 0.9694]

# Bar width and positions
bar_width = 0.35
index = np.arange(len(features))

# Create the bar plot
plt.figure(figsize=(14, 8))

# Plot AlphaFold bars (purple)
plt.bar(index, r2_scores_alphafold, bar_width, label='AlphaFold', color='purple')

# Plot PDB bars (green) shifted to the right by bar_width
plt.bar(index + bar_width, r2_scores_pdb, bar_width, label='PDB', color='green')

# Formatting the plot
plt.xlabel('Feature', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.title('Impact of Feature Removal on R² Scores: AlphaFold vs. PDB (Gradient Boosting Model)', fontsize=16)
plt.xticks(index + bar_width / 2, features, rotation=45, ha='right', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
