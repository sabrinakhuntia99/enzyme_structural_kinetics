import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create correlation matrix from your data
data = {
    'Feature': ['hull_presence', 'hull_presence', 'hull_presence', 'hull_presence', 'hull_presence',
                'hull_presence', 'hull_presence', 'hull_presence', 'hull_presence', 'hull_presence',
                'hull_presence', 'hull_presence', 'predicted_disorder_content', 'predicted_disorder_content',
                'predicted_disorder_content', 'predicted_disorder_content', 'predicted_disorder_content',
                'predicted_disorder_content', 'predicted_disorder_content', 'predicted_disorder_content',
                'predicted_disorder_content', 'predicted_disorder_content', 'predicted_disorder_content'],
    'Timepoint': ['0010min', '0015min', '0020min', '0030min', '0040min', '0050min', '0060min',
                  '0120min', '0180min', '0240min', '1440min', 'leftover', '0010min', '0015min',
                  '0020min', '0030min', '0040min', '0050min', '0060min', '0120min', '0180min',
                  '0240min', '1440min', 'leftover'],
    'Correlation': [0.6364, 0.2381, 0.6237, 0.0298, -0.3935, 0.0170, 0.2570, -0.7146, -0.7203,
                    -0.0883, 0.1183, -0.3075, 0.2939, -0.4735, -0.2331, -0.5186, 0.2702, -0.5508,
                    -0.5309, 0.1094, 0.0000, -0.3200, 0.4433, 0.3516]
}

df = pd.DataFrame(data)
corr_matrix = df.pivot(index='Feature', columns='Timepoint', values='Correlation')

# Sort columns by time duration
time_order = ['0010min', '0015min', '0020min', '0030min', '0040min', '0050min', '0060min',
              '0120min', '0180min', '0240min', '1440min', 'leftover']
corr_matrix = corr_matrix[time_order]

# Create heatmap
plt.figure(figsize=(12, 4))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='YlOrRd',  # Yellow-Orange-Red gradient
            fmt=".2f",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title("Correlation Matrix: Structural Features vs. Proteolysis Time Points", pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()