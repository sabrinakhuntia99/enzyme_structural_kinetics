import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the data file
df = pd.read_csv(
    r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\all_timepoints_dynamic_hull_id_030625_NCA_backbone_with_predictions.tsv',
    sep='\t'
)

# Drop rows where density is 0
df = df[df['density'] != 0].reset_index(drop=True)


def detect_breaking_point_from_hulls(row):
    """
    Detect the breaking point based on the hull presence values (transition from increasing to decreasing).
    """
    hull_columns = [
        'hull_presence_tryps_0010min', 'hull_presence_tryps_0015min', 'hull_presence_tryps_0020min',
        'hull_presence_tryps_0030min', 'hull_presence_tryps_0040min', 'hull_presence_tryps_0050min',
        'hull_presence_tryps_0060min', 'hull_presence_tryps_0120min', 'hull_presence_tryps_0180min',
        'hull_presence_tryps_0240min', 'hull_presence_tryps_1440min', 'hull_presence_tryps_leftover'
    ]

    hull_values = row[hull_columns].values  # Get the hull presence values

    # Find the point where hull values go from increasing to decreasing
    for i in range(2, len(hull_values)):
        if hull_values[i - 2] < hull_values[i - 1] and hull_values[i - 1] > hull_values[i]:
            return i  # Return the index where the change happens (this corresponds to the time)

    return None  # If no breaking point is found


# Function to predict the breaking point using protein features
def predict_breaking_time(features):
    """
    Predict the breaking time based on protein features using a simple linear regression model.
    The predicted time is rounded to the nearest positive integer.
    """
    # List of features to use for prediction
    feature_columns = [
        'density', 'radius_of_gyration', 'surface_area_to_volume_ratio', 'sphericity', 'euler_characteristic',
        'inradius', 'circumradius', 'hydrodynamic_radius', 'sequence_length', 'avg_plddt', 'predicted_disorder_content'
    ]

    # Prepare the feature array
    X = features[feature_columns].values.reshape(1, -1)

    # Predict the breaking time based on the features
    predicted_time = model.predict(X)[0]

    # Round the predicted time to the nearest positive integer
    return max(1, int(np.round(predicted_time)))  # Ensure it's a positive integer


# Prepare the dataset for linear regression (using features to predict breaking time)
# Since we don't have actual breaking points, let's use the detected ones for now
X_data = df[['density', 'radius_of_gyration', 'surface_area_to_volume_ratio', 'sphericity', 'euler_characteristic',
             'inradius', 'circumradius', 'hydrodynamic_radius', 'sequence_length', 'avg_plddt',
             'predicted_disorder_content']]
y_data = df.apply(detect_breaking_point_from_hulls, axis=1)

# Remove rows where breaking point could not be detected
valid_data = X_data[~y_data.isnull()]
y_valid = y_data[~y_data.isnull()]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(valid_data, y_valid, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Now predict breaking times for all proteins
detected_breaking_times = []
predicted_breaking_times = []

for i in range(len(df)):
    row = df.iloc[i]

    # Detect breaking point from hull data
    detected_breaking_time = detect_breaking_point_from_hulls(row)

    if detected_breaking_time is not None:
        detected_breaking_times.append(detected_breaking_time)

    # Predict the breaking point using the linear regression model
    predicted_time = predict_breaking_time(row)
    predicted_breaking_times.append(predicted_time)

# Plot histogram of detected and predicted breaking points
plt.figure(figsize=(10, 6))

# Plot histogram for detected breaking points
plt.hist(detected_breaking_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7,
         label='Detected Breaking Points')

# Plot histogram for predicted breaking points
plt.hist(predicted_breaking_times, bins=30, color='orange', edgecolor='black', alpha=0.7,
         label='Predicted Breaking Points')

# Add labels, title, and grid for better clarity
plt.xlabel("Breaking Time (Time Interval Index)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Detected vs Predicted Protein Breaking Points", fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Print frequencies of simulated breaking points
detected_breaking_point_counts = Counter(detected_breaking_times)
print("Number of proteins for each detected breaking point:")
for time, count in detected_breaking_point_counts.items():
    print(f"Detected Breaking Point: {time} - Count: {count}")

# Print frequencies of predicted breaking points
predicted_breaking_point_counts = Counter(predicted_breaking_times)
print("\nNumber of proteins for each predicted breaking point:")
for time, count in predicted_breaking_point_counts.items():
    print(f"Predicted Breaking Point: {time} - Count: {count}")
