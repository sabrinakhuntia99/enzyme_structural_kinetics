from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Added import
import matplotlib
matplotlib.use('Agg')

# === Load data ===
csv_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\structural_properties_with_distances.csv"
results_df = pd.read_csv(csv_path, index_col='uniprot_id')
print(len(results_df))
# === Filter usable entries ===
df = results_df[(results_df['breaking_time'].notnull()) & (results_df['breaking_time'] > 0)].copy()
feature_cols = [c for c in df.columns if c not in ('breaking_time', 'euler_characteristic')]
df = df.dropna(subset=feature_cols + ['breaking_time'])
print(len(results_df))

# === Split by circumradius into bottom 30% and top 70% ===
threshold_circum = df['circumradius'].quantile(0.3)
df_fracture = df[df['circumradius'] <= threshold_circum].copy()
print(len(df_fracture))
df_creep = df[df['circumradius'] > threshold_circum].copy()
print(len(df_creep))


# === Visualize breaking time distributions ===
def plot_breaking_time_distributions():
    plt.figure(figsize=(15, 6))

    # Fracture group distribution
    plt.subplot(1, 2, 1)
    plt.hist(df_fracture['breaking_time'], bins=50, color='blue', alpha=0.7,
             edgecolor='black', label='Fracture (Bottom 30%)')
    plt.title('Breaking Time Distribution - Fracture Group')
    plt.xlabel('Breaking Time (minutes)')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, linestyle='--', alpha=0.5)

    # Creep group distribution
    plt.subplot(1, 2, 2)
    plt.hist(df_creep['breaking_time'], bins=50, color='orange', alpha=0.7,
             edgecolor='black', label='Creep (Top 70%)')
    plt.title('Breaking Time Distribution - Creep Group')
    plt.xlabel('Breaking Time (minutes)')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Breaking Time Distribution by Circumradius Group', fontsize=14)
    plt.tight_layout()
    plt.show()


# === Preprocessing function ===
def preprocess(df_group):
    X = df_group[feature_cols].values.astype(np.float32)
    y = df_group['breaking_time'].values.astype(np.float32).reshape(-1, 1)

    vt = VarianceThreshold(threshold=0.0)
    X = vt.fit_transform(X)
    keep_idx = vt.get_support(indices=True)
    feature_names = [feature_cols[i] for i in keep_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, feature_names


# === Define fracture-inspired network ===
class FractureNet(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 128), nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


# === Define creep-inspired network ===
class CreepNet(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 128), nn.Sigmoid(),
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# === Training function ===
def train_model(model, X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss in {name} model at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"[{name}] Epoch {epoch:>3}/300  Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        test_loss = criterion(preds, y_test)
        print(f"\n{name} Test MSE: {test_loss.item():.4f}")
        print(f"Sample True vs. Pred ({name}):")
        for t, p in zip(y_test[:5], preds[:5]):
            print(f"  {t.item():>5.1f} min  →  {p.item():>5.1f} min")


# === Main execution ===
if __name__ == "__main__":
    # Plot distributions before training
    plot_breaking_time_distributions()

    # Run fracture model
    print("\n=== FRACTURE MODEL (Bottom 30% Circumradius) ===")
    Xf, yf, feat_f = preprocess(df_fracture)
    fracture_model = FractureNet(Xf.shape[1])
    train_model(fracture_model, Xf, yf, "Fracture")

    # Run creep model
    print("\n=== CREEP MODEL (Top 70% Circumradius) ===")
    Xc, yc, feat_c = preprocess(df_creep)
    creep_model = CreepNet(Xc.shape[1])
    train_model(creep_model, Xc, yc, "Creep")