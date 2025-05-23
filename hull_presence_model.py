import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

# Load and filter data
file_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\all_timepoints_dynamic_hull_id_032825_NCA_backbone_with_predictions.tsv"
df = pd.read_csv(file_path, sep='\t')
df = df[(df['hull_presence'] != 0) & (df['density'] != 0)]


def train_and_evaluate(X, y, dataset_name):
    # Split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline with feature selection
    pipeline = Pipeline([
        ('preprocessor', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler())
        ])),
        ('selector', SelectFromModel(Ridge(alpha=1.0))),  # Initial feature selection
        ('regressor', Ridge())
    ])

    # Parameter grid for tuning
    param_grid = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'selector__estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    }

    # Grid search with 5-fold CV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                               scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    metrics = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

    # Feature importance analysis
    poly_features = best_model.named_steps['preprocessor'].named_steps['poly'].get_feature_names_out(X.columns)
    selected = best_model.named_steps['selector'].get_support()
    important_features = poly_features[selected]

    coefficients = best_model.named_steps['regressor'].coef_

    print(f"\n{dataset_name} Model Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test R²: {metrics['R²']:.3f}")
    print(f"Test RMSE: {metrics['RMSE']:.3f}")
    print(f"Test MAE: {metrics['MAE']:.3f}")
    print("\nTop 10 Important Features:")
    for feat, coef in sorted(zip(important_features, coefficients),
                             key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"{feat}: {coef:.3f}")

    return metrics


# Prepare datasets
ordered_mask = df['predicted_disorder_content'] <= 0.3
X_ordered = df[ordered_mask].drop(columns=['uniprot_id', 'hull_presence'])
y_ordered = df[ordered_mask]['hull_presence']
X_disordered = df[~ordered_mask].drop(columns=['uniprot_id', 'hull_presence'])
y_disordered = df[~ordered_mask]['hull_presence']

# Prepare full dataset
X_full = df.drop(columns=['uniprot_id', 'hull_presence'])
y_full = df['hull_presence']

# Train and evaluate models
ordered_metrics = train_and_evaluate(X_ordered, y_ordered, 'Ordered')
disordered_metrics = train_and_evaluate(X_disordered, y_disordered, 'Disordered')
full_metrics = train_and_evaluate(X_full, y_full, 'Full')

# Compare model performances
metrics_comparison = pd.DataFrame({
    'Ordered': ordered_metrics,
    'Disordered': disordered_metrics,
    'Full': full_metrics

}).T

print("\nModel Performance Comparison:")
print(metrics_comparison.round(3))