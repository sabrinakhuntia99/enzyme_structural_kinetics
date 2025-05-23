import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    r'C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\disprot_lp_shared_uniprot_ids.tsv',
    sep='\t'
)

# Drop excluded columns
exclude_cols = ['uniprot_id'] + \
    [f'lys_arg_layer_{i}' for i in range(1, 11)] + \
    [f'hull_presence_tryps_{t}' for t in [
        '0010min', '0015min', '0020min', '0030min', '0040min', '0050min',
        '0060min', '0120min', '0180min', '0240min', '1440min', 'leftover'
    ]]

df_model = df.drop(columns=exclude_cols).dropna()

X = df_model.drop(columns='predicted_disorder_content')
y = df_model['predicted_disorder_content']

# Apply PowerTransformer (Box-Cox) to y
# Add small offset to avoid zero or negative values
pt = PowerTransformer(method='box-cox')
y_bc = pt.fit_transform((y + 1e-6).values.reshape(-1, 1)).flatten()

# Train/test split (still keep for plotting after cross-validation)
X_train, X_test, y_train_bc, y_test_bc = train_test_split(X, y_bc, test_size=0.2, random_state=42)

results = {}

# Helper function for CV scoring
def cross_val_metrics(model, X, y, cv=5):
    preds = cross_val_predict(model, X, y, cv=cv)
    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    return r2, rmse, preds

# 1. Random Forest (tuned)
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2')
rf_grid.fit(X_train, y_train_bc)
best_rf = rf_grid.best_estimator_
r2, rmse, preds = cross_val_metrics(best_rf, X, y_bc)
results['Random Forest'] = {'R²': r2, 'RMSE': rmse}

# 2. Gradient Boosting (tuned)
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='r2')
gb_grid.fit(X_train, y_train_bc)
best_gb = gb_grid.best_estimator_
r2, rmse, preds = cross_val_metrics(best_gb, X, y_bc)
results['Gradient Boosting'] = {'R²': r2, 'RMSE': rmse}

# 3. Lasso Regression
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=10000))
])
r2, rmse, preds = cross_val_metrics(lasso_pipeline, X, y_bc)
results['Lasso'] = {'R²': r2, 'RMSE': rmse}

# 4. ElasticNet Regression
elastic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000))
])
r2, rmse, preds = cross_val_metrics(elastic_pipeline, X, y_bc)
results['ElasticNet'] = {'R²': r2, 'RMSE': rmse}

# 5. Polynomial Regression (Ridge)
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])
r2, rmse, preds = cross_val_metrics(poly_pipeline, X, y_bc)
results['Polynomial (Ridge)'] = {'R²': r2, 'RMSE': rmse}

# 6. Stacking Regressor (Lasso, RF, GB)
stacking = StackingRegressor(
    estimators=[
        ('lasso', lasso_pipeline),
        ('rf', best_rf),
        ('gb', best_gb)
    ],
    final_estimator=Ridge()
)
r2, rmse, preds = cross_val_metrics(stacking, X, y_bc)
results['Stacking (Lasso + RF + GB)'] = {'R²': r2, 'RMSE': rmse}

# Show results
print("\n📊 Model Performance Comparison (with Box-Cox transformed target and CV):")
for name, scores in results.items():
    print(f"{name}: R² = {scores['R²']:.3f}, RMSE = {scores['RMSE']:.3f}")

# Fit best model on train set and plot actual vs predicted on test set
best_model_name = max(results, key=lambda k: results[k]['R²'])
print(f"\nBest model: {best_model_name}")

# Fit best model to training data
if best_model_name == 'Random Forest':
    best_model = best_rf
elif best_model_name == 'Gradient Boosting':
    best_model = best_gb
elif best_model_name == 'Lasso':
    best_model = lasso_pipeline
elif best_model_name == 'ElasticNet':
    best_model = elastic_pipeline
elif best_model_name == 'Polynomial (Ridge)':
    best_model = poly_pipeline
else:  # stacking
    best_model = stacking

best_model.fit(X_train, y_train_bc)
y_pred_bc = best_model.predict(X_test)

# Inverse transform the Box-Cox predictions and true values
y_test_orig = pt.inverse_transform(y_test_bc.reshape(-1,1)).flatten()
y_pred_orig = pt.inverse_transform(y_pred_bc.reshape(-1,1)).flatten()

# Plot Actual vs Predicted
plt.figure(figsize=(7,7))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel('Actual predicted_disorder_content')
plt.ylabel('Predicted predicted_disorder_content')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.grid(True)
plt.show()
