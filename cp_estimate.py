import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Uploading data
data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\lp_data_he_for_model.tsv"
df = pd.read_csv(data_path, sep='\t')

# Drop rows where hurst_exponent is NA
df = df.dropna(subset=['hurst_exponent'])
print("Number of rows after dropping NA:", len(df))

# Drop rows where feature values are 0.0
df = df[df['sequence_length'] != 0]
print("Number of rows after dropping proteins that have zero values for their properties:", len(df))

# Drop rows with very low Hurst exponent
df = df[df['hurst_exponent'] > 0.1]
print("Number of rows after dropping low HE:", len(df))

# Separating the features (x) from the output: hurst_exponent (y)
x = df.iloc[:, 1:]  # Features: all columns except the first
y = df.iloc[:, 0]   # Output: the first column (hurst_exponent)

# Data preparation for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # Standardizing the features
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

x_train_scaled = x_train
x_test_scaled = x_test


# Decision Tree Regressor with Hyperparameter Tuning
dt_param_grid = {
    'max_depth': [None, 5, 10, 14, 20],
    'min_samples_split': [2, 5, 10]
}
dt_grid_search = GridSearchCV(DecisionTreeRegressor(), dt_param_grid, cv=5, scoring='r2')
dt_grid_search.fit(x_train_scaled, y_train)
best_dt = dt_grid_search.best_estimator_
y_pred_dt = best_dt.predict(x_test_scaled)
print("Decision Tree:")
print("Best Params:", dt_grid_search.best_params_)
print("R^2 Score:", r2_score(y_test, y_pred_dt))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_dt))

# Random Forest Regressor with Hyperparameter Tuning
rf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5, scoring='r2')
rf_grid_search.fit(x_train_scaled, y_train)
best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(x_test_scaled)
print("Random Forest:")
print("Best Params:", rf_grid_search.best_params_)
print("R^2 Score:", r2_score(y_test, y_pred_rf))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))

# Gradient Boosting Regressor with Hyperparameter Tuning
gb_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
gb_grid_search = GridSearchCV(GradientBoostingRegressor(), gb_param_grid, cv=5, scoring='r2')
gb_grid_search.fit(x_train_scaled, y_train)
best_gb = gb_grid_search.best_estimator_
y_pred_gb = best_gb.predict(x_test_scaled)
print("Gradient Boosting:")
print("Best Params:", gb_grid_search.best_params_)
print("R^2 Score:", r2_score(y_test, y_pred_gb))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_gb))
