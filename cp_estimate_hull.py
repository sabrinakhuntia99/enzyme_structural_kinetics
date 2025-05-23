import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Uploading data
data_path = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\cleaned_lp_data_hull.tsv"
df = pd.read_csv(data_path, sep='\t')

# Separating the features (x) from the output: hull_presence (y)
x = df.iloc[:, :-1]  # Features: all columns except the last (hull_presence)
y = df['hull_presence']  # Output: hull_presence column

# Data preparation for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Standardizing the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Initialize the Support Vector Regressor with default parameters
svr = SVR()

# Train the SVR model
svr.fit(x_train_scaled, y_train)

# Predict the test set results
y_pred_svr = svr.predict(x_test_scaled)

# Print SVR results
print("Support Vector Machine Regressor (SVR):")
print("R^2 Score:", r2_score(y_test, y_pred_svr))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_svr))

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


'''
# Support Vector Regressor with Hyperparameter Tuning
svr_param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],   # Different kernel functions to test
    'C': [0.1, 1, 10],                    # Regularization parameter
    'epsilon': [0.1, 0.01, 0.001],        # Epsilon in the epsilon-SVR model
    'gamma': ['scale', 'auto']            # Kernel coefficient for 'rbf', 'poly'
}

svr_grid_search = GridSearchCV(SVR(), svr_param_grid, cv=5, scoring='r2')
svr_grid_search.fit(x_train_scaled, y_train)
best_svr = svr_grid_search.best_estimator_
y_pred_svr = best_svr.predict(x_test_scaled)

# Print SVR results
print("Support Vector Machine Regressor (SVR):")
print("Best Params:", svr_grid_search.best_params_)
print("R^2 Score:", r2_score(y_test, y_pred_svr))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_svr))
'''