import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')

# Step 2: Feature Normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Initialize
# Define a parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [1000], #adjust higher for better score
    'learning_rate': [0.01, 0.015, 0.02],
    'max_depth': [2, 3],
    'min_samples_leaf': [10, 13, 15],
    'subsample': [ 0.59, 0.6, 0.7],
    'max_features': [None]
}

# Initialize Gradient Boosting Regressor with early stopping
gb = GradientBoostingRegressor(random_state=42)


# Perform grid search
grid_search = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Best model
best_tree = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Calculate and print MSE of the best model
mse_score = mean_squared_error(y_train, best_tree.predict(x_train))
print("MSE on training data with best parameters: ", mse_score)




from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# Define an MSE scorer for cross-validation (negated for maximization in cross_val_score)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform cross-validation with the best model from grid search
cv_mse_scores = cross_val_score(
    best_tree,       # Best estimator from grid search
    x_train,         # Training data features
    y_train,         # Training data target
    scoring=mse_scorer,  # Scorer to calculate MSE
    cv=5,            # Number of cross-validation folds
    n_jobs=-1        # Use all available processors
)

# Calculate mean of the negative MSE scores (and negate to get positive MSE)
mean_cv_mse = -cv_mse_scores.mean()

print(f"Mean cross-validated MSE: {mean_cv_mse:.4f}")


y_test_hat = best_tree.predict(x_test)
y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(x_test))),
    'Predicted': y_test_hat.reshape(-1),
})


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('toshow.csv', index=False)

