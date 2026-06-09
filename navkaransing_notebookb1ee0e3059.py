import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

# Load the data
train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')

# Splitting features and target
X = train_data.drop(columns=['target'])
y = train_data['target']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# Preprocessing (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define base models
et_model = ExtraTreesRegressor(random_state=42)
gbr_model = GradientBoostingRegressor(random_state=42)
ridge_model = Ridge()

# Grid Search for hyperparameter tuning
param_grid_et = {
    'n_estimators': [391],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
}
grid_et = GridSearchCV(estimator=et_model, param_grid=param_grid_et, cv=3, scoring='r2', n_jobs=-1)
grid_et.fit(X_scaled, y)
best_et_model = grid_et.best_estimator_

param_grid_gbr = {
    'n_estimators': [391],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
}
grid_gbr = GridSearchCV(estimator=gbr_model, param_grid=param_grid_gbr, cv=3, scoring='r2', n_jobs=-1)
grid_gbr.fit(X_scaled, y)
best_gbr_model = grid_gbr.best_estimator_

# Stacking Regressor
stacking_model = StackingRegressor(
    estimators=[
        ('et', best_et_model),
        ('gbr', best_gbr_model),
        ('ridge', ridge_model),
    ],
    final_estimator=SVR(),
    n_jobs=-1
)

# Cross-validation for stacking model
stacking_r2_scores = []
stacking_mae_scores = []

for train_index, val_index in kf.split(X_scaled):
    X_train_kf, X_val_kf = X_scaled[train_index], X_scaled[val_index]
    y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
    
    stacking_model.fit(X_train_kf, y_train_kf)
    preds_val_stacking = stacking_model.predict(X_val_kf)
    
    # Calculate R2 and MAE
    r2_stacking = r2_score(y_val_kf, preds_val_stacking)
    mae_stacking = mean_absolute_error(y_val_kf, preds_val_stacking)
    
    stacking_r2_scores.append(r2_stacking)
    stacking_mae_scores.append(mae_stacking)

print(f"Stacking Model R2 scores from K-Fold: {stacking_r2_scores}")
print(f"Mean R2 score for Stacking Model: {np.mean(stacking_r2_scores)}")
print(f"Stacking Model MAE scores from K-Fold: {stacking_mae_scores}")
print(f"Mean MAE score for Stacking Model: {np.mean(stacking_mae_scores)}")

# Final predictions on the test set using the Stacking Regressor
stacking_model.fit(X_scaled, y)  # Fit on the entire training data
stacking_preds_test = stacking_model.predict(X_test_scaled)

# Create submission file for the Stacking Regressor
submission_stacking = pd.DataFrame({
    'id': test_ids,
    'target': stacking_preds_test
})
submission_stacking.to_csv('submission_stacking.csv', index=False)
print("Submission file created: submission_stacking.csv")

