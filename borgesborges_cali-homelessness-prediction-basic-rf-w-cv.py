import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


pd.set_option('display.max_columns', None)


data_train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
data_test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')



data_train.head()


# Split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data_train.drop(columns=['HOMELESS_RATE', 'ID']),
    data_train['HOMELESS_RATE'],
    test_size=0.2,
    random_state=42
)


# MODELS
# let the model to be used uncommented

# Random forest model
model = RandomForestRegressor(n_estimators=50, random_state=42)

# XGBoost model
# model = xgb.RandomForestRegressor(n_estimators=3, random_state=42)


# Cross Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")

# Convert negative MSE to positive
cv_mse = -cv_scores
print(f"Cross-validation MSE scores: {cv_mse}")
print(f"Average CV MSE: {np.mean(cv_mse):.10f}")


# Fitting on training set and validating
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse:.10f}")



# Retraining on full training data
final_model = RandomForestRegressor(n_estimators=50, random_state=42)

final_model.fit(data_train.drop(columns=['HOMELESS_RATE', 'ID']), data_train['HOMELESS_RATE'])


# Filling submission
test_predictions = final_model.predict(data_test.drop(columns=['ID']))

submission = pd.DataFrame({
    'ID': data_test['ID'],
    'HOMELESS_RATE': test_predictions
})
submission.to_csv('submission.csv', index=False)

