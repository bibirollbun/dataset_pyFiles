import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Keep others for reference
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.2 


def rmsle(y_true, y_pred):
    # Add 1 to true and predicted values to handle zeros
    y_true_plus_1 = np.log1p(y_true) # log1p(x) calculates log(1+x)
    y_pred_plus_1 = np.log1p(y_pred)

    # Calculate the squared difference of the logarithms
    squared_log_error = (y_true_plus_1 - y_pred_plus_1) ** 2

    # Calculate the mean of the squared log errors and take the square root
    return np.sqrt(np.mean(squared_log_error))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print("Datasets loaded successfully.")


# Display basic information
print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape: {test_df.shape}")


X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories']

test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)
if 'Calories' in X_test.columns:
    X_test = X_test.drop('Calories', axis=1)


categorical_features = ['Sex']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' 
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTraining data split into {X_train.shape[0]} training and {X_val.shape[0]} validation samples.")


# Optuna Objective Function for Hyperparameter Tuning
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1 
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_pred[y_pred < 0] = 0
    current_rmsle = rmsle(y_val, y_pred)

    return current_rmsle # Optuna minimizes the objective function

# Hyperparameter Tuning with Optuna
print("\nStarting Optuna hyperparameter tuning (optimizing for RMSLE)...")
study = optuna.create_study(direction='minimize', study_name='rf_calorie_prediction_rmsle')


study.optimize(objective, n_trials=100)

print("\nOptuna tuning finished.")
print(f"Best trial (RMSLE): {study.best_trial.value:.4f} with parameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")


# Get the best hyperparameters
best_params = study.best_trial.params


print("\nTraining final Random Forest model with best hyperparameters on the full training data...")
final_rf_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=RANDOM_STATE,
    n_jobs=-1 
)


final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', final_rf_model)])

final_pipeline.fit(X, y)
print("Final model training completed.")


print("\nEvaluating the final model on the validation set...")
y_val_pred = final_pipeline.predict(X_val)

y_val_pred[y_val_pred < 0] = 0

mae_val = mean_absolute_error(y_val, y_val_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)
rmsle_val = rmsle(y_val, y_val_pred) # Calculate RMSLE


print(f"Validation Set Performance Metrics:")
print(f"MAE: {mae_val:.4f}")
print(f"MSE: {mse_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"R²: {r2_val:.4f}")
print(f"RMSLE: {rmsle_val:.4f}") # Print RMSLE


print("\nMaking predictions on the test data...")

# The pipeline handles applying the fitted preprocessor and then the final regressor
test_predictions = final_pipeline.predict(X_test)

# Ensure test predictions are non-negative for submission (Calories must be >= 0)
test_predictions[test_predictions < 0] = 0


predictions_df = pd.DataFrame({'id': test_ids, 'Calories': test_predictions})

predictions_df.to_csv('calorie_predictions_rf_optuna_rmsle.csv', index=False)
print("\nPredictions saved to 'calorie_predictions_rf_optuna_rmsle.csv'")

