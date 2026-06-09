import h5py
import numpy as np
import cudf
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from cuml.preprocessing import MinMaxScaler
from cuml.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression

# --------------------------------------
# EDA & Data Preparation Pipeline Class
# --------------------------------------
class CellTypeEDAPipeline:
    """
    A unified pipeline for cell type prediction that performs EDA, hyperparameter optimization with Optuna,
    model calibration, and submission generation.
    
    Steps:
      1. Load training data from an H5 file.
      2. Prepare training features and targets (assumes first two columns 'x' and 'y' are coordinates,
         and remaining columns are continuous targets).
      3. Perform basic EDA (descriptive statistics and histograms).
    """
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.train_spot_tables = {}
        self.cell_type_columns = None

    def load_train_data(self):
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            for slide_name in train_spots.keys():
                spot_array = np.array(train_spots[slide_name])
                df = pd.DataFrame(spot_array)
                self.train_spot_tables[slide_name] = df
        print("Training data loaded successfully.")

    def prepare_training_set(self, slide_id='S_1'):
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training data.")
        df = self.train_spot_tables[slide_id]
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        self.cell_type_columns = target_cols
        X = df[feature_cols].astype('float32')
        y = df[target_cols].astype('float32')
        return X, y

    def perform_eda(self, slide_id='S_1'):
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training data.")
        df = self.train_spot_tables[slide_id]
        print("Descriptive statistics for slide", slide_id)
        print(df.describe())
        df[['x', 'y']].hist(bins=30, figsize=(12,5))
        plt.suptitle("Histograms of 'x' and 'y' features")
        plt.show()

# --------------------------
# Helper functions for test data and submission
# --------------------------
def load_test_data(h5_file_path, slide_id):
    with h5py.File(h5_file_path, "r") as f:
        test_spots = f["spots/Test"]
        if slide_id not in test_spots:
            raise ValueError(f"Slide {slide_id} not found in test data.")
        spot_array = np.array(test_spots[slide_id])
        df = pd.DataFrame(spot_array)
    print(f"Test data for slide {slide_id} loaded successfully.")
    return df

def create_submission(test_df, predictions, submission_filename, cell_type_columns):
    pred_df = pd.DataFrame(predictions, columns=cell_type_columns, index=test_df.index)
    pred_df.insert(0, 'ID', test_df.index)
    pred_df.to_csv(submission_filename, index=False)
    print(f"Submission file '{submission_filename}' created!")

def calibrate_model(best_model, X_train_np, y_train_np):
    """
    Calibrates the regression predictions using isotonic regression for each target.
    Returns a list of IsotonicRegression calibrators.
    """
    y_train_pred = best_model.predict(X_train_np)
    n_targets = y_train_np.shape[1]
    calibrators = []
    for i in range(n_targets):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_train_pred[:, i], y_train_np[:, i])
        calibrators.append(iso)
    return calibrators

def apply_calibration(predictions, calibrators):
    n_targets = predictions.shape[1]
    calibrated_preds = np.zeros_like(predictions)
    for i in range(n_targets):
        calibrated_preds[:, i] = calibrators[i].predict(predictions[:, i])
    return calibrated_preds

# --------------------------
# Optuna Hyperparameter Optimization
# --------------------------
def objective(trial):
    # Suggest hyperparameters.
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200])
    # Remove None option; use only integer values.
    max_depth = trial.suggest_categorical('max_depth', [10, 20])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Create pipeline with suggested hyperparameters.
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MultiOutputRegressor(
            RandomForestRegressor(random_state=42,
                                  n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split)
        ))
    ])
    
    # Evaluate using 5-fold CV.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in cv.split(X_train_np):
        X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
        y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    return np.mean(mse_scores)




# --------------------------
# Main Code
# --------------------------
if __name__ == "__main__":
    # Specify the path to the dataset.
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    
    # Initialize EDA pipeline and load data.
    eda_pipeline = CellTypeEDAPipeline(h5_file_path)
    eda_pipeline.load_train_data()
    eda_pipeline.perform_eda(slide_id='S_1')
    X_train, y_train = eda_pipeline.prepare_training_set(slide_id='S_1')
    
    # Convert training data to numpy arrays.
    if hasattr(X_train, 'to_pandas'):
        X_train_np = X_train.to_pandas().values
    else:
        X_train_np = X_train.values
    if hasattr(y_train, 'to_pandas'):
        y_train_np = y_train.to_pandas().values
    else:
        y_train_np = y_train.values
    
    # Optimize hyperparameters with Optuna.
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    best_trial = study.best_trial
    print("  MSE:", best_trial.value)
    print("  Params:", best_trial.params)
    
    # Build final pipeline with best hyperparameters.
    best_pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MultiOutputRegressor(
            RandomForestRegressor(
                random_state=42,
                n_estimators=best_trial.params['n_estimators'],
                max_depth=best_trial.params['max_depth'],
                min_samples_split=best_trial.params['min_samples_split']
            )
        ))
    ])
    best_pipe.fit(X_train_np, y_train_np)
    
    # Calibrate model predictions on training set.
    calibrators = calibrate_model(best_pipe, X_train_np, y_train_np)
    
    # Load test data.
    test_df = load_test_data(h5_file_path, slide_id='S_7')
    # Prepare test features: select 'x' and 'y' and convert to numpy.
    X_test = test_df[['x', 'y']].astype('float32')
    X_test_np = X_test.values
    
    # Predict on test data using the best model.
    test_predictions = best_pipe.predict(X_test_np)
    
    # Apply calibration to test predictions.
    calibrated_predictions = apply_calibration(test_predictions, calibrators)
    
    # Create the submission CSV file with calibrated predictions.
    create_submission(test_df, calibrated_predictions, submission_filename="submission.csv",
                      cell_type_columns=eda_pipeline.cell_type_columns)


