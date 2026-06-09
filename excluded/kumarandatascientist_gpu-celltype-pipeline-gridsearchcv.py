import h5py
import numpy as np
import cudf
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt

from cuml.preprocessing import MinMaxScaler
from cuml.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error

class CellTypeGPUPipeline:
    """
    A GPU-based pipeline for cell type prediction using cuML's MinMaxScaler and 
    RandomForestRegressor (wrapped in MultiOutputRegressor). This class handles
    loading data from an H5 file and preparing the training set.
    """
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.train_spot_tables = {}
        self.cell_type_columns = None

    def load_train_data(self):
        """
        Loads training spot data from the H5 file and converts each slide to a pandas DataFrame.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            for slide_name in train_spots.keys():
                spot_array = np.array(train_spots[slide_name])
                df = pd.DataFrame(spot_array)
                self.train_spot_tables[slide_name] = df
        print("Training data loaded (GPU version) successfully.")

    def prepare_training_set(self, slide_id='S_1'):
        """
        Prepares training features and targets from a given slide.
        Assumes that the first two columns are 'x' and 'y' (coordinates) and the rest are continuous targets.
        """
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training data.")
        df = self.train_spot_tables[slide_id]
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        self.cell_type_columns = target_cols
        X = df[feature_cols].astype('float32')
        y = df[target_cols].astype('float32')
        return X, y

def load_test_data(h5_file_path, slide_id):
    """
    Loads test data for a given slide and returns a pandas DataFrame.
    """
    with h5py.File(h5_file_path, "r") as f:
        test_spots = f["spots/Test"]
        if slide_id not in test_spots:
            raise ValueError(f"Slide {slide_id} not found in test data.")
        spot_array = np.array(test_spots[slide_id])
        df = pd.DataFrame(spot_array)
    print(f"Test data for slide {slide_id} loaded successfully.")
    return df

def create_submission(test_df, predictions, submission_filename="submission.csv", cell_type_columns=None):
    """
    Creates a submission CSV file from the predictions.
    """
    # Create a DataFrame with predictions and add an ID column.
    pred_df = pd.DataFrame(predictions, columns=cell_type_columns, index=test_df.index)
    pred_df.insert(0, 'ID', test_df.index)
    pred_df.to_csv(submission_filename, index=False)
    print(f"Submission file '{submission_filename}' created!")




# --------------------------
# Main grid/random search code
# --------------------------
if __name__ == "__main__":
    # Specify the path to your dataset file.
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    
    # Initialize and load training data using our pipeline object.
    pipeline_obj = CellTypeGPUPipeline(h5_file_path)
    pipeline_obj.load_train_data()
    X_train, y_train = pipeline_obj.prepare_training_set(slide_id='S_1')
    
    # Convert training data to numpy arrays.
    if hasattr(X_train, 'to_pandas'):
        X_train_np = X_train.to_pandas().values
    else:
        X_train_np = X_train.values
        
    if hasattr(y_train, 'to_pandas'):
        y_train_np = y_train.to_pandas().values
    else:
        y_train_np = y_train.values
    
    # Create a scikit-learn pipeline using cuML's MinMaxScaler and a multi-output RandomForest regressor.
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
    ])
    
    # Define a parameter grid for hyperparameter tuning.
    param_grid = {
        'model__estimator__n_estimators': [50, 100, 200],
        'model__estimator__max_depth': [5, 10, 15],
        'model__estimator__min_samples_split': [2, 5, 10]
    }
    
    # Set up GridSearchCV using a 5-fold cross-validation.
    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train_np, y_train_np)
    best_params_grid = grid_search.best_params_
    best_score_grid = -grid_search.best_score_
    print("Best parameters from GridSearchCV:", best_params_grid)
    print("Best CV MSE from GridSearchCV:", best_score_grid)
    
    # Set up RandomizedSearchCV as an alternative (n_iter=5).
    random_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', 
                                         n_iter=5, random_state=42, verbose=2)
    random_search.fit(X_train_np, y_train_np)
    best_params_random = random_search.best_params_
    best_score_random = -random_search.best_score_
    print("Best parameters from RandomizedSearchCV:", best_params_random)
    print("Best CV MSE from RandomizedSearchCV:", best_score_random)
    
    # Use the best estimator from grid search for final predictions.
    best_model = grid_search.best_estimator_
    
    # Load test data.
    test_df = load_test_data(h5_file_path, slide_id='S_7')
    # Prepare test features: here we simply select 'x' and 'y' and convert to numpy.
    X_test = test_df[['x', 'y']].astype('float32').values
    
    # Predict on test data using the best model.
    predictions = best_model.predict(X_test)
    
    # Create the submission CSV file.
    create_submission(test_df, predictions, submission_filename="submission.csv", cell_type_columns=pipeline_obj.cell_type_columns)


