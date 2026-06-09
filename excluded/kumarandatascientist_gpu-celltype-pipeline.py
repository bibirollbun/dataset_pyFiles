import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import VotingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class CellTypeEnsemblePipeline:
    """
    A pipeline for cell type prediction that loads data from an H5 file,
    prepares features and targets, trains a multi-output ensemble regressor 
    (combining SVR and ExtraTreesRegressor) with cross-validation (including OOF predictions),
    and performs finetuning via a simple grid search.
    
    Reproducibility is ensured using a fixed random_state.
    """
    
    def __init__(self, h5_file_path, random_state=42):
        self.h5_file_path = h5_file_path
        self.train_spot_tables = {}
        self.cell_type_columns = None
        self.model = None
        self.random_state = random_state

    def load_train_data(self):
        """
        Loads training data from the H5 file and converts each slide to a pandas DataFrame.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            for slide_name in train_spots.keys():
                spot_array = np.array(train_spots[slide_name])
                df = pd.DataFrame(spot_array)
                self.train_spot_tables[slide_name] = df
        print("Training data loaded successfully.")

    def prepare_training_set(self, slide_id='S_1'):
        """
        Prepares training features and targets from a given slide.
        Assumes that the first two columns are 'x' and 'y', and the remaining columns 
        are continuous cell type abundances.
        """
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training data.")
        df = self.train_spot_tables[slide_id]
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        self.cell_type_columns = target_cols
        X = df[feature_cols].values.astype(float)
        y = df[target_cols].values.astype(float)
        return X, y

    def load_test_data(self, slide_id):
        """
        Loads test data for a given slide and returns a pandas DataFrame.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            if slide_id not in test_spots:
                raise ValueError(f"Slide {slide_id} not found in test data.")
            spot_array = np.array(test_spots[slide_id])
            df = pd.DataFrame(spot_array)
        print(f"Test data for slide {slide_id} loaded successfully.")
        return df

    def build_model(self, et_params=None):
        """
        Builds a multi-output ensemble regressor using a VotingRegressor that combines
        an SVR and an ExtraTreesRegressor.
        
        Parameters:
            et_params (dict): ExtraTreesRegressor parameters. If not provided, defaults are used.
        """
        et_params = et_params if et_params is not None else {}
        # Ensure reproducibility in ExtraTreesRegressor
        et_params.setdefault('random_state', self.random_state)
        base_estimator = VotingRegressor(
            estimators=[
                ('svr', SVR()),
                ('etr', ExtraTreesRegressor(**et_params))
            ]
        )
        self.model = MultiOutputRegressor(base_estimator)
        print("Model built successfully.")
        return self.model

    def train(self, X, y, et_params=None):
        """
        Trains the multi-output ensemble regressor on the full training set.
        
        Parameters:
            X (np.array): Training features.
            y (np.array): Training targets.
            et_params (dict): ExtraTreesRegressor parameters.
        """
        if self.model is None:
            self.build_model(et_params=et_params)
        self.model.fit(X, y)
        print("Model training complete on full training set.")
        return self.model

    def predict(self, X):
        """
        Makes predictions on the provided feature array.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        predictions = self.model.predict(X)
        return predictions

    def create_submission(self, test_df, predictions, submission_filename="submission.csv"):
        """
        Creates a submission CSV file with spot IDs and predicted cell type abundances.
        """
        pred_df = pd.DataFrame(predictions, columns=self.cell_type_columns, index=test_df.index)
        pred_df.insert(0, 'ID', test_df.index)
        pred_df.to_csv(submission_filename, index=False)
        print(f"Submission file '{submission_filename}' created!")

    def cross_validate(self, X, y, n_splits=5, callback=None, et_params=None):
        """
        Performs K-Fold cross-validation with out-of-fold (OOF) predictions.
        
        Parameters:
            X (np.array): Feature array.
            y (np.array): Target array.
            n_splits (int): Number of folds.
            callback (function): Optional callback called after each fold with arguments:
                fold_index, fold_mse, and validation indices.
            et_params (dict): Parameters for ExtraTreesRegressor.
        
        Returns:
            mse_scores (list): MSE scores for each fold.
            oof_preds (np.array): Out-of-fold predictions.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        n_samples = X.shape[0]
        n_targets = y.shape[1]
        oof_preds = np.zeros((n_samples, n_targets))
        mse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Build and train a new model for this fold.
            model = self.build_model(et_params=et_params)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            fold_mse = mean_squared_error(y_val_fold, y_pred)
            mse_scores.append(fold_mse)
            oof_preds[val_idx, :] = y_pred
            
            print(f"Fold {fold+1} MSE: {fold_mse:.4f}")
            if callback is not None:
                callback(fold_index=fold+1, fold_mse=fold_mse, val_indices=val_idx)
                
        return mse_scores, oof_preds

    def finetune(self, X, y, param_grid, n_splits=5):
        """
        Performs a simple grid search to finetune ExtraTreesRegressor hyperparameters.
        
        Parameters:
            X (np.array): Feature array.
            y (np.array): Target array.
            param_grid (dict): Dictionary where keys are parameter names and values are lists of possible values.
            n_splits (int): Number of folds for CV.
        
        Returns:
            best_params (dict): Best parameter combination found.
            best_score (float): Corresponding average CV MSE.
        """
        best_params = None
        best_score = float('inf')
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_comb in itertools.product(*param_values):
            params = dict(zip(param_names, param_comb))
            print(f"Testing params: {params}")
            mse_scores, _ = self.cross_validate(X, y, n_splits=n_splits, et_params=params)
            avg_score = np.mean(mse_scores)
            print(f"Avg CV MSE for params {params}: {avg_score:.4f}")
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                
        print(f"Best params: {best_params}, with Avg CV MSE: {best_score:.4f}")
        return best_params, best_score

# Example callback function.
def fold_callback(fold_index, fold_mse, val_indices):
    print(f"Callback: Fold {fold_index} completed. MSE: {fold_mse:.4f}, Validation samples: {len(val_indices)}")




# Example usage.
if __name__ == "__main__":
    # Path to the dataset file.
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    
    # Initialize pipeline with reproducibility.
    pipeline = CellTypeEnsemblePipeline(h5_file_path, random_state=42)
    
    # Load training data and prepare the training set from slide 'S_1'.
    pipeline.load_train_data()
    X_train, y_train = pipeline.prepare_training_set(slide_id='S_1')
    
    # Perform K-Fold cross-validation with OOF predictions and a callback.
    mse_scores, oof_predictions = pipeline.cross_validate(
        X_train, y_train, n_splits=5, callback=fold_callback
    )
    print("Average CV MSE:", np.mean(mse_scores))
    
    # Finetune hyperparameters for the ExtraTreesRegressor.
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    best_params, best_score = pipeline.finetune(X_train, y_train, param_grid, n_splits=3)
    
    # Train final model on the full training set using the best hyperparameters.
    pipeline.train(X_train, y_train, et_params=best_params)
    
    # Load test data for slide 'S_7' and make predictions.
    test_df = pipeline.load_test_data(slide_id='S_7')
    X_test = test_df[['x', 'y']].values.astype(float)
    predictions = pipeline.predict(X_test)
    
    # Create the submission file.
    pipeline.create_submission(test_df, predictions, submission_filename="submission.csv")


