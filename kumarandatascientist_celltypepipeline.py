import h5py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class CellTypePipeline:
    """
    A pipeline for loading cell type data from an H5 file, training a model to predict 
    cell type abundances based on spot coordinates, and generating a submission file.
    """
    
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.train_spot_tables = {}
        self.cell_type_columns = None
        
    def load_train_data(self):
        """
        Load training spot data from the H5 file and store each slide as a DataFrame.
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
        Prepare training features and targets from a given slide.
        
        Parameters:
            slide_id (str): Identifier for the training slide to use.
        
        Returns:
            X (ndarray): Features array (using 'x' and 'y' columns).
            y (ndarray): Target cell type abundances.
        """
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training data.")
        df = self.train_spot_tables[slide_id]
        # Assume first two columns are coordinates and the rest are cell type abundances.
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        self.cell_type_columns = target_cols
        X = df[feature_cols].values.astype(float)
        y = df[target_cols].values.astype(float)
        return X, y
    
    def load_test_data(self, slide_id):
        """
        Load test spot data for a given slide.
        
        Parameters:
            slide_id (str): Identifier for the test slide.
        
        Returns:
            test_df (DataFrame): DataFrame containing test spot coordinates.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            if slide_id not in test_spots:
                raise ValueError(f"Slide {slide_id} not found in test data.")
            spot_array = np.array(test_spots[slide_id])
            test_df = pd.DataFrame(spot_array)
        print(f"Test data for slide {slide_id} loaded successfully.")
        return test_df
    
    def build_model_pipeline(self):
        """
        Build and return a scikit-learn pipeline that scales data and fits a multi-output regressor.
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
        ])
        return pipeline
    
    def train(self, X, y):
        """
        Train the model pipeline on provided features and targets.
        
        Parameters:
            X (ndarray): Training features.
            y (ndarray): Training target values.
        
        Returns:
            model (Pipeline): Trained scikit-learn pipeline.
        """
        model = self.build_model_pipeline()
        model.fit(X, y)
        print("Model training complete.")
        return model
    
    def predict(self, model, X_test):
        """
        Make predictions on test data using the trained model.
        
        Parameters:
            model (Pipeline): Trained model pipeline.
            X_test (ndarray): Test features.
        
        Returns:
            predictions (ndarray): Predicted cell type abundances.
        """
        predictions = model.predict(X_test)
        return predictions
    
    def create_submission(self, test_df, predictions, submission_filename="submission.csv"):
        """
        Create a submission CSV file with predicted cell type abundances.
        
        Parameters:
            test_df (DataFrame): Original test DataFrame (to get the index as spot IDs).
            predictions (ndarray): Predicted cell type abundances.
            submission_filename (str): Name of the CSV file to be created.
        """
        pred_df = pd.DataFrame(predictions, columns=self.cell_type_columns, index=test_df.index)
        pred_df.insert(0, 'ID', pred_df.index)
        pred_df.to_csv(submission_filename, index=False)
        print(f"Submission file '{submission_filename}' created!")




# Example usage:
if __name__ == "__main__":
    # File path to the dataset
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    
    # Initialize the pipeline
    pipeline = CellTypePipeline(h5_file_path)
    
    # Load training data and prepare training set from a specific slide (e.g., 'S_1')
    pipeline.load_train_data()
    X_train, y_train = pipeline.prepare_training_set(slide_id='S_1')
    
    # Train the model
    model = pipeline.train(X_train, y_train)
    
    # Load test data for a specific slide (e.g., 'S_7')
    test_df = pipeline.load_test_data(slide_id='S_7')
    X_test = test_df[['x', 'y']].values.astype(float)
    
    # Predict cell type abundances on test data
    predictions = pipeline.predict(model, X_test)
    
    # Create and save the submission file
    pipeline.create_submission(test_df, predictions, submission_filename="submission.csv")





