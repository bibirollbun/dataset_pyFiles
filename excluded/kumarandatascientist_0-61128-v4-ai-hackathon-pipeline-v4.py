import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor

class ElucidataPipeline:
    def __init__(self, h5_file_path, test_slide='S_7', submission_path='submission.csv', test_size=0.02, random_state=2024):
        self.h5_file_path = h5_file_path
        self.test_slide = test_slide
        self.submission_path = submission_path
        self.test_size = test_size
        self.random_state = random_state
        self.train_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.models = {}
        self.test_df = None
        self.predictions = None

    def load_train_data(self):
        print("Loading training data...")
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            train_spot_tables = {
                slide_name: pd.DataFrame(np.array(train_spots[slide_name]))
                for slide_name in train_spots.keys()
            }
        self.train_df = pd.concat(train_spot_tables.values(), ignore_index=True)
        print(f"Training data loaded. Shape: {self.train_df.shape}")

    def prepare_data(self):
        print("Preparing training data...")
        # Use spatial coordinates as features and the remaining columns as target labels.
        self.X = self.train_df[['x', 'y']]
        self.y = self.train_df.iloc[:, 2:]
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print("Data split into training and validation sets.")

    def define_models(self):
        print("Defining models...")
        self.models = {
            "RANSACRegressor": RANSACRegressor()
        }
        print("Models defined:", list(self.models.keys()))

    def train_models(self):
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
        print("Model training complete.")

    def validate_models(self):
        print("Validating models on the validation set...")
        preds_valid = {}
        for name, model in self.models.items():
            print(f"Predicting with {name} on validation data...")
            preds_valid[name] = model.predict(self.X_valid)
        return preds_valid

    def load_test_data(self):
        print("Loading test data...")
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            self.test_df = pd.DataFrame(np.array(test_spots[self.test_slide]))
        print(f"Test data loaded. Shape: {self.test_df.shape}")

    def predict_test(self):
        print("Predicting on test data...")
        X_test = self.test_df[['x', 'y']]
        test_preds = np.zeros((X_test.shape[0], self.y.shape[1]))
        for name, model in self.models.items():
            print(f"Predicting with {name}...")
            test_preds += model.predict(X_test)
        test_preds /= len(self.models)
        self.predictions = test_preds
        print("Test predictions complete.")

    def create_submission(self):
        print("Creating submission file...")
        submission_df = pd.DataFrame(self.predictions, columns=self.y.columns)
        submission_df.insert(0, 'ID', self.test_df.index)
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission file '{self.submission_path}' created!")

    def run_pipeline(self):
        self.load_train_data()
        self.prepare_data()
        self.define_models()
        self.train_models()
        _ = self.validate_models()  # Optionally use validation predictions
        self.load_test_data()
        self.predict_test()
        self.create_submission()




# Example usage:
if __name__ == "__main__":
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    pipeline = ElucidataPipeline(h5_file_path)
    pipeline.run_pipeline()


