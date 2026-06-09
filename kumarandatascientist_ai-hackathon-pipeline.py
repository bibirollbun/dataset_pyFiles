import h5py
import cudf
import cupy as cp
import numpy as np
from cuml.model_selection import train_test_split
from cuml.linear_model import LinearRegression

class ElucidataGPUPipeline:
    def __init__(self, h5_file_path, test_slide='S_7', submission_path='submission.csv', test_size=0.025, random_state=23):
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
        print("Loading training data on GPU...")
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            # Convert each slide's data to a cuDF DataFrame
            train_spot_tables = {
                slide_name: cudf.DataFrame(np.array(train_spots[slide_name]))
                for slide_name in train_spots.keys()
            }
        # Concatenate all training data into one cuDF DataFrame
        self.train_df = cudf.concat(list(train_spot_tables.values()), ignore_index=True)
        print(f"Training data loaded. Shape: {self.train_df.shape}")

    def prepare_data(self):
        print("Preparing training data on GPU...")
        # Use spatial coordinates as features and the remaining columns as target labels.
        self.X = self.train_df[['x', 'y']]
        self.y = self.train_df.drop(columns=['x', 'y'])
        # Use cuML's train_test_split to create GPU DataFrames for training/validation
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print("Data split into training and validation sets.")

    def define_models(self):
        print("Defining GPU models...")
        # Since RAPIDS does not provide a GPU RANSACRegressor, we use cuML's LinearRegression.
        self.models = {
            "LinearRegression": LinearRegression()
        }
        print("Models defined:", list(self.models.keys()))

    def train_models(self):
        print("Training models on GPU...")
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
        print("Loading test data on GPU...")
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            # Convert the selected test slide into a cuDF DataFrame
            self.test_df = cudf.DataFrame(np.array(test_spots[self.test_slide]))
        print(f"Test data loaded. Shape: {self.test_df.shape}")

    def predict_test(self):
        print("Predicting on test data on GPU...")
        X_test = self.test_df[['x', 'y']]
        # Determine the number of target columns from training labels
        num_targets = self.y.shape[1]
        # Create a Cupy array to accumulate predictions
        test_preds = cp.zeros((X_test.shape[0], num_targets))
        for name, model in self.models.items():
            print(f"Predicting with {name}...")
            preds = model.predict(X_test)
            # Ensure predictions are Cupy arrays
            if isinstance(preds, cudf.DataFrame):
                preds = preds.to_cupy()
            elif isinstance(preds, np.ndarray):
                preds = cp.asarray(preds)
            test_preds += preds
        test_preds /= len(self.models)
        # Convert predictions back to a cuDF DataFrame
        self.predictions = cudf.DataFrame(cp.asnumpy(test_preds), columns=self.y.columns)
        print("Test predictions complete.")

    def create_submission(self):
        print("Creating submission file...")
        submission_df = self.predictions.copy()
        # Convert the test DataFrame index to a CPU pandas Index for CSV output
        submission_df.insert(0, 'ID', self.test_df.index.to_pandas())
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
    pipeline = ElucidataGPUPipeline(h5_file_path)
    pipeline.run_pipeline()


