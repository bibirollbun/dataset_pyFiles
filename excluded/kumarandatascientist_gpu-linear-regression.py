import h5py
import cudf
import cupy as cp
import numpy as np
import logging
from cuml.model_selection import train_test_split, KFold  # Ensure KFold is available in your cuML version
from cuml.linear_model import LinearRegression
# Optionally import evaluation metrics if needed
# from cuml.metrics import mean_squared_error

class ElucidataGPUPipeline:
    """
    GPU accelerated pipeline for training, evaluation (via cross validation), 
    and test predictions using cuML.
    """
    def __init__(self, h5_file_path: str, test_slide: str = 'S_7', submission_path: str = 'submission.csv',
                 test_size: float = 0.02, random_state: int = 42, use_cv: bool = False, n_splits: int = 5):
        self.h5_file_path = h5_file_path
        self.test_slide = test_slide
        self.submission_path = submission_path
        self.test_size = test_size
        self.random_state = random_state
        self.use_cv = use_cv
        self.n_splits = n_splits
        
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

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_train_data(self) -> None:
        """
        Load training data from an HDF5 file and convert each slide's data into a cuDF DataFrame.
        """
        self.logger.info("Loading training data on GPU...")
        try:
            with h5py.File(self.h5_file_path, "r") as f:
                train_spots = f["spots/Train"]
                # Convert each slide's data to a cuDF DataFrame
                train_spot_tables = {
                    slide_name: cudf.DataFrame(np.array(train_spots[slide_name]))
                    for slide_name in train_spots.keys()
                }
            # Concatenate all training data into one cuDF DataFrame
            self.train_df = cudf.concat(list(train_spot_tables.values()), ignore_index=True)
            self.logger.info(f"Training data loaded. Shape: {self.train_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            raise
    
    def prepare_data(self) -> None:
        """
        Prepare data by splitting features and target labels.
        If cross validation is not used, also create a train/validation split.
        """
        self.logger.info("Preparing training data on GPU...")
        if 'x' not in self.train_df.columns or 'y' not in self.train_df.columns:
            raise ValueError("Input data must contain 'x' and 'y' columns.")
        # Separate features and labels
        self.X = self.train_df[['x', 'y']]
        self.y = self.train_df.drop(columns=['x', 'y'])
        
        if not self.use_cv:
            # Create a train/validation split using cuML's train_test_split
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            self.logger.info("Data split into training and validation sets.")
    
    def define_models(self) -> None:
        """
        Define the models to be used in the pipeline.
        """
        self.logger.info("Defining GPU models...")
        # For demonstration, we use only LinearRegression; you can add other models and hyperparameters.
        self.models = {
            "LinearRegression": LinearRegression()
        }
        self.logger.info(f"Models defined: {list(self.models.keys())}")
    
    def cross_validate_models(self) -> dict:
        """
        Perform K-fold cross validation on the entire training set.
        Returns a dictionary with model names as keys and the average MSE per model as values.
        """
        self.logger.info(f"Starting {self.n_splits}-fold cross validation...")
        cv_scores = {}
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Loop over each model
        for model_name, model in self.models.items():
            self.logger.info(f"Cross-validating model: {model_name}")
            fold_scores = []
            # kf.split returns indices for the folds
            for fold, (train_idx, valid_idx) in enumerate(kf.split(self.X)):
                # Select training and validation folds using .iloc
                X_train_fold = self.X.iloc[train_idx]
                X_valid_fold = self.X.iloc[valid_idx]
                y_train_fold = self.y.iloc[train_idx]
                y_valid_fold = self.y.iloc[valid_idx]
                
                # Train on the current fold
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_valid_fold)
                
                # Convert predictions to a NumPy array for error computation
                if isinstance(preds, cudf.DataFrame):
                    preds = preds.to_numpy()
                elif isinstance(preds, cp.ndarray):
                    preds = cp.asnumpy(preds)
                
                # Compute Mean Squared Error (MSE)
                mse = np.mean((y_valid_fold.to_numpy() - preds) ** 2)
                fold_scores.append(mse)
                self.logger.info(f"Fold {fold + 1} MSE for {model_name}: {mse}")
            avg_mse = np.mean(fold_scores)
            cv_scores[model_name] = avg_mse
            self.logger.info(f"{model_name} Average CV MSE: {avg_mse}")
        return cv_scores
    
    def train_models(self) -> None:
        """
        Train the defined models using the training split.
        (This method is used only when not using cross validation.)
        """
        self.logger.info("Training models on GPU...")
        for name, model in self.models.items():
            self.logger.info(f"Training {name} on training split...")
            model.fit(self.X_train, self.y_train)
        self.logger.info("Model training complete.")
    
    def validate_models(self) -> dict:
        """
        Validate models on the validation set and return their predictions.
        Optionally, evaluation metrics (e.g., MSE) can be computed here.
        (This method is used only when not using cross validation.)
        """
        self.logger.info("Validating models on the validation set...")
        preds_valid = {}
        for name, model in self.models.items():
            self.logger.info(f"Predicting with {name} on validation data...")
            preds = model.predict(self.X_valid)
            preds_valid[name] = preds
            # Optionally compute and log metrics here.
        return preds_valid
    
    def load_test_data(self) -> None:
        """
        Load test data from the HDF5 file and convert the selected slide's data into a cuDF DataFrame.
        """
        self.logger.info("Loading test data on GPU...")
        try:
            with h5py.File(self.h5_file_path, "r") as f:
                test_spots = f["spots/Test"]
                if self.test_slide not in test_spots:
                    raise ValueError(f"Test slide '{self.test_slide}' not found in file.")
                self.test_df = cudf.DataFrame(np.array(test_spots[self.test_slide]))
            self.logger.info(f"Test data loaded. Shape: {self.test_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise

    def predict_test(self) -> None:
        """
        Predict target values for the test data using ensemble averaging across all models.
        """
        self.logger.info("Predicting on test data on GPU...")
        if 'x' not in self.test_df.columns or 'y' not in self.test_df.columns:
            raise ValueError("Test data must contain 'x' and 'y' columns.")
        X_test = self.test_df[['x', 'y']]
        num_targets = self.y.shape[1]
        test_preds = cp.zeros((X_test.shape[0], num_targets))
        for name, model in self.models.items():
            self.logger.info(f"Predicting with {name} on test data...")
            preds = model.predict(X_test)
            # Convert predictions to Cupy arrays if necessary
            if isinstance(preds, cudf.DataFrame):
                preds = preds.to_cupy()
            elif isinstance(preds, np.ndarray):
                preds = cp.asarray(preds)
            test_preds += preds
        test_preds /= len(self.models)
        # Convert averaged predictions back to a cuDF DataFrame
        self.predictions = cudf.DataFrame(cp.asnumpy(test_preds), columns=self.y.columns)
        self.logger.info("Test predictions complete.")
    
    def create_submission(self) -> None:
        """
        Create a CSV submission file from test predictions.
        """
        self.logger.info("Creating submission file...")
        submission_df = self.predictions.copy()
        # Insert an 'ID' column from the test DataFrame index (converted to CPU pandas Index)
        submission_df.insert(0, 'ID', self.test_df.index.to_pandas())
        submission_df.to_csv(self.submission_path, index=False)
        self.logger.info(f"Submission file '{self.submission_path}' created!")
    
    def run_pipeline(self) -> None:
        """
        Run the entire pipeline:
          1. Load and prepare training data.
          2. Define models.
          3. Either perform cross validation or use a single train/validation split.
          4. Load test data, predict test targets, and create a submission file.
        """
        self.load_train_data()
        self.prepare_data()
        self.define_models()
        
        if self.use_cv:
            # Perform K-fold cross validation and log the CV metrics.
            cv_scores = self.cross_validate_models()
            self.logger.info(f"Cross validation scores: {cv_scores}")
            # Train final models on the entire training dataset.
            for name, model in self.models.items():
                self.logger.info(f"Training final {name} on entire training data...")
                model.fit(self.X, self.y)
        else:
            # Use the pre-defined train/validation split.
            self.train_models()
            _ = self.validate_models()  # Optionally use or log validation predictions/metrics
        
        self.load_test_data()
        self.predict_test()
        self.create_submission()




if __name__ == '__main__':
    # Example usage; set 'use_cv=True' to enable cross validation.
    pipeline = ElucidataGPUPipeline(
        h5_file_path='path/to/file.h5',
        test_slide='S_7',
        submission_path='submission.csv',
        use_cv=True,      # Set to True to use cross validation folding
        n_splits=5        # Number of folds for cross validation
    )
    pipeline.run_pipeline()


