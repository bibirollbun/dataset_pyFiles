import pandas as pd
import numpy as np
import time
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Import cuml’s TargetEncoder (if using RAPIDS cuML; otherwise, use an alternative)
from cuml.preprocessing import TargetEncoder

# Scikit-learn and XGBoost modules
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, callback

# tqdm for progress bars
from tqdm.auto import tqdm
from scipy.stats import uniform, randint


###############################################################################
# Custom Callback for tqdm Progress Bar during XGBoost Training
###############################################################################
class TqdmCallback(callback.TrainingCallback):
    """
    Custom XGBoost training callback that uses tqdm to display a progress bar
    and current RMSE for each iteration during training.
    """
    def __init__(self, total):
        # Initialize the tqdm progress bar with total iterations.
        self.pbar = tqdm(total=total, desc="Training Progress", unit="iter")

    def after_iteration(self, model, epoch, evals_log):
        # Update the progress bar with the current RMSE (if available).
        if evals_log and "validation_0" in evals_log and "rmse" in evals_log["validation_0"]:
            current_rmse = evals_log["validation_0"]["rmse"][-1]
            self.pbar.set_postfix(rmse=f"{current_rmse:.4f}")
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model):
        # Close the progress bar and return the model as required by the API.
        self.pbar.close()
        return model


###############################################################################
# XGBPipeline Class Definition with Logging to File
###############################################################################
class XGBPipeline:
    """
    A class pipeline for preprocessing, hyperparameter tuning, training,
    and prediction using XGBoost on a given dataset.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        """
        Initialize the pipeline with training/test data, target column, and features.
        
        Parameters:
            train (DataFrame): Training dataset.
            test (DataFrame): Test dataset.
            target (str): Target variable name.
            features (list): List of feature column names.
            cats (list): List of categorical column names.
            te_params (dict): Parameters for TargetEncoder (optional).
            sample_frac (float): Fraction of training data to use in hyperparameter tuning.
            random_state (int): Random seed.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state
        
        # Use provided target encoder parameters or default values
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        
        # Initialize the TargetEncoder
        self.TE = TargetEncoder(**self.te_params)
        
        # Placeholder for best hyperparameters, model, and best iteration
        self.best_params = None
        self.best_cv_rmse = None
        self.model = None
        self.best_iteration = None
        self.all_features = None

        # Initialize a list to hold log messages for later saving.
        self.metrics_log = []
    
    def log(self, message: str):
        """
        Logs a message by printing it and appending it to the internal log list.
        """
        print(message)
        self.metrics_log.append(message)
    
    def save_metrics_log(self, filename="ml_model_metrics_analysis.txt"):
        """
        Saves all logged messages into a text file.
        
        Parameters:
            filename (str): The name of the text file to save the log.
        """
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        """
        Preprocess the data:
          - Apply target encoding to each feature.
          - Fill missing values and cast categorical columns.
          - Create a combined feature list that includes both original and target-encoded features.
        """
        start_time = time.time()
        # Create target-encoded features for each original feature
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        
        # Ensure categorical columns are filled and cast as 'category'
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        
        # Combine the original features and target-encoded features
        self.all_features = self.features + [f"TE_{col}" for col in self.features]
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. All features created. (Time taken: {elapsed:.2f} seconds)")
    
    def hyperparameter_tuning(self):
        """
        Run hyperparameter tuning using RandomizedSearchCV on a subset of the training data.
        Sets self.best_params and self.best_cv_rmse.
        """
        start_time = time.time()
        # Take a subset of training data to speed up tuning
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
        
        # Define the parameter search space (narrowed to speed up training)
        param_distributions = {
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.1),  # values between 0.01 and 0.11
            "min_child_weight": randint(1, 100),
            "subsample": uniform(0.5, 1.0),
            "colsample_bytree": uniform(0.5, 1.0),
            "n_estimators": randint(500, 1500),   # lower range for faster training
        }
        
        # Initialize a base XGBRegressor with fixed settings.
        base_xgb = XGBRegressor(
            tree_method="gpu_hist",      # Use GPU acceleration
            enable_categorical=True,
            random_state=self.random_state,
        )
        
        # Set up RandomizedSearchCV with reduced iterations and CV folds.
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=20,  # Reduced number of parameter combinations
            scoring="neg_root_mean_squared_error",
            cv=3,       # Fewer folds to reduce runtime
            verbose=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit the search on the subset of data.
        random_search.fit(train_sample[self.all_features], train_sample[self.target])
        
        # Store the best parameters and CV RMSE.
        self.best_params = random_search.best_params_
        self.best_cv_rmse = -random_search.best_score_
        
        elapsed = time.time() - start_time
        self.log("Hyperparameter tuning complete.")
        self.log(f"Best parameters found: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse}")
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} seconds")
    
    def train_final_model(self, early_stopping_rounds: int = 100):
        """
        Train the final model using the best hyperparameters on the full training data.
        Uses early stopping with a hold-out validation set and displays training progress via tqdm.
        
        Parameters:
            early_stopping_rounds (int): Number of rounds with no improvement to stop training.
        """
        start_time = time.time()
        # Split the full training data into training and hold-out validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )
        
        # Create a new XGBRegressor instance with the best hyperparameters.
        self.model = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        
        # Initialize the tqdm callback for progress monitoring.
        tqdm_cb = TqdmCallback(total=self.model.get_params()["n_estimators"])
        
        # Fit the model using early stopping on the hold-out validation set.
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",            # Monitor RMSE on the validation set
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tqdm_cb],
            verbose=False  # Disable default verbose output in favor of tqdm
        )
        
        # Retrieve the best iteration found via early stopping.
        self.best_iteration = self.model.best_iteration
        
        # Predict on the hold-out validation set using the best iteration.
        val_preds = self.model.predict(X_val, iteration_range=(0, self.best_iteration + 1))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        elapsed = time.time() - start_time
        self.log("Final model training complete.")
        self.log(f"Hold-out Validation RMSE: {val_rmse}")
        self.log(f"Best Iteration: {self.best_iteration}")
        self.log(f"Final model training time: {elapsed:.2f} seconds")
    
    def predict_test(self):
        """
        Generate predictions on the test dataset using the best iteration.
        
        Returns:
            np.array: Predictions for the test set.
        """
        start_time = time.time()
        test_predictions = self.model.predict(self.test[self.all_features],
                                                iteration_range=(0, self.best_iteration + 1))
        elapsed = time.time() - start_time
        self.log(f"Test prediction time: {elapsed:.2f} seconds")
        return test_predictions
    
    def save_submission(self, predictions, filename="submission.csv"):
        """
        Create a submission file using the test set predictions.
        
        Parameters:
            predictions (np.array): Array of predictions.
            filename (str): Name of the CSV file to save.
        """
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} seconds)")
    
    def run_pipeline(self):
        """
        Execute the complete pipeline:
          1. Preprocess data.
          2. Hyperparameter tuning.
          3. Train final model with early stopping.
          4. Generate test predictions.
          5. Save submission file.
          6. Save all logged output to a text file.
        """
        overall_start = time.time()
        self.log("Starting pipeline execution...")
        
        self.preprocess_data()
        self.hyperparameter_tuning()
        self.train_final_model()
        predictions = self.predict_test()
        self.save_submission(predictions)
        
        overall_elapsed = time.time() - overall_start
        self.log("Pipeline execution complete.")
        self.log(f"Total pipeline time: {overall_elapsed:.2f} seconds")
        
        # Save the logged metrics to a text file.
        self.save_metrics_log()





###############################################################################
# TE & XGB pipeline Usage
###############################################################################

if __name__ == "__main__":
    # Load the datasets.
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

    # Define target and feature columns.
    target = "Price"
    features = [col for col in train.columns if col != target]
    
    # Define categorical columns (example: all columns except Price and Weight Capacity)
    cats = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]
    
    # Initialize and run the pipeline.
    pipeline = XGBPipeline(train=train, test=test, target=target, features=features, cats=cats)
    pipeline.run_pipeline()



%%writefile xgb_pipeline_with_tqdb_callback.py
import pandas as pd
import numpy as np
import time
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Import cuml’s TargetEncoder (if using RAPIDS cuML; otherwise, use an alternative)
from cuml.preprocessing import TargetEncoder

# Scikit-learn and XGBoost modules
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, callback

# tqdm for progress bars
from tqdm.auto import tqdm
from scipy.stats import uniform, randint


###############################################################################
# Custom Callback for tqdm Progress Bar during XGBoost Training
###############################################################################
class TqdmCallback(callback.TrainingCallback):
    """
    Custom XGBoost training callback that uses tqdm to display a progress bar
    and current RMSE for each iteration during training.
    """
    def __init__(self, total):
        # Initialize the tqdm progress bar with total iterations.
        self.pbar = tqdm(total=total, desc="Training Progress", unit="iter")

    def after_iteration(self, model, epoch, evals_log):
        # Update the progress bar with the current RMSE (if available).
        if evals_log and "validation_0" in evals_log and "rmse" in evals_log["validation_0"]:
            current_rmse = evals_log["validation_0"]["rmse"][-1]
            self.pbar.set_postfix(rmse=f"{current_rmse:.4f}")
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model):
        # Close the progress bar and return the model as required by the API.
        self.pbar.close()
        return model


###############################################################################
# XGBPipeline Class Definition with Logging to File
###############################################################################
class XGBPipeline:
    """
    A class pipeline for preprocessing, hyperparameter tuning, training,
    and prediction using XGBoost on a given dataset.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        """
        Initialize the pipeline with training/test data, target column, and features.
        
        Parameters:
            train (DataFrame): Training dataset.
            test (DataFrame): Test dataset.
            target (str): Target variable name.
            features (list): List of feature column names.
            cats (list): List of categorical column names.
            te_params (dict): Parameters for TargetEncoder (optional).
            sample_frac (float): Fraction of training data to use in hyperparameter tuning.
            random_state (int): Random seed.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state
        
        # Use provided target encoder parameters or default values
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        
        # Initialize the TargetEncoder
        self.TE = TargetEncoder(**self.te_params)
        
        # Placeholder for best hyperparameters, model, and best iteration
        self.best_params = None
        self.best_cv_rmse = None
        self.model = None
        self.best_iteration = None
        self.all_features = None

        # Initialize a list to hold log messages for later saving.
        self.metrics_log = []
    
    def log(self, message: str):
        """
        Logs a message by printing it and appending it to the internal log list.
        """
        print(message)
        self.metrics_log.append(message)
    
    def save_metrics_log(self, filename="ml_model_metrics_analysis.txt"):
        """
        Saves all logged messages into a text file.
        
        Parameters:
            filename (str): The name of the text file to save the log.
        """
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        """
        Preprocess the data:
          - Apply target encoding to each feature.
          - Fill missing values and cast categorical columns.
          - Create a combined feature list that includes both original and target-encoded features.
        """
        start_time = time.time()
        # Create target-encoded features for each original feature
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        
        # Ensure categorical columns are filled and cast as 'category'
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        
        # Combine the original features and target-encoded features
        self.all_features = self.features + [f"TE_{col}" for col in self.features]
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. All features created. (Time taken: {elapsed:.2f} seconds)")
    
    def hyperparameter_tuning(self):
        """
        Run hyperparameter tuning using RandomizedSearchCV on a subset of the training data.
        Sets self.best_params and self.best_cv_rmse.
        """
        start_time = time.time()
        # Take a subset of training data to speed up tuning
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
        
        # Define the parameter search space (narrowed to speed up training)
        param_distributions = {
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.1),  # values between 0.01 and 0.11
            "min_child_weight": randint(1, 100),
            "subsample": uniform(0.5, 1.0),
            "colsample_bytree": uniform(0.5, 1.0),
            "n_estimators": randint(500, 1500),   # lower range for faster training
        }
        
        # Initialize a base XGBRegressor with fixed settings.
        base_xgb = XGBRegressor(
            tree_method="gpu_hist",      # Use GPU acceleration
            enable_categorical=True,
            random_state=self.random_state,
        )
        
        # Set up RandomizedSearchCV with reduced iterations and CV folds.
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=20,  # Reduced number of parameter combinations
            scoring="neg_root_mean_squared_error",
            cv=3,       # Fewer folds to reduce runtime
            verbose=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit the search on the subset of data.
        random_search.fit(train_sample[self.all_features], train_sample[self.target])
        
        # Store the best parameters and CV RMSE.
        self.best_params = random_search.best_params_
        self.best_cv_rmse = -random_search.best_score_
        
        elapsed = time.time() - start_time
        self.log("Hyperparameter tuning complete.")
        self.log(f"Best parameters found: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse}")
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} seconds")
    
    def train_final_model(self, early_stopping_rounds: int = 100):
        """
        Train the final model using the best hyperparameters on the full training data.
        Uses early stopping with a hold-out validation set and displays training progress via tqdm.
        
        Parameters:
            early_stopping_rounds (int): Number of rounds with no improvement to stop training.
        """
        start_time = time.time()
        # Split the full training data into training and hold-out validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )
        
        # Create a new XGBRegressor instance with the best hyperparameters.
        self.model = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        
        # Initialize the tqdm callback for progress monitoring.
        tqdm_cb = TqdmCallback(total=self.model.get_params()["n_estimators"])
        
        # Fit the model using early stopping on the hold-out validation set.
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",            # Monitor RMSE on the validation set
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tqdm_cb],
            verbose=False  # Disable default verbose output in favor of tqdm
        )
        
        # Retrieve the best iteration found via early stopping.
        self.best_iteration = self.model.best_iteration
        
        # Predict on the hold-out validation set using the best iteration.
        val_preds = self.model.predict(X_val, iteration_range=(0, self.best_iteration + 1))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        elapsed = time.time() - start_time
        self.log("Final model training complete.")
        self.log(f"Hold-out Validation RMSE: {val_rmse}")
        self.log(f"Best Iteration: {self.best_iteration}")
        self.log(f"Final model training time: {elapsed:.2f} seconds")
    
    def predict_test(self):
        """
        Generate predictions on the test dataset using the best iteration.
        
        Returns:
            np.array: Predictions for the test set.
        """
        start_time = time.time()
        test_predictions = self.model.predict(self.test[self.all_features],
                                                iteration_range=(0, self.best_iteration + 1))
        elapsed = time.time() - start_time
        self.log(f"Test prediction time: {elapsed:.2f} seconds")
        return test_predictions
    
    def save_submission(self, predictions, filename="submission.csv"):
        """
        Create a submission file using the test set predictions.
        
        Parameters:
            predictions (np.array): Array of predictions.
            filename (str): Name of the CSV file to save.
        """
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} seconds)")
    
    def run_pipeline(self):
        """
        Execute the complete pipeline:
          1. Preprocess data.
          2. Hyperparameter tuning.
          3. Train final model with early stopping.
          4. Generate test predictions.
          5. Save submission file.
          6. Save all logged output to a text file.
        """
        overall_start = time.time()
        self.log("Starting pipeline execution...")
        
        self.preprocess_data()
        self.hyperparameter_tuning()
        self.train_final_model()
        predictions = self.predict_test()
        self.save_submission(predictions)
        
        overall_elapsed = time.time() - overall_start
        self.log("Pipeline execution complete.")
        self.log(f"Total pipeline time: {overall_elapsed:.2f} seconds")
        
        # Save the logged metrics to a text file.
        self.save_metrics_log()



# Move all necessary files
!cp -r /kaggle/input/aiagenticoptimizedcodellm/other/default/1/* /kaggle/working/
import sys 
sys.path.append('/kaggle/working/AIAgenticOptimizedCodeLLM.py')


import os
from kaggle_secrets import UserSecretsClient
from google import genai
from IPython.display import display, HTML
from AIAgenticOptimizedCodeLLM import AIAgenticOptimizedCodeLLM 


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Get the API key using Kaggle's secrets (adjust as needed for your environment)
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    # Define file paths for metrics, original code, and the output HTML file.
    metrics_file_path = "/kaggle/working/ml_model_metrics_analysis.txt"
    code_file_path = "/kaggle/working/xgb_pipeline_with_tqdb_callback.py"
    optimized_html_file_path = "/kaggle/working/xgb_pipeline_with_tqdb_callback_v1.html"

    # User prompt for improvements
    user_prompt = """
Please provide ways to improve the ML model metrics such as RMSE.
Additionally, share suggestions on how to generalize the ML model.
Specifically:
1. How can I reduce the RMSE score of my model?
2. How can I ensure my model generalizes well to unseen data?
3. share any deep learning with feed forward & back propagation models & any vector 
Regression with xgb, lgbm and catboost meta ensemble !!  ensemble all these meta models
and save submission.csv file similar like all these xgb pipeline features time logging & tqdm progress & callback?
"""

    optimizer = AIAgenticOptimizedCodeLLM(api_key, metrics_file_path, code_file_path, optimized_html_file_path)
    html_response = optimizer.optimize_model_code(user_prompt)

    #print("Final HTML Response:")
    #print(html_response)
    #print("Final HTML Response:")
    display(HTML(html_response))





