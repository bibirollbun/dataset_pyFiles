import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cuml.preprocessing import TargetEncoder  # GPU-accelerated target encoding

# PyTorch (for XGBoost GPU acceleration, if applicable)
import torch

# Scikit-learn imports for cross-validation and metrics
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# XGBoost modules
from xgboost import XGBRegressor, callback

# For probability distributions
from scipy.stats import uniform, randint

warnings.simplefilter('ignore')

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------ Custom Callback for tqdm --------------------------
class TqdmCallback(callback.TrainingCallback):
    """
    Custom XGBoost training callback that uses tqdm to display a progress bar
    and current RMSE for each iteration during training.
    """
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training Progress", unit="iter")

    def after_iteration(self, model, epoch, evals_log):
        if evals_log and "validation_0" in evals_log and "rmse" in evals_log["validation_0"]:
            current_rmse = evals_log["validation_0"]["rmse"][-1]
            self.pbar.set_postfix(rmse=f"{current_rmse:.4f}")
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model):
        self.pbar.close()
        return model

# ----------------------- XGBPipeline Class Definition -------------------------
class XGBPipeline:
    """
    Pipeline for preprocessing, hyperparameter tuning, final model training,
    and test predictions using XGBoost.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state
        
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        
        self.TE = TargetEncoder(**self.te_params)
        
        self.best_params = None
        self.best_cv_rmse = None
        self.model = None
        self.best_iteration = None
        self.all_features = None
        self.metrics_log = []
    
    def log(self, message: str):
        print(message)
        self.metrics_log.append(message)
    
    def save_metrics_log(self, filename="ml_model_metrics_analysis.txt"):
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        start_time = time.time()
        self.log("Starting data preprocessing...")
        # --- Target Encoding: Create TE_feature for each original feature.
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        
        # Ensure categorical columns are filled and cast as 'category'
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        
        # --- Feature Scaling: For numeric features, fill missing values and scale.
        # We'll use StandardScaler here.
        num_features = [col for col in self.features if col not in self.cats]
        scaler = StandardScaler()
        self.train[num_features] = scaler.fit_transform(self.train[num_features])
        self.test[num_features] = scaler.transform(self.test[num_features])
        self.scaler = scaler  # save scaler if needed
        
        # --- Combine Original and TE features
        self.all_features = self.features + [f"TE_{col}" for col in self.features]
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. All features created: {self.all_features} (Time taken: {elapsed:.2f} sec)")

    def hyperparameter_tuning(self):
        start_time = time.time()
        self.log("Starting hyperparameter tuning using RandomizedSearchCV with parallel processing...")
        # Use a subset of data for faster tuning.
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
        
        # Expanded search space:
        param_distributions = {
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.005, 0.095),  # [0.005, 0.1)
            "min_child_weight": randint(1, 50),
            "subsample": uniform(0.5, 0.5),           # between 0.5 and 1.0
            "colsample_bytree": uniform(0.5, 0.5),      # between 0.5 and 1.0
            "n_estimators": randint(500, 2000)
        }
        
        base_xgb = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
        )
        
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=50,  # Increase iterations for a finer search.
            scoring="neg_root_mean_squared_error",
            cv=3,
            verbose=2,
            random_state=self.random_state,
            n_jobs=-1  # Parallel processing
        )
        
        random_search.fit(train_sample[self.all_features], train_sample[self.target])
        self.best_params = random_search.best_params_
        self.best_cv_rmse = -random_search.best_score_
        elapsed = time.time() - start_time
        self.log("Hyperparameter tuning complete.")
        self.log(f"Best parameters found: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse:.4f}")
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} seconds")

    def train_final_model(self, early_stopping_rounds: int = 100):
        start_time = time.time()
        self.log("Training final model using XGBoost with best hyperparameters...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )
        
        self.model = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        
        tqdm_cb = TqdmCallback(total=self.model.get_params()["n_estimators"])
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tqdm_cb],
            verbose=False
        )
        
        self.best_iteration = self.model.best_iteration
        val_preds = self.model.predict(X_val, iteration_range=(0, self.best_iteration + 1))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        elapsed = time.time() - start_time
        self.log("Final model training complete.")
        self.log(f"Hold-out Validation RMSE: {val_rmse:.4f}")
        self.log(f"Best Iteration: {self.best_iteration}")
        self.log(f"Final model training time: {elapsed:.2f} seconds")

    def predict_test(self):
        start_time = time.time()
        test_predictions = self.model.predict(self.test[self.all_features],
                                               iteration_range=(0, self.best_iteration + 1))
        elapsed = time.time() - start_time
        self.log(f"Test prediction time: {elapsed:.2f} seconds")
        return test_predictions

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} seconds)")

    def run_pipeline(self):
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
import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cuml.preprocessing import TargetEncoder  # GPU-accelerated target encoding

# PyTorch (for XGBoost GPU acceleration, if applicable)
import torch

# Scikit-learn imports for cross-validation and metrics
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# XGBoost modules
from xgboost import XGBRegressor, callback

# For probability distributions
from scipy.stats import uniform, randint

warnings.simplefilter('ignore')

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------ Custom Callback for tqdm --------------------------
class TqdmCallback(callback.TrainingCallback):
    """
    Custom XGBoost training callback that uses tqdm to display a progress bar
    and current RMSE for each iteration during training.
    """
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training Progress", unit="iter")

    def after_iteration(self, model, epoch, evals_log):
        if evals_log and "validation_0" in evals_log and "rmse" in evals_log["validation_0"]:
            current_rmse = evals_log["validation_0"]["rmse"][-1]
            self.pbar.set_postfix(rmse=f"{current_rmse:.4f}")
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model):
        self.pbar.close()
        return model

# ----------------------- XGBPipeline Class Definition -------------------------
class XGBPipeline:
    """
    Pipeline for preprocessing, hyperparameter tuning, final model training,
    and test predictions using XGBoost.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state
        
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        
        self.TE = TargetEncoder(**self.te_params)
        
        self.best_params = None
        self.best_cv_rmse = None
        self.model = None
        self.best_iteration = None
        self.all_features = None
        self.metrics_log = []
    
    def log(self, message: str):
        print(message)
        self.metrics_log.append(message)
    
    def save_metrics_log(self, filename="ml_model_metrics_analysis.txt"):
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        start_time = time.time()
        self.log("Starting data preprocessing...")
        # --- Target Encoding: Create TE_feature for each original feature.
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        
        # Ensure categorical columns are filled and cast as 'category'
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        
        # --- Feature Scaling: For numeric features, fill missing values and scale.
        # We'll use StandardScaler here.
        num_features = [col for col in self.features if col not in self.cats]
        scaler = StandardScaler()
        self.train[num_features] = scaler.fit_transform(self.train[num_features])
        self.test[num_features] = scaler.transform(self.test[num_features])
        self.scaler = scaler  # save scaler if needed
        
        # --- Combine Original and TE features
        self.all_features = self.features + [f"TE_{col}" for col in self.features]
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. All features created: {self.all_features} (Time taken: {elapsed:.2f} sec)")

    def hyperparameter_tuning(self):
        start_time = time.time()
        self.log("Starting hyperparameter tuning using RandomizedSearchCV with parallel processing...")
        # Use a subset of data for faster tuning.
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
        
        # Expanded search space:
        param_distributions = {
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.005, 0.095),  # [0.005, 0.1)
            "min_child_weight": randint(1, 50),
            "subsample": uniform(0.5, 0.5),           # between 0.5 and 1.0
            "colsample_bytree": uniform(0.5, 0.5),      # between 0.5 and 1.0
            "n_estimators": randint(500, 2000)
        }
        
        base_xgb = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
        )
        
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=50,  # Increase iterations for a finer search.
            scoring="neg_root_mean_squared_error",
            cv=3,
            verbose=2,
            random_state=self.random_state,
            n_jobs=-1  # Parallel processing
        )
        
        random_search.fit(train_sample[self.all_features], train_sample[self.target])
        self.best_params = random_search.best_params_
        self.best_cv_rmse = -random_search.best_score_
        elapsed = time.time() - start_time
        self.log("Hyperparameter tuning complete.")
        self.log(f"Best parameters found: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse:.4f}")
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} seconds")

    def train_final_model(self, early_stopping_rounds: int = 100):
        start_time = time.time()
        self.log("Training final model using XGBoost with best hyperparameters...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )
        
        self.model = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        
        tqdm_cb = TqdmCallback(total=self.model.get_params()["n_estimators"])
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tqdm_cb],
            verbose=False
        )
        
        self.best_iteration = self.model.best_iteration
        val_preds = self.model.predict(X_val, iteration_range=(0, self.best_iteration + 1))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        elapsed = time.time() - start_time
        self.log("Final model training complete.")
        self.log(f"Hold-out Validation RMSE: {val_rmse:.4f}")
        self.log(f"Best Iteration: {self.best_iteration}")
        self.log(f"Final model training time: {elapsed:.2f} seconds")

    def predict_test(self):
        start_time = time.time()
        test_predictions = self.model.predict(self.test[self.all_features],
                                               iteration_range=(0, self.best_iteration + 1))
        elapsed = time.time() - start_time
        self.log(f"Test prediction time: {elapsed:.2f} seconds")
        return test_predictions

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} seconds)")

    def run_pipeline(self):
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





