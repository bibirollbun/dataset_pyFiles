import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch  # Added import to fix NameError: name 'torch' is not defined
from cuml.preprocessing import TargetEncoder  # GPU-accelerated target encoding

# Scikit-learn imports for cross-validation and metrics
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas.api.types as ptypes  # For checking numeric types

# For Optuna-based optimization:
import optuna

# Import CatBoostRegressor (make sure to install catboost: !pip install catboost)
from catboost import CatBoostRegressor

warnings.simplefilter('ignore')

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_seed(42)

###############################################################################
# CatBoost-Only Pipeline
###############################################################################
class CatBoostPipeline:
    """
    A pipeline for preprocessing, hyperparameter tuning, final model training,
    and test prediction using CatBoostRegressor.
    
    This pipeline:
      1. Applies target encoding to each original feature.
      2. Ensures categorical columns are processed.
      3. Combines original features with their target‑encoded versions.
      4. Optionally creates interaction features if the total count isn’t huge.
      5. Optionally scales numerical features via StandardScaler.
      
    Hyperparameter tuning is performed using Optuna over a refined search space.
    Next, instead of a simple hold-out split, the final model is trained with KFold
    cross-validation to obtain out‑of‑fold (OOF) predictions and averaged test predictions.
    Finally, a full model is trained on the complete training set, saved to a file
    (catboost_backpack_v1.pth), and used for test predictions.
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
        
        self.cat_best_params = None
        self.best_cv_rmse_cat = None
        self.model_cat = None
        self.all_features = None
        self.metrics_log = []
        self.model_path = "catboost_backpack_v1.pth"  # File to save the pretrained model

    def log(self, message: str):
        print(message)
        self.metrics_log.append(message)

    def save_metrics_log(self, filename="catboost_model_metrics_log.txt"):
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    def preprocess_data(self):
        start_time = time.time()
        self.log("Starting data preprocessing...")
        # 1. Apply target encoding to each original feature.
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        
        # 2. Ensure categorical columns are properly set.
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        
        # 3. Combine original features and their TE versions.
        self.all_features = self.features + [f"TE_{col}" for col in self.features]
        
        # 4. (Optional) Create interaction features if total count isn’t huge.
        if len(self.cats) * len(self.features) < 50:
            for cat_col in self.cats:
                for num_col in self.features:
                    inter_col = f"{cat_col}_x_{num_col}"
                    self.train[inter_col] = (self.train[cat_col].astype(str) + "_" + self.train[num_col].astype(str)).astype('category').cat.codes
                    self.test[inter_col] = (self.test[cat_col].astype(str) + "_" + self.test[num_col].astype(str)).astype('category').cat.codes
                    self.all_features.append(inter_col)
        
        # 5. Convert non-numeric features in all_features to numeric codes.
        for col in self.all_features:
            if not ptypes.is_numeric_dtype(self.train[col]):
                self.train[col] = self.train[col].astype('category').cat.codes
                self.test[col] = self.test[col].astype('category').cat.codes

        # 6. Fill missing values (using median) and scale features.
        for col in self.all_features:
            median_val = self.train[col].median()
            self.train[col] = self.train[col].fillna(median_val)
            self.test[col] = self.test[col].fillna(median_val)
        scaler = StandardScaler()
        self.train[self.all_features] = scaler.fit_transform(self.train[self.all_features])
        self.test[self.all_features] = scaler.transform(self.test[self.all_features])
        self.scaler = scaler

        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. Features: {self.all_features} (Time taken: {elapsed:.2f} sec)")

    def tune_catboost(self, n_trials: int = 50):
        """
        Use Optuna to tune hyperparameters for CatBoostRegressor on a subsample
        with 2-fold CV to speed up execution.
        """
        self.log("Starting hyperparameter tuning with Optuna for CatBoostRegressor...")
        # Use a subsample to speed up tuning.
        train_sample = self.train.sample(frac=0.5, random_state=self.random_state)
        def objective_cat(trial):
            params = {
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "iterations": trial.suggest_int("iterations", 300, 1000),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            }
            # Always use GPU if available.
            params["task_type"] = "GPU" if torch.cuda.is_available() else "CPU"
            params["verbose"] = 0

            cv = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
            cv_scores = []
            for train_idx, val_idx in cv.split(train_sample):
                X_train_cv = train_sample.iloc[train_idx][self.all_features]
                y_train_cv = train_sample.iloc[train_idx][self.target]
                X_val_cv = train_sample.iloc[val_idx][self.all_features]
                y_val_cv = train_sample.iloc[val_idx][self.target]

                model = CatBoostRegressor(random_state=self.random_state, **params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                preds = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
                cv_scores.append(rmse)
            return np.mean(cv_scores)

        study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study_cat.optimize(objective_cat, n_trials=n_trials)
        self.cat_best_params = study_cat.best_trial.params
        self.best_cv_rmse_cat = study_cat.best_value
        self.log(f"CatBoost tuning complete. Best params: {self.cat_best_params}")
        self.log(f"Best CV RMSE (CatBoost): {self.best_cv_rmse_cat:.4f}")

    def train_kfold_model(self, n_folds: int = 5, early_stopping_rounds: int = 20):
        """
        Train CatBoost using KFold cross-validation to generate out-of-fold (OOF)
        predictions on the training data and averaged predictions for the test set.
        """
        self.log("Starting KFold training for OOF predictions...")
        oof_preds = np.zeros(len(self.train))
        test_preds = np.zeros(len(self.test))
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        # tqdm progress bar for folds
        for fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(self.train), total=n_folds, desc="KFold Training"), 1):
            X_train = self.train.iloc[train_idx][self.all_features]
            y_train = self.train.iloc[train_idx][self.target]
            X_val = self.train.iloc[val_idx][self.all_features]
            y_val = self.train.iloc[val_idx][self.target]
            
            # Use tuned hyperparameters if available
            params = self.cat_best_params.copy() if self.cat_best_params is not None else {}
            params["task_type"] = "GPU" if torch.cuda.is_available() else "CPU"
            params["verbose"] = 0
            model = CatBoostRegressor(random_state=self.random_state, **params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=early_stopping_rounds,
                      verbose=False)
            
            # Predict on validation fold and on test set
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(self.test[self.all_features]) / n_folds
            
            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            self.log(f"Fold {fold} RMSE: {fold_rmse:.4f}")
        
        overall_rmse = np.sqrt(mean_squared_error(self.train[self.target], oof_preds))
        self.log(f"Overall CV RMSE (OOF): {overall_rmse:.4f}")
        self.oof_preds = oof_preds
        self.test_preds = test_preds

    def save_model(self, filename="catboost_backpack_v1.pth"):
        start_time = time.time()
        self.model_cat.save_model(filename)
        elapsed = time.time() - start_time
        self.log(f"Pretrained model saved to {filename}. (Time taken: {elapsed:.2f} sec)")

    def load_model(self, filename="catboost_backpack_v1.pth"):
        start_time = time.time()
        self.model_cat = CatBoostRegressor(verbose=0)
        self.model_cat.load_model(filename)
        elapsed = time.time() - start_time
        self.log(f"Pretrained model loaded from {filename}. (Time taken: {elapsed:.2f} sec)")

    def predict_test(self):
        start_time = time.time()
        test_preds = self.model_cat.predict(self.test[self.all_features])
        elapsed = time.time() - start_time
        self.log(f"Test prediction complete. (Time taken: {elapsed:.2f} sec)")
        return test_preds

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} sec)")

    def run_pipeline(self):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning (CatBoost)", self.tune_catboost),
            ("KFold Training & OOF Predictions (CatBoost)", self.train_kfold_model),
        ]
        # Iterate with tqdm progress over pipeline steps.
        for step_name, step_func in tqdm(steps, desc="Pipeline Steps", unit="step"):
            self.log(f"----- Starting step: {step_name} -----")
            # For the KFold training step, pass required arguments.
            if step_name == "KFold Training & OOF Predictions (CatBoost)":
                step_func(n_folds=5, early_stopping_rounds=20)
            else:
                step_func()
            self.log(f"----- Completed step: {step_name} -----")
        
        # Train final model on the full training set for saving and final test predictions.
        self.log("Training final model on full training set...")
        params = self.cat_best_params.copy() if self.cat_best_params is not None else {}
        params["task_type"] = "GPU" if torch.cuda.is_available() else "CPU"
        params["verbose"] = 0
        self.model_cat = CatBoostRegressor(random_state=self.random_state, **params)
        self.model_cat.fit(self.train[self.all_features], self.train[self.target], verbose=False)
        
        # Save and load the model.
        self.save_model(filename=self.model_path)
        self.load_model(filename=self.model_path)
        
        # Predict test set using the final full-model.
        self.log("Predicting Test Set using final full model...")
        predictions_final = self.predict_test()
        self.save_submission(predictions_final, filename="submission_final.csv")
        
        # Also save submission using KFold ensemble predictions (if available).
        if hasattr(self, 'test_preds'):
            self.log("Saving KFold ensemble submission...")
            self.save_submission(self.test_preds, filename="submission_kfold.csv")
        
        self.save_metrics_log()
        overall_elapsed = time.time() - overall_start
        self.log("Pipeline execution complete.")
        self.log(f"Total pipeline time: {overall_elapsed:.2f} sec")





###############################################################################
# CatBoost Pipeline Usage
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
    pipeline = CatBoostPipeline(train=train, test=test, target=target, features=features, cats=cats)
    pipeline.run_pipeline()



import pandas as pd
df_final = pd.read_csv('submission_final.csv')
df_final


df_kfold = pd.read_csv('submission_kfold.csv')
df_kfold


df_lb_better = pd.read_csv('/kaggle/input/lb-better/lb_better.csv')



blended = df_lb_better.copy()

blended['Price'] = (
    (0.99500) * df_lb_better['Price'] +
    (0.00280) * df_kfold['Price'] +
    (0.00220) * df_final['Price'] 
) 
# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)

