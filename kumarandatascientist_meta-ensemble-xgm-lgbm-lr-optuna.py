import pandas as pd
import numpy as np
import time
import warnings
warnings.simplefilter('ignore')

# Cuml's TargetEncoder (GPU‐accelerated)
from cuml.preprocessing import TargetEncoder

# Models and utilities
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, callback
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from tqdm.auto import tqdm
import optuna
from sklearn.preprocessing import StandardScaler
import pandas.api.types as ptypes  # For checking numeric types

###############################################################################
# Custom Callback for tqdm Progress Bar during XGBoost Training
###############################################################################
class TqdmCallback(callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="XGB Training", unit="iter")

    def after_iteration(self, model, epoch, evals_log):
        if evals_log and "validation_0" in evals_log and "rmse" in evals_log["validation_0"]:
            current_rmse = evals_log["validation_0"]["rmse"][-1]
            self.pbar.set_postfix(rmse=f"{current_rmse:.4f}")
        self.pbar.update(1)
        return False  # Continue training

    def after_training(self, model):
        self.pbar.close()
        return model

###############################################################################
# Enhanced Pipeline with Optuna Hyperparameter Tuning and Stacking Ensemble
###############################################################################
class XGBPipeline:
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

        # Default target encoder parameters
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        self.TE = TargetEncoder(**self.te_params)

        self.best_params = None
        self.best_cv_rmse = None
        self.model_xgb = None
        self.model_lgbm = None
        self.model_lr = None
        self.stacking_model = None
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

        # 1. Target encoding on each original feature.
        for col in self.features:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])

        # 2. Ensure categorical columns are properly set.
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')

        # 3. Combine original and target‐encoded features.
        self.all_features = self.features + [f"TE_{col}" for col in self.features]

        # 4. Create interaction features if the total isn’t huge.
        if len(self.cats) * len(self.features) < 50:
            for cat_col in self.cats:
                for num_col in self.features:
                    inter_col = f'{cat_col}_x_{num_col}'
                    self.train[inter_col] = self.train[cat_col].astype(str) + '_' + self.train[num_col].astype(str)
                    self.test[inter_col] = self.test[cat_col].astype(str) + '_' + self.test[num_col].astype(str)
                    self.train[inter_col] = self.train[inter_col].astype('category').cat.codes
                    self.test[inter_col] = self.test[inter_col].astype('category').cat.codes
                    self.all_features.append(inter_col)

        # 5. Convert any non-numeric columns (in the feature set) to numeric codes.
        for col in self.all_features:
            if not ptypes.is_numeric_dtype(self.train[col]):
                self.train[col] = self.train[col].astype('category').cat.codes
                self.test[col] = self.test[col].astype('category').cat.codes

        # 6. Fill missing values (using the median) and scale features.
        for col in self.all_features:
            median_val = self.train[col].median()
            self.train[col] = self.train[col].fillna(median_val)
            self.test[col] = self.test[col].fillna(median_val)
        scaler = StandardScaler()
        self.train[self.all_features] = scaler.fit_transform(self.train[self.all_features])
        self.test[self.all_features] = scaler.transform(self.test[self.all_features])

        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. (Time taken: {elapsed:.2f} sec)")

    def hyperparameter_tuning(self, n_trials: int = 20):
        """
        Use Optuna to tune hyperparameters for XGBoost on a subsample with 3-fold CV.
        """
        self.log("Starting hyperparameter tuning with Optuna...")
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
            }

            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_scores = []
            for train_idx, val_idx in cv.split(train_sample):
                X_train_cv = train_sample.iloc[train_idx][self.all_features]
                y_train_cv = train_sample.iloc[train_idx][self.target]
                X_val_cv = train_sample.iloc[val_idx][self.all_features]
                y_val_cv = train_sample.iloc[val_idx][self.target]

                model = XGBRegressor(
                    tree_method="gpu_hist",
                    enable_categorical=True,
                    random_state=self.random_state,
                    **params
                )
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    eval_metric="rmse",
                    early_stopping_rounds=50,
                    verbose=False
                )
                preds = model.predict(X_val_cv, iteration_range=(0, model.best_iteration + 1))
                rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
                cv_scores.append(rmse)
            return np.mean(cv_scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.best_cv_rmse = study.best_value
        self.log(f"Hyperparameter tuning complete. Best params: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse}")

    def train_final_models(self, early_stopping_rounds: int = 50):
        """
        Train base models (XGBoost, LightGBM, Linear Regression) on a hold-out split,
        then build a stacking ensemble.
        """
        start_time = time.time()
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.all_features], self.train[self.target],
            test_size=0.2, random_state=self.random_state
        )

        # --------------------- XGBoost ---------------------
        self.log("Training XGBoost...")
        self.model_xgb = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        tqdm_cb_xgb = TqdmCallback(total=self.model_xgb.get_params()["n_estimators"])
        self.model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tqdm_cb_xgb],
            verbose=False
        )
        self.best_iteration = self.model_xgb.best_iteration
        preds_xgb = self.model_xgb.predict(X_val, iteration_range=(0, self.best_iteration + 1))
        rmse_xgb = np.sqrt(mean_squared_error(y_val, preds_xgb))
        self.log(f"XGBoost hold-out RMSE: {rmse_xgb:.4f} (Best Iteration: {self.best_iteration})")

        # --------------------- LightGBM ---------------------
        self.log("Training LightGBM...")
        # Use GPU acceleration for LightGBM by adding the device parameter
        lgb_params = self.best_params.copy()
        lgb_params["device"] = "gpu"
        self.model_lgbm = lgb.LGBMRegressor(random_state=self.random_state, **lgb_params)
        self.model_lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_rounds)]
        )
        preds_lgbm = self.model_lgbm.predict(X_val)
        rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds_lgbm))
        self.log(f"LightGBM hold-out RMSE: {rmse_lgbm:.4f}")

        # --------------------- Linear Regression ---------------------
        self.log("Training Linear Regression...")
        self.model_lr = LinearRegression()
        self.model_lr.fit(X_train, y_train)
        preds_lr = self.model_lr.predict(X_val)
        rmse_lr = np.sqrt(mean_squared_error(y_val, preds_lr))
        self.log(f"Linear Regression hold-out RMSE: {rmse_lr:.4f}")

        # --------------------- Stacking Ensemble ---------------------
        self.log("Training stacking ensemble...")
        self.stacking_model = StackingRegressor(
            estimators=[
                ('xgb', self.model_xgb),
                ('lgbm', self.model_lgbm),
                ('lr', self.model_lr)
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=KFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
            passthrough=True  # Optionally, let the meta-model see original features too.
        )
        # Fit stacking on the entire training data.
        self.stacking_model.fit(self.train[self.all_features], self.train[self.target])
        # Evaluate stacking on the hold-out set
        stacking_preds = self.stacking_model.predict(X_val)
        rmse_stack = np.sqrt(mean_squared_error(y_val, stacking_preds))
        self.log(f"Stacking Ensemble hold-out RMSE: {rmse_stack:.4f}")

        elapsed = time.time() - start_time
        self.log(f"Final model training complete. (Time taken: {elapsed:.2f} sec)")

    def predict_test(self):
        """
        Use the stacking ensemble to predict on the test set.
        """
        start_time = time.time()
        test_preds = self.stacking_model.predict(self.test[self.all_features])
        elapsed = time.time() - start_time
        self.log(f"Test prediction complete. (Time taken: {elapsed:.2f} sec)")
        return test_preds

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({ "id": self.test.index, self.target: predictions })
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} sec)")

    def run_pipeline(self):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning", self.hyperparameter_tuning),
            ("Training Final Models", self.train_final_models),
        ]
        
        with tqdm(total=len(steps), desc="Pipeline Steps", unit="step") as pbar:
            self.log("Starting pipeline execution...")
            for step_name, step_func in steps:
                self.log(f"Starting step: {step_name}")
                step_func()
                pbar.update(1)
                self.log(f"Completed step: {step_name}")
        
        self.log("Predicting Test Set...")
        predictions = self.predict_test()
        self.log("Saving Submission...")
        self.save_submission(predictions)
        self.save_metrics_log()
        
        overall_elapsed = time.time() - overall_start
        self.log("Pipeline execution complete.")
        self.log(f"Total pipeline time: {overall_elapsed:.2f} sec")



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


import pandas as pd

df = pd.read_csv('submission.csv')


df.head(10)

