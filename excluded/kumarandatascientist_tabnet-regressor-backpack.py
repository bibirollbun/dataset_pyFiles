!pip install skorch


!pip install pytorch-tabnet



from pytorch_tabnet.tab_model import TabNetRegressor



import pandas as pd
import numpy as np
import time
import warnings
import random
from tqdm.auto import tqdm
from cuml.preprocessing import TargetEncoder  # RAPIDS cuML target encoder

# PyTorch (used by TabNet internally)
import torch

# Scikit-learn imports for cross-validation and metrics
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# For Optuna-based optimization:
import optuna

# Import TabNetRegressor (ensure you have installed pytorch-tabnet via !pip install pytorch-tabnet)
from pytorch_tabnet.tab_model import TabNetRegressor

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Set reproducibility seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


###############################################################################
# LRPipeline Class Definition using TabNetRegressor
###############################################################################
class LRPipeline:
    """
    This pipeline performs:
      1. Data preprocessing (target encoding for categoricals, imputation and scaling for numerics,
         optional log transformation and aggregated features).
      2. K-fold cross-validation with out-of-fold (OOF) RMSE computation using TabNetRegressor.
      3. Final model tuning on full training data using either Optuna or RandomizedSearchCV.
      4. Saving/loading of the final TabNetRegressor model and generating test predictions.
      
    An optional parameter fast_mode can be set to True to use smaller search spaces,
    fewer trials/iterations, and a lower max_epochs for faster execution.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 random_state: int = 42):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats  # columns for target encoding
        self.random_state = random_state

        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params

        self.TE = TargetEncoder(**self.te_params)
        self.all_features = None
        self.metrics_log = []
        self.model = None

        # Default training parameters (starting values)
        self.best_params = {
            "max_epochs": 100,
            "learning_rate": 0.01,
            "weight_decay": 0.0,   # L2 regularization (default off)
            "early_stopping_rounds": 10
        }

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
        # (1) Target encode categorical columns.
        for col in self.cats:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        # (2) Numeric features: those in features not in cats.
        num_cols = [col for col in self.features if col not in self.cats]
        self.train[num_cols] = self.train[num_cols].fillna(self.train[num_cols].mean())
        self.test[num_cols] = self.test[num_cols].fillna(self.train[num_cols].mean())
        # Min-max normalization
        for col in num_cols:
            min_val = self.train[col].min()
            max_val = self.train[col].max()
            if max_val - min_val == 0:
                self.train[col] = 0.0
                self.test[col] = 0.0
            else:
                self.train[col] = (self.train[col] - min_val) / (max_val - min_val)
                self.test[col] = (self.test[col] - min_val) / (max_val - min_val)
        # (3) Optional log transformation on "Weight Capacity (kg)"
        if "Weight Capacity (kg)" in num_cols:
            self.train["Weight Capacity (kg)"] = self.train["Weight Capacity (kg)"].clip(lower=0)
            self.test["Weight Capacity (kg)"] = self.test["Weight Capacity (kg)"].clip(lower=0)
            self.train["log_Weight_Capacity"] = np.log1p(self.train["Weight Capacity (kg)"])
            self.test["log_Weight_Capacity"] = np.log1p(self.test["Weight Capacity (kg)"])
            self.log("Created log-transformed Weight Capacity feature: log_Weight_Capacity")
        else:
            self.log("Weight Capacity (kg) not found among numeric features.")
        # (4) Aggregated TE features from target-encoded columns.
        te_cols = [f"TE_{col}" for col in self.cats]
        self.train["sum_TE"] = self.train[te_cols].sum(axis=1)
        self.train["mean_TE"] = self.train[te_cols].mean(axis=1)
        self.train["std_TE"] = self.train[te_cols].std(axis=1)
        self.test["sum_TE"] = self.test[te_cols].sum(axis=1)
        self.test["mean_TE"] = self.test[te_cols].mean(axis=1)
        self.test["std_TE"] = self.test[te_cols].std(axis=1)
        self.log("Created aggregated TE features: sum_TE, mean_TE, std_TE")
        # (5) Final feature set: numeric + TE features + engineered features.
        engineered_features = []
        if "Weight Capacity (kg)" in num_cols:
            engineered_features.append("log_Weight_Capacity")
        engineered_features += ["sum_TE", "mean_TE", "std_TE"]
        self.all_features = num_cols + te_cols + engineered_features
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. Features created: {self.all_features} (Time taken: {elapsed:.2f} sec)")

    def hyperparameter_tuning(self, n_trials: int = 20):
        start_time = time.time()
        self.log("Skipping default hyperparameter tuning. Using default parameters:")
        self.log(str(self.best_params))
        elapsed = time.time() - start_time
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} sec")

    def train_kfold_model(self, n_splits: int = 5, fast_mode: bool = False):
        overall_start = time.time()
        self.log("Starting k-fold training with out-of-fold validation using TabNetRegressor...")
        # If fast_mode is True, override max_epochs for faster training.
        fold_max_epochs = 50 if fast_mode else self.best_params["max_epochs"]
        X = self.train[self.all_features].values.astype(np.float32)
        y = self.train[self.target].values.astype(np.float32).reshape(-1, 1)
        oof_preds = np.zeros_like(y)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_iter = tqdm(kf.split(X), total=n_splits, desc="Folds")
        for fold, (train_idx, val_idx) in enumerate(fold_iter, 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = TabNetRegressor(
                verbose=0,
                seed=self.random_state,
                optimizer_params={
                    "lr": self.best_params["learning_rate"],
                    "weight_decay": self.best_params["weight_decay"]
                }
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=["rmse"],
                max_epochs=fold_max_epochs,
                patience=self.best_params["early_stopping_rounds"],
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            y_val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, y_val_pred)
            fold_rmse = np.sqrt(val_loss)
            self.log(f"Fold {fold}: Best Validation RMSE: {fold_rmse:.4f}")
            oof_preds[val_idx] = y_val_pred.reshape(-1, 1)
        overall_mse = mean_squared_error(y, oof_preds)
        overall_rmse = np.sqrt(overall_mse)
        self.log(f"Overall Out-of-Fold RMSE (TabNetRegressor): {overall_rmse:.4f}")
        overall_elapsed = time.time() - overall_start
        self.log(f"K-Fold training complete. Total time: {overall_elapsed:.2f} sec")
        self.oof_rmse = overall_rmse
        self.oof_preds = oof_preds

    def tune_final_model_optuna(self, validation_split: float = 0.2, n_trials: int = 20, fast_mode: bool = False):
        self.log("Starting Optuna hyperparameter tuning on full training data using TabNetRegressor...")
        # In fast_mode, reduce number of trials and max_epochs range.
        X_full = self.train[self.all_features].values.astype(np.float32)
        y_full = self.train[self.target].values.astype(np.float32).reshape(-1, 1)
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_full, y_full, test_size=validation_split, random_state=self.random_state)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        def objective(trial):
            lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
            # In fast_mode, use a lower max_epochs range.
            if fast_mode:
                max_epochs = trial.suggest_int("max_epochs", 30, 80)
            else:
                max_epochs = trial.suggest_int("max_epochs", 50, 200)
            wd = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
            model = TabNetRegressor(
                verbose=0,
                seed=self.random_state,
                optimizer_params={"lr": lr, "weight_decay": wd}
            )
            model.fit(
                X_train_full, y_train_full,
                eval_set=[(X_val, y_val)],
                eval_metric=["rmse"],
                max_epochs=max_epochs,
                patience=self.best_params["early_stopping_rounds"],
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            y_val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, y_val_pred)
            return val_loss

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        self.log(f"Optuna tuning complete. Best parameters: {best_params}, Best Val Loss: {study.best_trial.value:.4f}")
        # Retrain final model on full training data with best parameters.
        final_model = TabNetRegressor(
            verbose=0,
            seed=self.random_state,
            optimizer_params={"lr": best_params["lr"], "weight_decay": best_params["weight_decay"]}
        )
        final_model.fit(
            X_full, y_full,
            eval_set=[(X_full, y_full)],  # training on full data
            eval_metric=["rmse"],
            max_epochs=best_params["max_epochs"],
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        self.model = final_model
        self.log("Final model retraining with Optuna best parameters complete.")

    def tune_final_model_randomizedsearch(self, cv: int = 3, n_iter: int = 10, fast_mode: bool = False):
        self.log("Starting RandomizedSearchCV hyperparameter tuning using TabNetRegressor...")
        # Adjust candidate grid based on fast_mode.
        if fast_mode:
            max_epochs_candidates = [50, 60, 80]
            lr_candidates = [0.005, 0.01, 0.02]
            wd_candidates = [0.0, 0.001, 0.01]
        else:
            max_epochs_candidates = [100, 150, 200]
            lr_candidates = [0.005, 0.01, 0.02, 0.03]
            wd_candidates = [0.0, 0.001, 0.01, 0.1]
        net = TabNetRegressor(
            verbose=0,
            seed=self.random_state
        )
        param_dist = {
            "max_epochs": max_epochs_candidates,
            "optimizer_params": [{"lr": lr, "weight_decay": wd} 
                                 for lr in lr_candidates
                                 for wd in wd_candidates]
        }
        rs = RandomizedSearchCV(
            net,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            verbose=0
        )
        X_full = self.train[self.all_features].values.astype(np.float32)
        y_full = self.train[self.target].values.astype(np.float32).ravel()
        rs.fit(X_full, y_full)
        best_params = rs.best_params_
        best_score = rs.best_score_
        self.log(f"RandomizedSearchCV tuning complete. Best parameters: {best_params}, Best CV MSE: {-best_score:.4f}")
        final_net = TabNetRegressor(
            verbose=0,
            max_epochs=best_params['max_epochs'],
            seed=self.random_state,
            optimizer_params=best_params['optimizer_params']
        )
        final_net.fit(X_full, y_full,
                      eval_set=[(X_full, y_full)],
                      eval_metric=["rmse"],
                      batch_size=1024,
                      virtual_batch_size=128,
                      num_workers=0,
                      drop_last=False)
        self.model = final_net
        self.log("Final model retraining with RandomizedSearchCV best parameters complete.")

    def save_pretrained_model(self, filename="tabnet_model.zip"):
        start_time = time.time()
        self.model.save_model(filename)
        elapsed = time.time() - start_time
        self.log(f"Pretrained model saved to {filename} (Time taken: {elapsed:.2f} sec)")

    def load_pretrained_model(self, filename="tabnet_model.zip"):
        start_time = time.time()
        self.model = TabNetRegressor(verbose=0)
        self.model.load_model(filename)
        elapsed = time.time() - start_time
        self.log(f"Pretrained model loaded from {filename} (Time taken: {elapsed:.2f} sec)")

    def predict_test(self):
        start_time = time.time()
        X_test_np = self.test[self.all_features].values.astype(np.float32)
        preds = self.model.predict(X_test_np)
        elapsed = time.time() - start_time
        self.log(f"Test predictions complete. (Time taken: {elapsed:.2f} sec)")
        return preds

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename} (Time taken: {elapsed:.2f} sec)")

    def run_pipeline(self, n_trials: int = 100, n_splits: int = 5, tuning_method: str = 'optuna', fast_mode: bool = False):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning (default)", lambda: self.hyperparameter_tuning(n_trials)),
            ("K-Fold Training", lambda: self.train_kfold_model(n_splits, fast_mode=fast_mode))
        ]
        for step_name, step_func in steps:
            self.log(f"----- Starting step: {step_name} -----")
            step_func()
            self.log(f"----- Completed step: {step_name} -----")
        
        # Choose final model tuning method:
        if tuning_method.lower() == 'optuna':
            self.log("----- Tuning Final Model on Full Data with Optuna -----")
            self.tune_final_model_optuna(validation_split=0.2, n_trials=n_trials if not fast_mode else 20)
        elif tuning_method.lower() == 'randomized':
            self.log("----- Tuning Final Model on Full Data with RandomizedSearchCV -----")
            self.tune_final_model_randomizedsearch(cv=3, n_iter=10 if not fast_mode else 5, fast_mode=fast_mode)
        else:
            self.log("----- Using default hyperparameters for Final Model Training -----")
            self.tune_final_model_optuna(validation_split=0.2, n_trials=1)  # trivial run

        self.log("----- Saving Pretrained Model -----")
        self.save_pretrained_model()
        self.log("----- Loading Pretrained Model for Predictions -----")
        self.load_pretrained_model()
        predictions = self.predict_test()
        self.save_submission(predictions)
        self.save_metrics_log()
        overall_elapsed = time.time() - overall_start
        self.log(f"Pipeline execution complete. Total time: {overall_elapsed:.2f} sec")





###############################################################################
# Pipeline Usage with TabNetRegressor, Enhanced Feature Engineering,
# K-Fold CV, and Hyperparameter Tuning via Optuna or RandomizedSearchCV,
# with a fast_mode option.
###############################################################################
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # Load datasets (update paths if needed)
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

    target = "Price"
    features = [col for col in train.columns if col != target]
    cats = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]
    
    pipeline = LRPipeline(train=train, test=test,
                          target=target, features=features, cats=cats)
    # Set tuning_method to 'optuna' or 'randomized' and fast_mode to True for faster execution.
    pipeline.run_pipeline(n_trials=100, n_splits=5, tuning_method='optuna', fast_mode=True)
    # Alternatively, to use RandomizedSearchCV with fast_mode:
    # pipeline.run_pipeline(n_trials=100, n_splits=5, tuning_method='randomized', fast_mode=True)



import pandas as pd
df= pd.read_csv('submission.csv')
df.head(10)

