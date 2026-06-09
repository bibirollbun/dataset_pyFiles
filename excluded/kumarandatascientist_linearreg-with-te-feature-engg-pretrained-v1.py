import pandas as pd
import numpy as np
import time
import warnings
import random
from tqdm.auto import tqdm
from cuml.preprocessing import TargetEncoder  # RAPIDS cuML target encoder

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn imports for cross-validation and metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

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
# PyTorch Linear Regression Model Definition
###############################################################################
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


###############################################################################
# LRPipeline Class Definition with Enhanced Feature Engineering,
# K-Fold CV (with OOF validation), Final Model Tuning via Grid Search,
# Reproducibility, and tqdm callbacks.
###############################################################################
class LRPipeline:
    """
    A pipeline for preprocessing, k-fold training with out-of-fold validation,
    final model tuning on the full data (using a grid search over hyperparameters),
    saving/loading a target-encoded linear regression model using PyTorch,
    and generating test predictions.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 random_state: int = 42):
        """
        Parameters:
            train (DataFrame): Training dataset.
            test (DataFrame): Test dataset.
            target (str): Name of the target variable.
            features (list): List of feature names to use.
            cats (list): List of categorical feature names (for target encoding).
            te_params (dict): Parameters for the TargetEncoder (optional).
            random_state (int): Random seed.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features
        self.cats = cats  # categorical columns for target encoding
        self.random_state = random_state

        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params

        self.TE = TargetEncoder(**self.te_params)

        self.all_features = None
        self.metrics_log = []
        self.model = None

        self.best_params = {
            "num_epochs": 100,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
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
        # Apply target encoding to categorical columns.
        for col in self.cats:
            self.train[f"TE_{col}"] = self.TE.fit_transform(self.train[col], self.train[self.target])
            self.test[f"TE_{col}"] = self.TE.transform(self.test[col])
        self.train[self.cats] = self.train[self.cats].fillna('Missing').astype('category')
        self.test[self.cats] = self.test[self.cats].fillna('Missing').astype('category')
        # Identify numeric features (features not in cats)
        num_cols = [col for col in self.features if col not in self.cats]
        self.train[num_cols] = self.train[num_cols].fillna(self.train[num_cols].mean())
        self.test[num_cols] = self.test[num_cols].fillna(self.train[num_cols].mean())
        # Normalize numeric features with min-max scaling.
        for col in num_cols:
            min_val = self.train[col].min()
            max_val = self.train[col].max()
            if max_val - min_val == 0:
                self.train[col] = 0.0
                self.test[col] = 0.0
            else:
                self.train[col] = (self.train[col] - min_val) / (max_val - min_val)
                self.test[col] = (self.test[col] - min_val) / (max_val - min_val)
        # Log transformation on "Weight Capacity (kg)" if present.
        if "Weight Capacity (kg)" in num_cols:
            self.train["Weight Capacity (kg)"] = self.train["Weight Capacity (kg)"].clip(lower=0)
            self.test["Weight Capacity (kg)"] = self.test["Weight Capacity (kg)"].clip(lower=0)
            self.train["log_Weight_Capacity"] = np.log1p(self.train["Weight Capacity (kg)"])
            self.test["log_Weight_Capacity"] = np.log1p(self.test["Weight Capacity (kg)"])
            self.log("Created log-transformed Weight Capacity feature: log_Weight_Capacity")
        else:
            self.log("Weight Capacity (kg) not found among numeric features.")
        # Create aggregated TE features.
        te_cols = [f"TE_{col}" for col in self.cats]
        self.train["sum_TE"] = self.train[te_cols].sum(axis=1)
        self.train["mean_TE"] = self.train[te_cols].mean(axis=1)
        self.train["std_TE"] = self.train[te_cols].std(axis=1)
        self.test["sum_TE"] = self.test[te_cols].sum(axis=1)
        self.test["mean_TE"] = self.test[te_cols].mean(axis=1)
        self.test["std_TE"] = self.test[te_cols].std(axis=1)
        self.log("Created aggregated TE features: sum_TE, mean_TE, std_TE")
        # Final feature set.
        engineered_features = []
        if "Weight Capacity (kg)" in num_cols:
            engineered_features.append("log_Weight_Capacity")
        engineered_features += ["sum_TE", "mean_TE", "std_TE"]
        self.all_features = num_cols + te_cols + engineered_features
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. Features created: {self.all_features} (Time taken: {elapsed:.2f} sec)")

    def hyperparameter_tuning(self, n_trials: int = 20):
        start_time = time.time()
        self.log("Skipping hyperparameter tuning for linear regression. Using default parameters:")
        self.log(str(self.best_params))
        elapsed = time.time() - start_time
        self.log(f"Hyperparameter tuning time: {elapsed:.2f} sec")

    def train_kfold_model(self, n_splits: int = 5):
        overall_start = time.time()
        self.log("Starting k-fold training with out-of-fold validation...")
        X = self.train[self.all_features].values.astype(np.float32)
        y = self.train[self.target].values.astype(np.float32).reshape(-1, 1)
        oof_preds = np.zeros_like(y)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_rmse_list = []
        fold_iter = tqdm(kf.split(X), total=n_splits, desc="Folds")
        for fold, (train_idx, val_idx) in enumerate(fold_iter, 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_tensor = torch.from_numpy(X_train).to(device)
            y_train_tensor = torch.from_numpy(y_train).to(device)
            X_val_tensor = torch.from_numpy(X_val).to(device)
            y_val_tensor = torch.from_numpy(y_val).to(device)
            input_dim = len(self.all_features)
            model = LinearRegressionModel(input_dim=input_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.best_params["learning_rate"],
                                   weight_decay=self.best_params["weight_decay"])
            num_epochs = self.best_params["num_epochs"]
            early_stopping_rounds = self.best_params["early_stopping_rounds"]
            best_val_loss = np.inf
            epochs_without_improve = 0
            best_state_dict = None
            epoch_iter = tqdm(range(num_epochs), desc=f"Fold {fold} Epochs", leave=False)
            for epoch in epoch_iter:
                model.train()
                optimizer.zero_grad()
                predictions = model(X_train_tensor)
                loss = criterion(predictions, y_train_tensor)
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()
                rmse_epoch = np.sqrt(loss.item())
                epoch_iter.set_postfix({"Train RMSE": f"{rmse_epoch:.4f}", "Val Loss": f"{val_loss:.4f}"})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = model.state_dict()
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                if epochs_without_improve >= early_stopping_rounds:
                    self.log(f"Fold {fold}: Early stopping triggered at epoch {epoch+1}")
                    break
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
            model.eval()
            with torch.no_grad():
                fold_preds = model(X_val_tensor).cpu().numpy()
            oof_preds[val_idx] = fold_preds
            fold_rmse = np.sqrt(best_val_loss)
            fold_rmse_list.append(fold_rmse)
            self.log(f"Fold {fold}: Best Validation RMSE: {fold_rmse:.4f}")
        overall_mse = mean_squared_error(y, oof_preds)
        overall_rmse = np.sqrt(overall_mse)
        self.log(f"Overall Out-of-Fold RMSE: {overall_rmse:.4f}")
        overall_elapsed = time.time() - overall_start
        self.log(f"K-Fold training complete. Total time: {overall_elapsed:.2f} sec")
        self.oof_rmse = overall_rmse
        self.oof_preds = oof_preds

    def tune_final_model_full(self, validation_split: float = 0.2):
        """
        Perform a simple grid search over a set of hyperparameters (e.g. learning rate and number of epochs)
        using a hold-out validation split from the full training data. This aims to lower the final RMSE.
        """
        self.log("Starting final model tuning on full training data...")
        # Split full training data into train and validation sets.
        X_full = self.train[self.all_features].values.astype(np.float32)
        y_full = self.train[self.target].values.astype(np.float32).reshape(-1, 1)
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_full, y_full, test_size=validation_split, random_state=self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.from_numpy(X_train_full).to(device)
        y_train_tensor = torch.from_numpy(y_train_full).to(device)
        X_val_tensor = torch.from_numpy(X_val).to(device)
        y_val_tensor = torch.from_numpy(y_val).to(device)
        input_dim = len(self.all_features)
        criterion = nn.MSELoss()
        
        # Define grid search candidates
        lr_list = [0.005, 0.01, 0.02]  # candidate learning rates
        epoch_list = [100, 150, 200]    # candidate max epochs
        
        best_val_loss = np.inf
        best_lr = None
        best_epochs = None
        best_model_state = None
        
        for lr in lr_list:
            for num_epochs in epoch_list:
                model = LinearRegressionModel(input_dim=input_dim).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.best_params["weight_decay"])
                early_stopping_rounds = self.best_params["early_stopping_rounds"]
                current_best_val = np.inf
                epochs_without_improve = 0
                # Use tqdm to show progress for each grid point.
                grid_iter = tqdm(range(num_epochs), desc=f"LR {lr}, Epochs {num_epochs}", leave=False)
                for epoch in grid_iter:
                    model.train()
                    optimizer.zero_grad()
                    predictions = model(X_train_tensor)
                    loss = criterion(predictions, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        val_predictions = model(X_val_tensor)
                        val_loss = criterion(val_predictions, y_val_tensor).item()
                    grid_iter.set_postfix({"Val Loss": f"{val_loss:.4f}"})
                    if val_loss < current_best_val:
                        current_best_val = val_loss
                        best_state = model.state_dict()
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1
                    if epochs_without_improve >= early_stopping_rounds:
                        break
                self.log(f"Grid search: LR={lr}, Epochs={num_epochs}, Best Val Loss={current_best_val:.4f}")
                if current_best_val < best_val_loss:
                    best_val_loss = current_best_val
                    best_lr = lr
                    best_epochs = num_epochs
                    best_model_state = best_state
        
        self.log(f"Tuning complete. Best LR: {best_lr}, Best Epochs: {best_epochs}, Best Val Loss: {best_val_loss:.4f}")
        # Retrain final model on full training data using the best hyperparameters.
        X_full_tensor = torch.from_numpy(X_full).to(device)
        y_full_tensor = torch.from_numpy(y_full).to(device)
        final_model = LinearRegressionModel(input_dim=input_dim).to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=self.best_params["weight_decay"])
        final_num_epochs = best_epochs  # use the best epoch count found
        final_iter = tqdm(range(final_num_epochs), desc="Final Model Retraining", unit="epoch")
        for epoch in final_iter:
            final_model.train()
            optimizer.zero_grad()
            predictions = final_model(X_full_tensor)
            loss = criterion(predictions, y_full_tensor)
            loss.backward()
            optimizer.step()
            final_iter.set_postfix({"Loss RMSE": f"{np.sqrt(loss.item()):.4f}"})
        self.model = final_model
        self.log("Final model tuning complete.")

    def save_pretrained_model(self, filename="backpack_lr_te.pth"):
        start_time = time.time()
        torch.save(self.model.state_dict(), filename)
        elapsed = time.time() - start_time
        self.log(f"Pretrained model saved to {filename} (Time taken: {elapsed:.2f} sec)")

    def load_pretrained_model(self, filename="backpack_lr_te.pth"):
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = len(self.all_features)
        model = LinearRegressionModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(filename, map_location=device))
        self.model = model
        elapsed = time.time() - start_time
        self.log(f"Pretrained model loaded from {filename} (Time taken: {elapsed:.2f} sec)")

    def predict_test(self):
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_np = self.test[self.all_features].values.astype(np.float32)
        X_test_tensor = torch.from_numpy(X_test_np).to(device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test_tensor).cpu().numpy().flatten()
        elapsed = time.time() - start_time
        self.log(f"Test predictions complete. (Time taken: {elapsed:.2f} sec)")
        return preds

    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename} (Time taken: {elapsed:.2f} sec)")

    def run_pipeline(self, n_trials: int = 20, n_splits: int = 5):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning", lambda: self.hyperparameter_tuning(n_trials)),
            ("K-Fold Training", lambda: self.train_kfold_model(n_splits))
        ]
        for step_name, step_func in steps:
            self.log(f"----- Starting step: {step_name} -----")
            step_func()
            self.log(f"----- Completed step: {step_name} -----")
        
        # Instead of directly training on full data, now perform additional tuning.
        self.log("----- Tuning Final Model on Full Data -----")
        self.tune_final_model_full(validation_split=0.2)
        
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
# TE & Linear Regression Pipeline Usage with Enhanced Feature Engineering,
# K-Fold CV, Reproducibility, and tqdm callbacks
###############################################################################
if __name__ == "__main__":
    # Set reproducibility seeds (if not already set above)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # Load the datasets.
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

    # Define target and feature columns.
    target = "Price"
    features = [col for col in train.columns if col != target]
    
    # Define categorical columns (for example, all columns except Price and Weight Capacity)
    cats = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]
    
    # Initialize and run the pipeline.
    pipeline = LRPipeline(train=train, test=test,
                          target=target, features=features, cats=cats)
    # For example, use 100 trials (n_trials) and 15 splits for KFold.
    pipeline.run_pipeline(n_trials=100, n_splits=15)



import pandas as pd
df= pd.read_csv('submission.csv')
df.head(10)




