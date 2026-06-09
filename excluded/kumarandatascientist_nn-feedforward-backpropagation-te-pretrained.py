import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from tqdm.auto import tqdm
from cuml.preprocessing import TargetEncoder  # GPU-accelerated target encoding

warnings.simplefilter('ignore')

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------ Neural Network Definition --------------------------
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        """
        Build a feedforward network with specified hidden layer sizes and dropout.
        """
        super(FeedforwardNN, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        # Final output layer (regression output)
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ----------------------- Neural Network Pipeline Class -----------------------
class NNPipeline:
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        features: list,
        cats: list,
        sample_frac: float = 0.5,
        random_state: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fast_mode: bool = False
    ):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.features = features.copy()  # make a copy so we can append new features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state
        self.device = device
        self.fast_mode = fast_mode  # new flag to speed up execution

        self.best_params = None
        self.best_cv_rmse = None
        self.model = None
        self.all_features = None
        self.metrics_log = []
        self.model_path = "best_nn_model.pth"  # path to save the pretrained model

    def log(self, message: str):
        print(message)
        self.metrics_log.append(message)

    def save_metrics_log(self, filename="nn_model_metrics_log.txt"):
        with open(filename, "w") as f:
            for message in self.metrics_log:
                f.write(message + "\n")
        self.log(f"Metrics log saved to {filename}.")

    # ------------------------- Data Preprocessing -------------------------
    def preprocess_data(self):
        start_time = time.time()
        self.log("Starting data preprocessing...")

        # ------------- Target Encoding -------------
        # If "Weight Capacity (kg)" exists, apply target encoding.
        if "Weight Capacity (kg)" in self.train.columns:
            self.log("Applying target encoding to Weight Capacity (kg)...")
            te_feature = "TE_Weight Capacity (kg)"
            TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
            self.train[te_feature] = TE.fit_transform(self.train["Weight Capacity (kg)"], self.train[self.target])
            self.test[te_feature] = TE.transform(self.test["Weight Capacity (kg)"])
            s = np.sqrt(np.mean((self.train[self.target] - self.train[te_feature])**2.0))
            self.log(f"Validation RMSE using Target Encode Weight Capacity = {s:.4f}")
            if te_feature not in self.features:
                self.features.append(te_feature)

        # ------------- Categorical Processing -------------
        # Fill missing and convert to codes.
        for col in self.cats:
            self.train[col] = self.train[col].fillna("Missing").astype("category").cat.codes
            self.test[col] = self.test[col].fillna("Missing").astype("category").cat.codes

        # ------------- Numerical Processing -------------
        num_cols = [col for col in self.features if col not in self.cats]
        for col in num_cols:
            median_val = self.train[col].median()
            self.train[col] = self.train[col].fillna(median_val)
            self.test[col] = self.test[col].fillna(median_val)

        # ------------- Scaling -------------
        scaler = StandardScaler()
        self.train[self.features] = scaler.fit_transform(self.train[self.features])
        self.test[self.features] = scaler.transform(self.test[self.features])
        self.scaler = scaler

        self.all_features = self.features
        elapsed = time.time() - start_time
        self.log(f"Preprocessing complete. (Time taken: {elapsed:.2f} sec)")

    # --------------------- Utility: DataLoader Creation ---------------------
    def create_dataloader(self, X, y=None, batch_size: int = 32, shuffle: bool = True):
        if y is not None:
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            )
        else:
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # --------------- Optuna Objective for Hyperparameter Tuning ---------------
    def objective(self, trial):
        # Use fast_mode settings if enabled.
        if self.fast_mode:
            epoch_low, epoch_high = 5, 10
            hidden_low, hidden_high = 8, 32
            batch_options = [128, 256]
        else:
            epoch_low, epoch_high = 10, 30
            hidden_low, hidden_high = 16, 128
            batch_options = [32, 64, 128]
            
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", hidden_low, hidden_high)
            hidden_dims.append(hidden_dim)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_options)
        epochs = trial.suggest_int("epochs", epoch_low, epoch_high)

        # Subsample the training data for faster tuning.
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
        X = train_sample[self.all_features].values
        y = train_sample[self.target].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        train_loader = self.create_dataloader(X_train, y_train, batch_size)
        val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)

        input_dim = X_train.shape[1]
        model = FeedforwardNN(input_dim, hidden_dims, dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = np.inf
        for epoch in tqdm(range(epochs), desc="Optuna Trial Epochs", leave=False):
            model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            best_val_loss = min(best_val_loss, avg_val_loss)
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.sqrt(best_val_loss)

    def hyperparameter_tuning(self, n_trials: int = 20):
        self.log("Starting hyperparameter tuning with Optuna for Neural Network...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.best_cv_rmse = study.best_trial.value
        self.log(f"Hyperparameter tuning complete. Best params: {self.best_params}")
        self.log(f"Best CV RMSE: {self.best_cv_rmse:.4f}")

    # ------------------ Training Final Model and Saving ------------------
    def train_final_model(self):
        start_time = time.time()
        self.log("Training final Neural Network model...")

        best = self.best_params
        n_layers = best["n_layers"]
        hidden_dims = [best[f"hidden_dim_{i}"] for i in range(n_layers)]
        dropout = best["dropout"]
        learning_rate = best["learning_rate"]
        weight_decay = best["weight_decay"]
        batch_size = best["batch_size"]
        epochs = best["epochs"]

        X = self.train[self.all_features].values
        y = self.train[self.target].values
        train_loader = self.create_dataloader(X, y, batch_size)

        input_dim = X.shape[1]
        self.model = FeedforwardNN(input_dim, hidden_dims, dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_loss = np.inf
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()
            epoch_losses = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            self.log(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_path)
                self.log(f"--> Best model updated (loss: {best_loss:.4f}).")
        elapsed = time.time() - start_time
        self.log(f"Final model training complete. (Time taken: {elapsed:.2f} sec)")

    # --------------- Load Pretrained Model for Inference ----------------
    def load_pretrained_model(self):
        self.log("Loading pretrained model from disk...")
        best = self.best_params
        n_layers = best["n_layers"]
        hidden_dims = [best[f"hidden_dim_{i}"] for i in range(n_layers)]
        dropout = best["dropout"]
        input_dim = self.train[self.all_features].shape[1]
        self.model = FeedforwardNN(input_dim, hidden_dims, dropout).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.log("Pretrained model loaded successfully.")

    # -------------------- Predict on Test Set --------------------
    def predict_test(self):
        start_time = time.time()
        self.log("Predicting on test data...")
        X_test = self.test[self.all_features].values
        test_loader = self.create_dataloader(X_test, y=None, batch_size=128, shuffle=False)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                batch_X = batch[0].to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
        elapsed = time.time() - start_time
        self.log(f"Test prediction complete. (Time taken: {elapsed:.2f} sec)")
        return predictions

    # -------------------- Save Submission File --------------------
    def save_submission(self, predictions, filename="submission.csv"):
        start_time = time.time()
        sub = pd.DataFrame({"id": self.test.index, self.target: predictions})
        sub.to_csv(filename, index=False)
        elapsed = time.time() - start_time
        self.log(f"Submission saved to {filename}. (Time taken: {elapsed:.2f} sec)")

    # --------------------- Run Full Pipeline ---------------------
    def run_pipeline(self, n_trials: int = 20):
        overall_start = time.time()
        steps = [
            ("Preprocessing Data", self.preprocess_data),
            ("Hyperparameter Tuning", lambda: self.hyperparameter_tuning(n_trials)),
            ("Training Final Model", self.train_final_model),
        ]
        for step_name, step_func in steps:
            self.log(f"----- Starting step: {step_name} -----")
            step_func()
            self.log(f"----- Completed step: {step_name} -----")
        self.load_pretrained_model()
        predictions = self.predict_test()
        self.save_submission(predictions)
        self.save_metrics_log()
        overall_elapsed = time.time() - overall_start
        self.log(f"Pipeline execution complete. Total time: {overall_elapsed:.2f} sec")





###############################################################################
# Pipeline Usage with FeedforwardNN, Hyperparameter Tuning via Optuna,
# and Fast Mode Options
###############################################################################
if __name__ == "__main__":
    # Reproducibility
    set_seed(42)

    # Load datasets (adjust paths if needed)
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

    target = "Price"
    features = [col for col in train.columns if col != target]
    cats = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]
    
    # For faster execution, you can reduce sample_frac (e.g. 0.2) and set fast_mode=True.
    pipeline = NNPipeline(train=train, test=test,
                          target=target, features=features, cats=cats,
                          sample_frac=0.2, random_state=42, fast_mode=True)
    # Use a reduced number of trials for fast mode tuning.
    pipeline.run_pipeline(n_trials=20)



import pandas as pd

df= pd.read_csv('submission.csv')
df.head(10)




