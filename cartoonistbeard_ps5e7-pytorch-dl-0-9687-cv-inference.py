import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import optuna
from optuna.trial import TrialState

import warnings
warnings.filterwarnings("ignore")


# Configuration
class config:
    train_path = "/kaggle/input/playground-series-s5e7/train.csv"
    test_path = "/kaggle/input/playground-series-s5e7/test.csv"
    sample_sub = "/kaggle/input/playground-series-s5e7/sample_submission.csv"
    
    target = "Personality"
    seed = 42

    # Model & Training
    num_classes = 2
    epochs = 70
    lr = 0.001477411305509788
    batch_size = 64
    input_dim = None  # will be set dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
config = config()
torch.manual_seed(config.seed)

# Data Ingestion
class Dataingestion:
    def __init__(self):
        self.train = pd.read_csv(config.train_path, index_col="id")
        self.test = pd.read_csv(config.test_path, index_col="id")
        self.target = config.target
        display(self.train.head())
        display(self.test.head())

    def get_train_test(self):
        return self.train, self.test

# Preprocessing
class Preprocessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def impute_categorical(self):
        for col in self.test.select_dtypes(include=['object']).columns:
            self.train[col].fillna(self.train[col].mode()[0], inplace=True)
            self.test[col].fillna(self.test[col].mode()[0], inplace=True)

    def impute_numerical(self):
        for col in self.test.select_dtypes(include=['float64', 'int64']).columns:
            self.train[col].fillna(self.train[col].mean(), inplace=True)
            self.test[col].fillna(self.test[col].mean(), inplace=True)

    def encode_categorical(self):
        mapper = {"Yes": 0, "No": 1}
        for col in self.test.select_dtypes(include=['object']).columns:
            self.train[col] = self.train[col].map(mapper)
            self.test[col] = self.test[col].map(mapper)

    def encode_target(self):
        mapper = {"Extrovert": 0, "Introvert": 1}
        self.train[config.target] = self.train[config.target].map(mapper)

    def preprocess(self):
        self.impute_categorical()
        self.impute_numerical()
        self.encode_categorical()
        self.encode_target()
        display(self.train.head())
        display(self.test.head())
        print("Preprocessing complete.")
        return self.train, self.test

# Dataset
class CustomDataset(Dataset):
    def __init__(self, data, split="train"):
        features = data.copy()
        self.split = split
        if split == "train":
            labels = features.pop(config.target)
            self.labels = torch.tensor(labels.values, dtype=torch.long)
        self.features = torch.tensor(features.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.split == "train":
            return self.features[index], self.labels[index]
        else:
            return self.features[index]


def define_model(trial):
    n_layers = trial.suggest_int("n_layers",1,5)
    layers = []
    in_fetr = 7
    for i in range(n_layers):
        out_fetr = trial.suggest_int(f"n_units_{i}",4,128)
        layers.append(nn.Linear(in_fetr,out_fetr))
        layers.append(nn.ReLU())
        p = trial.suggest_float(f"dropout_{i}",0.2,0.5)
        layers.append(nn.Dropout(p))
        in_fetr = out_fetr
    layers.append(nn.Linear(in_fetr,2))
    return nn.Sequential(*layers)

def objective(trial):
    model = define_model(trial).to(config.device)

    # Optimizer selection
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    epochs = trial.suggest_int("epochs", 10, 100, step=10)

    optimizer = getattr(optim, opt_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train.drop(config.target, axis=1), train[config.target])):
        train_df = train.iloc[train_idx].copy()
        val_df = train.iloc[val_idx].copy()

        train_dataset = CustomDataset(train_df, "train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        model.train()
        for epoch in range(epochs):
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(config.device)
                batch_labels = batch_labels.to(config.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        # Validation
        val_features = torch.tensor(val_df.drop(config.target, axis=1).values, dtype=torch.float32).to(config.device)
        val_labels = val_df[config.target].values

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

        acc = accuracy_score(val_labels, val_preds)
        val_scores.append(acc)

    avg_val_acc = np.mean(val_scores)
    return avg_val_acc

                
                
# --- Running the pipeline ---
ingester = Dataingestion()
train, test = ingester.get_train_test()

processor = Preprocessing(train, test)
train, test = processor.preprocess()

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=25)

# print("Best trial:")
# print(study.best_trial)

# print("Best hyperparameters:")
# for key, value in study.best_trial.params.items():
#     print(f"{key}: {value}")


# Model
class NeuralNets(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 51),
            nn.Dropout(0.336682495915063),
            nn.ReLU(),
            nn.Linear(51, 16),
            nn.Dropout(0.2899754225673039),
            nn.ReLU(),
            nn.Linear(16, 80),
            nn.Dropout(0.4363500829739193),
            nn.ReLU(),
            nn.Linear(80, 48),
            nn.Dropout(0.37355415962243),
            nn.Linear(48, config.num_classes)
        )

    def forward(self, inp):
        return self.model(inp)


class Trainer:
    def __init__(self, train_data):
        self.data = train_data.copy()
        config.input_dim = self.data.drop(config.target, axis=1).shape[1]
        self.model = NeuralNets(config.input_dim).to(config.device)
        self.criterion = config.criterion
        self.optim = optim.Adam(self.model.parameters(), lr=config.lr,weight_decay=1e-4)
        self.dataset = CustomDataset(self.data, "train")
        self.loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    def train(self):
        self.model.train()
        self.global_loss = 0
        for epoch in range(config.epochs):
            total_loss = 0
            for batch_features, batch_labels in self.loader:
                batch_features, batch_labels = batch_features.to(config.device), batch_labels.to(config.device)
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.loader)
            self.global_loss += avg_loss

    def predict(self, test_data):
        test_data = test_data.copy()
        test_dataset = CustomDataset(test_data, "test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_features in test_loader:
                batch_features = batch_features.to(config.device)
                preds = self.model(batch_features)
                predicted_classes = torch.argmax(preds, dim=1)
                predictions.extend(predicted_classes.cpu().numpy())

        return predictions

class CrossValidator:
    def __init__(self, full_data, test_data):
        self.full_data = full_data
        self.test_data = test_data
        self.oof = np.zeros(len(full_data))

    def run_kfold(self, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.seed)
        fold_accuracies = []
        test_preds_ensemble = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.full_data.drop(config.target,axis=1),self.full_data[config.target])):
            print(f"\n====== Fold {fold+1} / {k} ======")
            train_df = self.full_data.iloc[train_idx].copy()
            val_df = self.full_data.iloc[val_idx].copy()

            trainer = Trainer(train_df)
            trainer.train()

            train_features = train_df.drop(config.target, axis=1)
            train_labels = train_df[config.target].values
            
            val_features = val_df.drop(config.target, axis=1)
            val_labels = val_df[config.target].values

            val_preds = trainer.predict(val_features)
            train_preds = trainer.predict(train_features)

            val_acc = accuracy_score(val_labels, val_preds)
            train_acc = accuracy_score(train_labels, train_preds)
            print(f"Fold {fold+1} | Train_Accuracy: {train_acc:.4f} | Val_Accuracy: {val_acc:.4f} | Loss: {trainer.global_loss:.4f}")
            fold_accuracies.append(val_acc)

            # Predict on test set
            test_preds = trainer.predict(self.test_data)
            test_preds_ensemble.append(test_preds)

        print("\n=== K-Fold Validation Complete ===")
        print(f"Average Accuracy: {sum(fold_accuracies)/k:.4f}")

        # Ensemble predictions (majority vote)
        test_preds_ensemble = list(zip(*test_preds_ensemble))  # transpose
        final_test_preds = [max(set(row), key=row.count) for row in test_preds_ensemble]

        return final_test_preds

cv = CrossValidator(train, test)
final_predictions = cv.run_kfold(k=5)

sub = pd.read_csv(config.sample_sub)
sub[config.target] = final_predictions
sub[config.target] = sub[config.target].map({0:"Extrovert",1:"Introvert"})
sub.to_csv("submission.csv", index=False)
print("!!!! submission saved !!!!")
display(sub.head())
sub[config.target].value_counts().plot(kind="bar")

