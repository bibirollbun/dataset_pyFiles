!pip install rtdl_num_embeddings
!git clone https://github.com/yandex-research/tabm.git

import sys
sys.path.append('/kaggle/working/tabm/tabm_reference')


import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
import torch
from torch import nn
from torch.optim import AdamW
from sklearn.base import BaseEstimator
from tabm.tabm_reference import Model, make_parameter_groups
import rtdl_num_embeddings
from torch.utils.data import TensorDataset, DataLoader
import math
from sklearn.metrics import roc_auc_score, accuracy_score

#############################################
# Base Class for TabM Models (Regressor/Classifer)
#############################################

class TabMBase(BaseEstimator):
    def __init__(self, arch_type='tabm-mini', max_epochs=1000, early_stopping_rounds=16, batch_size=256, learning_rate=2e-3, device=None):
        self.arch_type = arch_type
        self.max_epochs = max_epochs
        self.early_stopping_rounds = early_stopping_rounds  # Patience for early stopping
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    def _preprocess(self, X, y=None, fit=True):
        X = X.copy()
        if fit:
            self.cat_cols = X.select_dtypes('category').columns.tolist()
            self.cont_cols = X.select_dtypes(exclude='category').columns.tolist()
            if self.cat_cols:
                self.encoder = OrdinalEncoder().fit(X[self.cat_cols])
            self.qt = QuantileTransformer(output_distribution='normal').fit(X[self.cont_cols])
        
        if self.cat_cols:
            X_cat = self.encoder.transform(X[self.cat_cols])
        else:
            X_cat = None
        X_cont = self.qt.transform(X[self.cont_cols])
        
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device) if X_cat is not None else None
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)
        
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        return X_cat, X_cont, y

#########################
# TabMRegressor Class   #
#########################

class TabMRegressor(TabMBase):
    def __init__(self, metric="mse", **kwargs):
        """
        metric: Evaluation metric for early stopping. Options include "rmse" or "mae".
        """
        super().__init__(**kwargs)
        self.metric = metric.lower()

    def fit(self, X, y, eval_set):
        X_cat, X_cont, y = self._preprocess(X, y, fit=True)
        X_cat_val, X_cont_val, y_val = self._preprocess(*eval_set, fit=False)
        
        cat_cardinalities = [X[col].nunique() for col in self.cat_cols] if self.cat_cols else []
        
        self.model = Model(
            n_num_features=X_cont.shape[1],
            bins=None,
            cat_cardinalities=cat_cardinalities,
            n_classes=None,
            backbone={'type': 'MLP', 'n_blocks': 3, 'd_block': 512, 'dropout': 0.1},
            arch_type=self.arch_type,
            k=32
        ).to(self.device)
        
        optimizer = AdamW(make_parameter_groups(self.model), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        if X_cat is None:
            train_dataset = TensorDataset(X_cont, y)
            def collate_fn(batch):
                X_cont_batch, y_batch = zip(*batch)
                return torch.stack(X_cont_batch), None, torch.stack(y_batch)
        else:
            train_dataset = TensorDataset(X_cont, X_cat, y)
            collate_fn = None
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
        best_metric = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch_X_cont, batch_X_cat, batch_y = batch
                optimizer.zero_grad()
                # Get raw output from the model (shape: [batch_size, k])
                output = self.model(batch_X_cont, batch_X_cat)
                # Average the ensemble outputs along dimension 1 to get shape [batch_size]
                preds = output.mean(dim=1)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_cont_val, X_cat_val)
                val_preds = val_output.mean(dim=1)
                # Then compute validation metric, e.g., RMSE or MAE:
                if self.metric == "mae":
                    val_metric = nn.functional.l1_loss(val_preds, y_val).item()
                elif self.metric == "rmse":
                    mse = nn.functional.mse_loss(val_preds, y_val).item()
                    val_metric = math.sqrt(mse)
                else:
                    val_metric = nn.functional.mse_loss(val_preds, y_val).item()  # default to MSE
            
            if val_metric < best_metric:
                best_metric = val_metric
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
            
            print(f"Epoch {epoch}: Train loss {np.mean(train_losses):.4f}, Val {self.metric} {val_metric:.4f}")
            if epochs_without_improvement >= self.early_stopping_rounds:
                print("Early stopping triggered.")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        self.model.eval()
        X_cat, X_cont, _ = self._preprocess(X, fit=False)
        with torch.no_grad():
            preds = self.model(X_cont, X_cat).squeeze()
        return preds.cpu().numpy()

#########################
# TabMClassifier Class  #
#########################

class TabMClassifier(TabMBase):
    def __init__(self, eval_metrics=["accuracy", "auc"], early_stopping_metric="auc", **kwargs):
        """
        eval_metrics: List of evaluation metric names. Supported: "accuracy", "auc", "logloss".
        early_stopping_metric: Primary metric used for early stopping.
        """
        super().__init__(**kwargs)
        self.eval_metrics = [m.lower() for m in eval_metrics]
        self.early_stopping_metric = early_stopping_metric.lower()

    def fit(self, X, y, eval_set):
        X_cat, X_cont, y = self._preprocess(X, y, fit=True)
        y = y.to(torch.long)  
        X_cat_val, X_cont_val, y_val = self._preprocess(*eval_set, fit=False)
        y_val = y_val.to(torch.long)
        
        cat_cardinalities = [X[col].nunique() for col in self.cat_cols] if self.cat_cols else []
        n_classes = int(len(np.unique(y.cpu().numpy())))
        
        self.model = Model(
            n_num_features=X_cont.shape[1],
            bins=None,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            backbone={'type': 'MLP', 'n_blocks': 3, 'd_block': 512, 'dropout': 0.1},
            arch_type=self.arch_type,
            k=32
        ).to(self.device)
        
        optimizer = AdamW(make_parameter_groups(self.model), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        if X_cat is None:
            train_dataset = TensorDataset(X_cont, y)
            def collate_fn(batch):
                X_cont_batch, y_batch = zip(*batch)
                return torch.stack(X_cont_batch), None, torch.stack(y_batch)
        else:
            train_dataset = TensorDataset(X_cont, X_cat, y)
            collate_fn = None
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
        best_metric = -float('inf')  # For classification, higher is better
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch_X_cont, batch_X_cat, batch_y = batch
                optimizer.zero_grad()
                logits = self.model(batch_X_cont, batch_X_cat)
    
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_cont_val, X_cat_val)
                if len(val_logits.shape) == 3:
                    avg_logits = val_logits.mean(dim=1)
                else:
                    avg_logits = val_logits
                
                probs = nn.functional.softmax(avg_logits, dim=1)
                preds = avg_logits.argmax(dim=1)
                val_accuracy = accuracy_score(y_val.cpu().numpy(), preds.cpu().numpy())
                try:
                    if n_classes == 2:
                        val_auc = roc_auc_score(y_val.cpu().numpy(), probs[:, 1].cpu().numpy())
                    else:
                        val_auc = roc_auc_score(y_val.cpu().numpy(), probs.cpu().numpy(), multi_class='ovr')
                except Exception as e:
                    val_auc = 0.0
                val_logloss = criterion(avg_logits, y_val).item()
                metrics = {"accuracy": val_accuracy, "auc": val_auc, "logloss": val_logloss}
                current_metric = metrics.get(self.early_stopping_metric, 0.0)
            
            if current_metric > best_metric:
                best_metric = current_metric
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
            
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch}: Train loss {np.mean(train_losses):.4f}, Val metrics: {metrics_str} (Best {self.early_stopping_metric}: {best_metric:.4f})")
            
            if epochs_without_improvement >= self.early_stopping_rounds:
                print("Early stopping triggered.")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict_proba(self, X):
        self.model.eval()
        X_cat, X_cont, _ = self._preprocess(X, fit=False)
        with torch.no_grad():
            logits = self.model(X_cont, X_cat)
            if len(logits.shape) == 3:
                avg_logits = logits.mean(dim=1)
            else:
                avg_logits = logits
            probs = nn.functional.softmax(avg_logits, dim=1)
        return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train['year_group'] = train['id']//365 

RMV = ['rainfall','id','bucket','year_group']
FEATURES = [c for c in list( train.columns ) if not c in RMV]


train_imputed = train[FEATURES].interpolate(method='linear', limit_direction='both')
test_imputed = test[FEATURES].interpolate(method='linear', limit_direction='both')

train[FEATURES] = train_imputed
test = pd.DataFrame(test_imputed, columns=FEATURES)


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


X = train[FEATURES].copy()
y = train["rainfall"].copy()
X_test = test[FEATURES].copy()


FOLDS = train.year_group.nunique()
gkf = GroupKFold(n_splits=FOLDS)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(X_test), FOLDS))

fold = 1
for train_idx, val_idx in gkf.split(X, y, groups=train.year_group):
    val_groups = train.year_group.iloc[val_idx].unique()
    print(f"Training fold {fold} ...")
    print(f"Validation groups for fold {fold}: {val_groups}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = TabMClassifier(
        max_epochs=100,              
        early_stopping_rounds=10,     
        batch_size=256,
        learning_rate=2e-3,
        device='cpu'
    )

    model.fit(X_train, y_train.values, eval_set=(X_val, y_val.values))

    # Predict probabilities and extract the positive class probability
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}\n")

    test_preds[:, fold - 1] = model.predict_proba(X_test)[:, 1]

    fold += 1

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

final_test_preds = test_preds.mean(axis=1)
print("\nFinal test predictions (first 10 samples):")
print(final_test_preds[:10])


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub.rainfall = final_test_preds
sub.to_csv('submission.csv', index=False)
sub.head()




