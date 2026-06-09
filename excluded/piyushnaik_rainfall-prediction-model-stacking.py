import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


duplicate_rows = df[df.duplicated()]

print("Number of duplicate rows:", len(duplicate_rows))


null_counts = df.isnull().sum()
null_counts



df['rainfall'].value_counts()




X = df.drop('rainfall', axis=1)
y = df['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)  # You can set a random state for reproducibility
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['rainfall'] = y_resampled

print(resampled_df['rainfall'].value_counts())



class ModelParams(BaseModel):
    objective: str = "binary"
    metric: str = "auc"
    learning_rate: float = 0.02
    num_leaves: int = 31
    max_depth: int = 4
    min_child_samples: int = Field(75, alias="min_child_sample")
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    lambda_l1: int = 5
    lambda_l2: int = 5
    n_estimators: int = 1000
    
# Define Pydantic Model for Training Configuration
class TrainingConfig(BaseModel):
    n_splits: int = 5
    early_stopping_rounds: int = 50

# Define Stacking Model Trainer
class StackingModelTrainer:
    def __init__(self, base_models: List, meta_model, config: TrainingConfig):
        self.base_models = base_models
        self.meta_model = meta_model
        self.config = config
        self.fitted_base_models = []
        self.auc_scores: List[float] = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        kf = KFold(n_splits=self.config.n_splits)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        self.fitted_base_models = [None] * len(self.base_models)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            print(f"\n### Training Fold {fold + 1} ###")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            for i, model in enumerate(self.base_models):
                cloned_model = clone(model)
                cloned_model.fit(X_train, y_train)
                meta_features[valid_idx, i] = cloned_model.predict_proba(X_valid)[:, 1]
                self.fitted_base_models[i] = cloned_model
            
            # Check if y_valid has both classes
            if len(np.unique(y_valid)) < 2:
                print(f"Skipping Fold {fold + 1} due to only one class in validation set.")
                continue
            
            # Compute AUC for this fold
            fold_auc = roc_auc_score(y_valid, meta_features[valid_idx].mean(axis=1))
            self.auc_scores.append(fold_auc)
            print(f'Fold {fold + 1} AUC-ROC: {fold_auc:.6f}')
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
        # Evaluate model
        final_preds = self.meta_model.predict_proba(meta_features)[:, 1]
        overall_auc = roc_auc_score(y, final_preds)
        print(f'\nStacking Model Overall AUC-ROC: {overall_auc:.6f}')
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_base_models:
            raise ValueError("Models are not fitted. Call train() before predict().")
        
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
        
        return self.meta_model.predict_proba(meta_features)[:, 1].reshape(-1, 1)  # Ensure 2D output

# Example Usage
base_models = [
    LGBMClassifier(n_estimators=500, learning_rate=0.05),
    RandomForestClassifier(n_estimators=200, max_depth=6),
    XGBClassifier(n_estimators=300, learning_rate=0.05)
]
meta_model = LogisticRegression()
config = TrainingConfig(n_splits=5, early_stopping_rounds=50)
trainer = StackingModelTrainer(base_models, meta_model, config)

trainer.train(X_resampled, y_resampled)


