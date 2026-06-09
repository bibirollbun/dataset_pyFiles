


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import category_encoders as ce


class FinalTabularMLPipeline:
    """Final pipeline with proper ID handling for submission"""
    def __init__(self, target_column, random_state=42):
        self.target_column = target_column
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.cat_encoder = ce.TargetEncoder(cols=['Geography', 'Gender'])  # Specify categorical columns
        self.best_threshold = 0.5

    def load_and_prepare_data(self, train_path, test_path=None, submission_path=None):
        self.train = pd.read_csv(train_path)
        if test_path:
            self.test = pd.read_csv(test_path)
            self.test_ids = self.test['id']  # Store test IDs separately
        if submission_path:
            self.submission_sample = pd.read_csv(submission_path)
        
        # Identify feature columns
        self.categorical = ['Geography', 'Gender']  # Explicitly specify
        self.numerical = [col for col in self.train.columns 
                         if col not in self.categorical + [self.target_column, 'id', 'Surname']]
        
        print(f"Categorical features: {self.categorical}")
        print(f"Numerical features: {self.numerical}")

    def preprocess_features(self):
        # Encode target
        y = self.label_encoder.fit_transform(self.train[self.target_column])
        
        # Prepare features
        X = self.train[self.numerical + self.categorical].copy()
        if hasattr(self, 'test'):
            X_test = self.test[self.numerical + self.categorical].copy()
        
        # Target encode categoricals
        self.cat_encoder.fit(X[self.categorical], y)
        X[self.categorical] = self.cat_encoder.transform(X[self.categorical])
        if hasattr(self, 'test'):
            X_test[self.categorical] = self.cat_encoder.transform(X_test[self.categorical])
        
        self.X = X
        self.y = y
        if hasattr(self, 'test'):
            self.X_test = X_test

    def train_model(self, params=None, n_splits=5, n_repeats=2, num_boost_round=200):
        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "tree_method": "hist"
            }
        
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, 
                                    random_state=self.random_state)
        self.oof_preds = np.zeros(len(self.X))
        if hasattr(self, 'X_test'):
            self.test_preds = np.zeros(len(self.X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params, dtrain, num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=0
            )
            
            self.oof_preds[val_idx] += model.predict(dval) / n_repeats
            if hasattr(self, 'X_test'):
                dtest = xgb.DMatrix(self.X_test)
                self.test_preds += model.predict(dtest) / (n_repeats * n_splits)
            
            if (fold + 1) % 2 == 0:
                print(f"Completed fold {fold + 1}")
        
        self.final_model = model

    def create_submission(self, filename='submission.csv'):
        if not hasattr(self, 'test_preds'):
            print("No test predictions available")
            return None
            
        # Create submission with ID column
        submission = pd.DataFrame({
            'id': self.test_ids,
            'Exited': self.test_preds
        })
        
        # Ensure same order as sample submission
        if hasattr(self, 'submission_sample'):
            submission = submission[self.submission_sample.columns]
        
        submission.to_csv(filename, index=False)
        print(f"Submission file created with {len(submission)} rows")
        print(f"ID column range: {submission['id'].min()} to {submission['id'].max()}")
        return submission


# Execute the final pipeline
pipeline = FinalTabularMLPipeline(target_column='Exited', random_state=42)
pipeline.load_and_prepare_data(
    train_path="/kaggle/input/playground-series-s4e1/train.csv",
    test_path="/kaggle/input/playground-series-s4e1/test.csv",
    submission_path='/kaggle/input/playground-series-s4e1/sample_submission.csv'
)
pipeline.preprocess_features()
pipeline.train_model()
submission = pipeline.create_submission('submission.csv')




