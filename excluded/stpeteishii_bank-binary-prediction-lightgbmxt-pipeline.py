


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import category_encoders as ce

class FinalTabularMLPipeline:
    def __init__(self, target_column='stroke', random_state=42, max_categories=20):
        self.target_column = target_column
        self.random_state = random_state
        self.max_categories = max_categories
        self.label_encoder = LabelEncoder()
        self.cat_encoder = None
        self._initialize_placeholders()

    def _initialize_placeholders(self):
        self.train = None
        self.test = None
        self.test_ids = None
        self.submission_sample = None
        self.X = None
        self.y = None
        self.X_test = None
        self.oof_preds = None
        self.test_preds = None
        self.final_model = None
        self.categorical = []
        self.numerical = []

    def _auto_detect_features(self, df):
        self.categorical = []
        self.numerical = []
        
        for col in df.columns:
            if col == self.target_column or col == 'id':
                continue
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                self.categorical.append(col)
            elif df[col].nunique() <= self.max_categories:
                self.categorical.append(col)
            else:
                self.numerical.append(col)
        
        print(f"Auto-detected categorical: {self.categorical}")
        print(f"Auto-detected numerical: {self.numerical}")

    def load_and_prepare_data(self, train_path, test_path=None, submission_path=None):
        self.train = pd.read_csv(train_path)
        
        if test_path:
            self.test = pd.read_csv(test_path)
            self.test_ids = self.test['id']
        
        if submission_path:
            self.submission_sample = pd.read_csv(submission_path)
        
        self._auto_detect_features(self.train)
        self.cat_encoder = ce.TargetEncoder(cols=self.categorical)

    def preprocess_features(self):
        y = self.label_encoder.fit_transform(self.train[self.target_column])
        X = self.train[self.numerical + self.categorical].copy()
        if hasattr(self, 'test'):
            X_test = self.test[self.numerical + self.categorical].copy()
        
        self.cat_encoder.fit(X[self.categorical], y)
        X[self.categorical] = self.cat_encoder.transform(X[self.categorical])
        if hasattr(self, 'test'):
            X_test[self.categorical] = self.cat_encoder.transform(X_test[self.categorical])
        
        self.X = X
        self.y = y
        if hasattr(self, 'test'):
            self.X_test = X_test

    def train_model(self, params=None, n_splits=5, n_repeats=2, num_boost_round=1000):
        if params is None:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "extra_trees": True,   
                "verbosity": -1
            }
        
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=self.random_state)
        self.oof_preds = np.zeros(len(self.X))
        if hasattr(self, 'X_test'):
            self.test_preds = np.zeros(len(self.X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.log_evaluation(period=0)  
                ]
            )
            
            self.oof_preds[val_idx] += model.predict(X_val, num_iteration=model.best_iteration) / n_repeats
            if hasattr(self, 'X_test'):
                self.test_preds += model.predict(self.X_test, num_iteration=model.best_iteration) / (n_repeats * n_splits)
            
            if (fold + 1) % 2 == 0:
                print(f"Completed fold {fold + 1}")
        
        self.final_model = model

    def create_submission(self, filename='submission.csv'):
        if not hasattr(self, 'test_preds'):
            print("No test predictions available")
            return None
            
        submission = pd.DataFrame({
            'id': self.test_ids,
            'y': self.test_preds
        })
        
        if hasattr(self, 'submission_sample'):
            submission = submission[self.submission_sample.columns]
        
        submission.to_csv(filename, index=False)
        print(f"Submission file created with {len(submission)} rows")
        print(f"ID column range: {submission['id'].min()} to {submission['id'].max()}")
        return submission


pipeline = FinalTabularMLPipeline(target_column='y', random_state=42)
pipeline.load_and_prepare_data(
    train_path="/kaggle/input/playground-series-s5e8/train.csv",
    test_path="/kaggle/input/playground-series-s5e8/test.csv",
    submission_path='/kaggle/input/playground-series-s5e8/sample_submission.csv'
)
pipeline.preprocess_features()
pipeline.train_model()
submission = pipeline.create_submission('submission.csv')
display(submission)





