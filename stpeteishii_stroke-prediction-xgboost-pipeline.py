


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import category_encoders as ce


class FinalTabularMLPipeline:
    """Complete stroke prediction pipeline with all fixes"""
    
    def __init__(self, target_column='stroke', random_state=42):
        # Initialize all components
        self.target_column = target_column
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.cat_encoder = ce.TargetEncoder(
            cols=['gender', 'ever_married', 'work_type', 
                 'Residence_type', 'smoking_status']
        )
        
        # Initialize all data placeholders
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
        
        # Define feature columns
        self.categorical = ['gender', 'ever_married', 'work_type',
                          'Residence_type', 'smoking_status']
        self.numerical = ['age', 'hypertension', 'heart_disease', 
                         'avg_glucose_level', 'bmi']
    
    def load_and_prepare_data(self, train_path, test_path=None, submission_path=None):
        """Load data with proper initialization"""
        self.train = pd.read_csv(train_path)
        
        if test_path:
            self.test = pd.read_csv(test_path)
            self.test_ids = self.test['id']
        
        if submission_path:  # This was missing in original
            self.submission_sample = pd.read_csv(submission_path)
        else:
            self.submission_sample = None
    
    def preprocess_features(self):
        """Handle all preprocessing"""
        # Handle missing values
        self.train['bmi'] = self.train['bmi'].fillna(self.train['bmi'].median())
        if hasattr(self, 'test') and self.test is not None:
            self.test['bmi'] = self.test['bmi'].fillna(self.train['bmi'].median())
        
        # Encode target
        y = self.label_encoder.fit_transform(self.train[self.target_column])
        
        # Prepare features
        X = self.train[self.numerical + self.categorical].copy()
        if hasattr(self, 'test') and self.test is not None:
            X_test = self.test[self.numerical + self.categorical].copy()
        
        # Target encode categoricals
        self.cat_encoder.fit(X[self.categorical], y)
        X[self.categorical] = self.cat_encoder.transform(X[self.categorical])
        if hasattr(self, 'test') and self.test is not None:
            X_test[self.categorical] = self.cat_encoder.transform(X_test[self.categorical])
        
        self.X = X
        self.y = y
        if hasattr(self, 'test') and self.test is not None:
            self.X_test = X_test
    
    def train_model(self, params=None, n_splits=5, n_repeats=2, num_boost_round=200):
        """Train model with cross-validation"""
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
        if hasattr(self, 'X_test') and self.X_test is not None:
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
            if hasattr(self, 'X_test') and self.X_test is not None:
                dtest = xgb.DMatrix(self.X_test)
                self.test_preds += model.predict(dtest) / (n_repeats * n_splits)
            
            if (fold + 1) % 2 == 0:
                print(f"Completed fold {fold + 1}")
        
        self.final_model = model
    
    def create_submission(self, filename='submission.csv'):
        """Generate submission file with all safeguards"""
        if not hasattr(self, 'test_preds') or self.test_preds is None:
            raise ValueError("No test predictions available. Run train_model() first.")
        
        if not hasattr(self, 'test_ids') or self.test_ids is None:
            raise ValueError("Test IDs not found. Did you load test data?")
        
        submission = pd.DataFrame({
            'id': self.test_ids,
            self.target_column: self.test_preds
        })
        
        if self.submission_sample is not None:
            submission = submission[self.submission_sample.columns]
        
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        print(f"Prediction range: {submission[self.target_column].min():.4f} to {submission[self.target_column].max():.4f}")
        return submission


# How to use the pipeline correctly
stroke_pipeline = FinalTabularMLPipeline(target_column='stroke')
stroke_pipeline.load_and_prepare_data(
    train_path="/kaggle/input/playground-series-s3e2/train.csv",
    test_path="/kaggle/input/playground-series-s3e2/test.csv",
    submission_path='/kaggle/input/playground-series-s3e2/sample_submission.csv'  # This was missing
)
stroke_pipeline.preprocess_features()
stroke_pipeline.train_model(params={'max_depth': 5, 'scale_pos_weight': 2})
stroke_submission = stroke_pipeline.create_submission('submission.csv')
display(stroke_submission)




