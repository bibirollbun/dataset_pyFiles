# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import optuna
import os
import gc
import matplotlib.pyplot as plt

class PersonalityPredictor:
    """Class to predict personality types using XGBoost with pseudo-labeling and original dataset."""
    def __init__(self, train_path, test_path, original_path, n_splits=5, confidence_threshold=0.95):
        self.train_path = train_path
        self.test_path = test_path
        self.original_path = original_path
        self.n_splits = n_splits
        self.confidence_threshold = confidence_threshold
        self.num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                         'Friends_circle_size', 'Post_frequency']
        self.cat_cols = ['Stage_fear', 'Drained_after_socializing', 'match_p']
        self.target_col = 'Personality'
        self.train_data = None
        self.test_data = None
        self.original_data = None
        self.train_id = None
        self.test_id = None
        self.X = None
        self.y = None
        self.X_test = None
        self.target_encoder = LabelEncoder()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        self.models = []
        self.best_params = None
        self.best_threshold = 0.5
        self.oof_preds = None
        self.test_preds_folds = []

    def load_data(self):
        print("--- Loading Data ---")
        # Verify file existence
        for path, name in [(self.train_path, 'train.csv'), (self.test_path, 'test.csv'), (self.original_path, 'personality_datasert.csv')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist.")

        try:
            self.train_data = pd.read_csv(self.train_path)
            print(f"Train data loaded. Shape: {self.train_data.shape}")
            print(f"Train columns: {self.train_data.columns.tolist()}")
            self.test_data = pd.read_csv(self.test_path)
            print(f"Test data loaded. Shape: {self.test_data.shape}")
            print(f"Test columns: {self.test_data.columns.tolist()}")
            self.original_data = pd.read_csv(self.original_path)
            print(f"Original data loaded. Shape: {self.original_data.shape}")
            print(f"Original columns: {self.original_data.columns.tolist()}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

        # Merge original dataset
        print("\n--- Merging Original Dataset ---")
        self.original_data = self.original_data.rename(columns={'Personality': 'match_p'})
        drop_cols = [col for col in self.original_data.columns if col != 'match_p']
        self.original_data = self.original_data.drop_duplicates(subset=drop_cols)
        print(f"Original data after deduplication. Shape: {self.original_data.shape}")

        self.train_data = self.train_data.merge(self.original_data, how='left')
        self.test_data = self.test_data.merge(self.original_data, how='left')
        print(f"Merged train shape: {self.train_data.shape}, test shape: {self.test_data.shape}")

        # Add match_p_is_null feature
        self.train_data['match_p_is_null'] = self.train_data['match_p'].isna().astype(int)
        self.test_data['match_p_is_null'] = self.test_data['match_p'].isna().astype(int)
        self.num_cols.append('match_p_is_null')

        # Handle ID column
        if 'id' in self.train_data.columns and 'id' in self.test_data.columns:
            self.train_id = self.train_data['id'].copy()
            self.test_id = self.test_data['id'].copy()
            self.train_data.drop('id', axis=1, inplace=True)
            self.test_data.drop('id', axis=1, inplace=True)
        else:
            print("No 'id' column found. Generating synthetic IDs.")
            self.train_id = pd.Series(range(len(self.train_data)), name='id')
            self.test_id = pd.Series(range(len(self.train_data), len(self.train_data) + len(self.test_data)), name='id')

        # Drop Personality from test_data if present
        if self.target_col in self.test_data.columns:
            self.test_data.drop(self.target_col, axis=1, inplace=True)
            print(f"Dropped '{self.target_col}' from test_data.")

        # Validate columns
        missing_cols = [col for col in self.num_cols + self.cat_cols[:2] + [self.target_col] if col not in self.train_data.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in train_data. Available: {self.train_data.columns.tolist()}")

        # Check numerical column types
        invalid_num_cols = [col for col in self.num_cols if not np.issubdtype(self.train_data[col].dtype, np.number)]
        if invalid_num_cols:
            raise ValueError(f"Columns {invalid_num_cols} are not numerical. Types: {self.train_data[self.num_cols].dtypes}")

    def preprocess(self):
        print("\n--- Preprocessing Data ---")
        # Impute missing values
        try:
            self.train_data[self.num_cols] = self.num_imputer.fit_transform(self.train_data[self.num_cols])
            self.test_data[self.num_cols] = self.num_imputer.transform(self.test_data[self.num_cols])
            self.train_data[self.cat_cols] = self.cat_imputer.fit_transform(self.train_data[self.cat_cols])
            self.test_data[self.cat_cols] = self.cat_imputer.transform(self.test_data[self.cat_cols])
        except Exception as e:
            raise Exception(f"Error in imputation: {e}")

        # Encode categorical features
        for col in self.cat_cols:
            self.label_encoders[col] = LabelEncoder()
            try:
                self.train_data[col] = self.label_encoders[col].fit_transform(self.train_data[col].astype(str)).astype(int)
                test_categories = set(self.test_data[col].unique())
                train_categories = set(self.label_encoders[col].classes_)
                if not test_categories.issubset(train_categories):
                    raise ValueError(f"Test data contains unseen categories in {col}: {test_categories - train_categories}")
                self.test_data[col] = self.label_encoders[col].transform(self.test_data[col].astype(str)).astype(int)
            except Exception as e:
                raise Exception(f"Error encoding {col}: {e}")

        # Encode target
        self.train_data[self.target_col] = self.target_encoder.fit_transform(self.train_data[self.target_col])
        print(f"Target encoded: {self.target_encoder.classes_} -> {self.target_encoder.transform(self.target_encoder.classes_)}")

        # Scale numerical features
        self.train_data[self.num_cols] = self.scaler.fit_transform(self.train_data[self.num_cols])
        self.test_data[self.num_cols] = self.scaler.transform(self.test_data[self.num_cols])

        # Feature engineering
        self.train_data['Alone_Drained'] = self.train_data['Time_spent_Alone'] * self.train_data['Drained_after_socializing']
        self.test_data['Alone_Drained'] = self.test_data['Time_spent_Alone'] * self.test_data['Drained_after_socializing']
        self.train_data['Social_Stage'] = self.train_data['Social_event_attendance'] * self.train_data['Stage_fear']
        self.test_data['Social_Stage'] = self.test_data['Social_event_attendance'] * self.test_data['Stage_fear']
        self.num_cols.extend(['Alone_Drained', 'Social_Stage'])

        # Prepare data
        self.X = self.train_data.drop(columns=[self.target_col])
        self.y = self.train_data[self.target_col]
        self.X_test = self.test_data

    def tune_model(self):
        print("\n--- Tuning XGBoost Model ---")
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'scale_pos_weight': len(self.y[self.y == 0]) / len(self.y[self.y == 1]),
                'random_state': 42,
                'tree_method': 'hist'
            }
            model = xgb.XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X, self.y, cv=skf, scoring='accuracy')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        self.best_params = study.best_params
        print(f"Best XGBoost Accuracy: {study.best_value:.4f}, Params: {self.best_params}")

    def pseudo_labeling(self):
        print("\n--- Generating Pseudo-Labels ---")
        # Train initial model
        initial_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            **self.best_params
        }
        initial_model = xgb.XGBClassifier(**initial_params)
        initial_model.fit(self.X, self.y)

        # Generate pseudo-labels
        test_pred_proba = initial_model.predict_proba(self.X_test)[:, 1]
        pseudo_labels = np.where(
            (test_pred_proba > self.confidence_threshold) | (test_pred_proba < (1 - self.confidence_threshold)),
            (test_pred_proba > 0.5).astype(int), -1
        )
        confident_mask = pseudo_labels != -1
        self.X_pseudo = self.X_test[confident_mask].copy()
        self.y_pseudo = pseudo_labels[confident_mask]
        print(f"Number of pseudo-labeled samples: {len(self.y_pseudo)}")

        # Combine data
        self.X_combined = pd.concat([self.X, self.X_pseudo], axis=0, ignore_index=True)
        self.y_combined = pd.concat([self.y, pd.Series(self.y_pseudo)], axis=0, ignore_index=True)

    def train_final_model(self):
        print("\n--- Training Final Model with Pseudo-Labels ---")
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            **self.best_params
        }
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.oof_preds = np.zeros(len(self.y))
        self.test_preds_folds = []
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"--- Fold {fold+1}/{self.n_splits} ---")
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y.iloc[val_idx]

            model = xgb.XGBClassifier(**final_params)
            model.fit(self.X_combined, self.y_combined)
            self.oof_preds[val_idx] = model.predict(X_val)
            fold_acc = accuracy_score(y_val, self.oof_preds[val_idx])
            fold_accuracies.append(fold_acc)
            print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")
            self.test_preds_folds.append(model.predict_proba(self.X_test)[:, 1])
            self.models.append(model)
            
            del X_train, X_val, y_train, y_val, model
            gc.collect()

        print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    def optimize_threshold(self):
        print("\n--- Optimizing Threshold ---")
        final_model = xgb.XGBClassifier(**{
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            **self.best_params
        })
        final_model.fit(self.X_combined, self.y_combined)
        y_pred_proba = final_model.predict_proba(self.X)[:, 1]
        thresholds = np.arange(0.3, 0.7, 0.01)
        self.best_threshold = 0.5
        best_accuracy = accuracy_score(self.y, final_model.predict(self.X))
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            acc = accuracy_score(self.y, y_pred)
            print(f"Threshold {thresh:.2f}: Accuracy {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                self.best_threshold = thresh
        print(f"Best Threshold: {self.best_threshold:.2f}, Best Accuracy: {best_accuracy:.4f}")

    def make_submission(self, file='submission.csv'):
        print("\n--- Generating Submission ---")
        final_test_preds_proba = np.mean(self.test_preds_folds, axis=0)
        final_test_preds_int = (final_test_preds_proba > self.best_threshold).astype(int)
        final_test_predictions_labels = self.target_encoder.inverse_transform(final_test_preds_int)
        submission_df = pd.DataFrame({'id': self.test_id, 'Personality': final_test_predictions_labels})
        submission_df.to_csv(file, index=False)
        print(f"Submission saved as '{file}':")
        print(submission_df.head())

        # Feature importance
        importances = np.mean([m.feature_importances_ for m in self.models], axis=0)
        feat = pd.DataFrame({'Feature': self.X_test.columns, 'Importance': importances})
        feat = feat.sort_values('Importance', ascending=False)
        print("\nFeature Importances:")
        print(feat.head(10))
        
        xgb.plot_importance(self.models[0])
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()

        # Save OOF predictions
        oof_df = pd.DataFrame({'id': self.train_id, 'oof_preds_class': self.oof_preds, 'target': self.y})
        oof_df.to_csv('oof_predictions.csv', index=False)
        print(f"OOF predictions saved to: oof_predictions.csv")

    def run(self):
        self.load_data()
        self.preprocess()
        self.tune_model()
        self.pseudo_labeling()
        self.train_final_model()
        self.optimize_threshold()
        self.make_submission()

# --- Run Pipeline ---
if __name__ == "__main__":
    predictor = PersonalityPredictor(
        train_path='/kaggle/input/playground-series-s5e7/train.csv', 
        test_path='/kaggle/input/playground-series-s5e7/test.csv',    
        original_path='/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv', 
        n_splits=5,
        confidence_threshold=0.95
    )
    predictor.run()

