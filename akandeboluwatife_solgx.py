import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 100
VERBOSE = False

# ==================== MAIN PIPELINE ====================
class SolanaGraduationPredictor:
    """
    End-to-end prediction pipeline for NFT graduation status
    """
    
    def __init__(self):
        self.input_dir = '/kaggle/input'
        self.output_dir = '/kaggle/working'
        self.dataset_name = 'solana-skill-sprint-memcoin-graduation'
        self.required_features = ['slot_min', 'is_valid']
        self.target_col = 'has_graduated'
        
    def load_data(self):
        """Load and validate datasets"""
        base_path = os.path.join(self.input_dir, self.dataset_name)
        train = pd.read_csv(os.path.join(base_path, 'train.csv'))
        test = pd.read_csv(os.path.join(base_path, 'test_unlabeled.csv'))
        
        print("\nğŸ”� Data Samples:")
        print("Training Data:\n", train.head(3).to_markdown(tablefmt="grid"))
        print("\nTest Data:\n", test.head(3).to_markdown(tablefmt="grid"))
        
        return train, test

    def preprocess(self, train, test):
        """Feature engineering and preprocessing"""
        X_train = train[self.required_features].copy()
        y_train = train[self.target_col].astype(int)
        X_test = test[self.required_features].copy()
        
        # Add feature engineering (example: aggregating data or creating interaction features)
        X_train['slot_min_log'] = np.log1p(X_train['slot_min'])  # Log transformation
        X_test['slot_min_log'] = np.log1p(X_test['slot_min'])
        
        # Add interaction term (example)
        X_train['slot_min_is_valid'] = X_train['slot_min'] * X_train['is_valid']
        X_test['slot_min_is_valid'] = X_test['slot_min'] * X_test['is_valid']
        
        # Standardize features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        return X_train, y_train, X_test

    def train_and_predict(self, X_train, y_train, X_test):
        """Cross-validated model training"""
        model = lgb.LGBMClassifier(
            n_estimators=2000,  # Increased number of estimators
            learning_rate=0.01,  # Lower learning rate for more stability
            num_leaves=127,  # Increased number of leaves to capture more complex patterns
            max_depth=-1,  # Let the model decide depth
            subsample=0.85,  # Random sampling of data points for each tree
            colsample_bytree=0.85,  # Random sampling of features for each tree
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        test_preds = np.zeros(X_test.shape[0])
        fold_scores = []
        
        print("\nğŸ“Š Cross-Validation Results:")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(VERBOSE)
                ]
            )
            
            val_pred = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            print(f"Fold {fold} AUC: {fold_auc:.5f}")
            test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        return test_preds, np.mean(fold_scores)

    def generate_submission(self, test, predictions):
        """Create competition submission file"""
        output_path = os.path.join(self.output_dir, 'submission.csv')
        pd.DataFrame({
            'mint': test['mint'],
            self.target_col: predictions
        }).to_csv(output_path, index=False)
        return output_path

    def run(self):
        """Execute full pipeline"""
        print("\nğŸ”§ Starting Pipeline...")
        try:
            # Data Loading
            train, test = self.load_data()
            
            # Preprocessing
            X_train, y_train, X_test = self.preprocess(train, test)
            
            # Model Training
            predictions, mean_auc = self.train_and_predict(X_train, y_train, X_test)
            
            # Generate Submission
            sub_path = self.generate_submission(test, predictions)
            
            # Final Report
            print(f"""
            ğŸ�¯ Final Report
            {'='*40}
            Model Performance:
            â€¢ Mean CV AUC: {mean_auc:.5f}
            â€¢ Best Fold: {max(fold_scores):.5f}
            
            Submission File:
            â€¢ Saved to: {sub_path}
            â€¢ Sample Predictions:
              {predictions[:3]}
            """)
            
        except Exception as e:
            print(f"\nâ�Œ Pipeline Failed: {str(e)}")

# ==================== EXECUTION ====================
if __name__ == "__main__":
    predictor = SolanaGraduationPredictor()
    predictor.run()
    print("\nâœ¨ Pipeline executed successfully!")

