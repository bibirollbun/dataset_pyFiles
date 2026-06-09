import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from xgboost import XGBClassifier
import gc
import os
import random
import time

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Generate random seed based on current time
RANDOM_SEED = int(time.time()) % 10000
print(f"ğŸ�² Using random seed: {RANDOM_SEED}")

# Set random seeds for reproducibility within this run
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("ğŸš€ Loading data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# Enhanced data combination with deduplication
print("ğŸ”„ Processing and combining datasets...")
train = pd.concat([train, original], axis=0, ignore_index=True)
train = train.drop_duplicates().reset_index(drop=True)
print(f"âœ… Combined dataset shape: {train.shape}")

class OptimizedFeatureProcessor:
    """Optimized feature processing using only original columns"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def encode_categorical_features(self, train_df, test_df):
        """Encode categorical features"""
        print("ğŸ”§ Encoding categorical features...")
        cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and id columns
        exclude_cols = ['Fertilizer Name', 'id']
        cat_cols = [col for col in cat_cols if col not in exclude_cols]
        
        for col in cat_cols:
            le = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
            le.fit(combined_values)
            
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            self.label_encoders[col] = le
            
            print(f"âœ… Encoded {col}: {len(le.classes_)} unique values")
            
        return train_df, test_df
        
    def handle_outliers(self, df):
        """Handle outliers using IQR method"""
        print("ğŸ”§ Handling outliers...")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        if 'Fertilizer Name' in numerical_cols:
            numerical_cols.remove('Fertilizer Name')
            
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df
    
    def scale_features(self, train_df, test_df):
        """Scale numerical features"""
        print("ğŸ”§ Scaling numerical features...")
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id', 'Fertilizer Name']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        train_df[numerical_cols] = self.scaler.fit_transform(train_df[numerical_cols])
        test_df[numerical_cols] = self.scaler.transform(test_df[numerical_cols])
        
        return train_df, test_df

def mapk(actual, predicted, k=3):
    """Calculate MAP@K score"""
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

class OptimizedXGBoostClassifier:
    """Optimized XGBoost classifier"""
    
    def __init__(self):
        self.model = None
        
    def get_optimized_params(self):
        """Enhanced XGBoost parameters for higher accuracy with randomization"""
        # Add some randomization to hyperparameters
        base_lr = 0.03
        lr_variation = np.random.uniform(-0.01, 0.01)
        learning_rate = max(0.01, base_lr + lr_variation)
        
        base_depth = 10
        depth_variation = np.random.randint(-2, 3)
        max_depth = max(3, min(15, base_depth + depth_variation))
        
        colsample_variation = np.random.uniform(-0.1, 0.1)
        colsample_bytree = max(0.5, min(1.0, 0.85 + colsample_variation))
        
        subsample_variation = np.random.uniform(-0.1, 0.1)
        subsample = max(0.5, min(1.0, 0.85 + subsample_variation))
        
        print(f"ğŸ�² Randomized params - LR: {learning_rate:.4f}, Depth: {max_depth}, Colsample: {colsample_bytree:.3f}, Subsample: {subsample:.3f}")
        
        return {
            'max_depth': max_depth,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'n_estimators': 5000,
            'learning_rate': learning_rate,
            'gamma': 0.05,
            'min_child_weight': 1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'max_leaves': 0,
            'grow_policy': 'depthwise',
            'objective': 'multi:softprob',
            'random_state': RANDOM_SEED,  # Use the time-based random seed
            'tree_method': 'hist',
            'device': 'cuda:0',
            'verbosity': 0,
            'eval_metric': 'mlogloss'
        }
    
    def train_model(self, X, y, X_test):
        """Train XGBoost with cross-validation"""
        print("ğŸš€ Training XGBoost model...")
        
        FOLDS = 7  # Increased folds for better validation
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)  # Use random seed
        
        num_classes = y.nunique()
        oof_predictions = np.zeros((len(X), num_classes))
        test_predictions = np.zeros((len(X_test), num_classes))
        
        fold_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*20} FOLD {fold+1} {'='*20}")
            
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            # Initialize XGBoost model with randomized parameters
            xgb_params = self.get_optimized_params()
            model = XGBClassifier(**xgb_params)
            
            print(f"ğŸ”„ Training XGBoost...")
            model.fit(X_train, y_train, 
                     eval_set=[(X_valid, y_valid)], 
                     early_stopping_rounds=150,
                     verbose=False)
            
            # Predictions
            valid_pred = model.predict_proba(X_valid)
            test_pred = model.predict_proba(X_test)
            
            oof_predictions[valid_idx] = valid_pred
            test_predictions += test_pred / FOLDS
            
            # Calculate MAP@3 for this fold
            top_3_preds = np.argsort(valid_pred, axis=1)[:, -3:][:, ::-1]
            actual = [[label] for label in y_valid]
            map3_score = mapk(actual, top_3_preds)
            fold_scores.append(map3_score)
            print(f"âœ… XGBoost FOLD {fold+1} MAP@3: {map3_score:.5f}")
            
            # Memory cleanup
            gc.collect()
        
        # Overall CV score
        overall_map3 = np.mean(fold_scores)
        print(f"\nğŸ�† OVERALL XGBoost CV MAP@3: {overall_map3:.5f} Â± {np.std(fold_scores):.5f}")
        
        return oof_predictions, test_predictions, overall_map3

# Main execution
print("ğŸ”§ Initializing feature processing...")
processor = OptimizedFeatureProcessor()

# Process data - keeping only original columns
train_processed = processor.handle_outliers(train.copy())
test_processed = test.copy()

# Encode categorical features BEFORE target encoding
train_processed, test_processed = processor.encode_categorical_features(train_processed, test_processed)

# Target encoding
fer_label_enc = LabelEncoder()
train_processed["Fertilizer Name"] = fer_label_enc.fit_transform(train_processed["Fertilizer Name"])

# Scaling
train_processed, test_processed = processor.scale_features(train_processed, test_processed)

print(f"âœ… Final processed train shape: {train_processed.shape}")
print(f"âœ… Final processed test shape: {test_processed.shape}")

# Prepare data for training
X = train_processed.drop(columns=["id", "Fertilizer Name"])
y = train_processed["Fertilizer Name"]
X_test = test_processed.drop(columns=["id"])

print(f"ğŸ“Š Number of unique fertilizers: {y.nunique()}")
print(f"ğŸ“Š Feature count: {X.shape[1]}")
print(f"ğŸ“Š Training samples: {X.shape[0]}")

# Train XGBoost
xgb_classifier = OptimizedXGBoostClassifier()
oof_preds, test_preds, cv_score = xgb_classifier.train_model(X, y, X_test)

# Generate final predictions
print("\nğŸ�¯ Generating final predictions...")
top_3_indices = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_label_enc.inverse_transform(top_3_indices.ravel()).reshape(top_3_indices.shape)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)
print(f"âœ… Submission saved as {submission_filename} with CV MAP@3: {cv_score:.5f}")
print(f"ğŸš€ Optimized XGBoost training completed with seed {RANDOM_SEED}!")




