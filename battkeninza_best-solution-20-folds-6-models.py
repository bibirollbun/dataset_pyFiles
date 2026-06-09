# =========================
# Imports
# =========================
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# def feature_engineering(df):
#     """
#     Enhanced feature engineering with improved group-based imputation
#     """
#     df = df.copy()
    
#     # # Group-based imputation for Time_spent_Alone
#     # df['social_attend_bin'] = pd.qcut(
#     #     df['Social_event_attendance'], 
#     #     q=[0, 0.25, 0.5, 0.75, 1.0], 
#     #     labels=['Q1', 'Q2', 'Q3', 'Q4'],
#     #     duplicates='drop'
#     # )
    
#     # Fill missing values with group median
#     df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(
#         df.groupby('social_attend_bin', observed=True)['Time_spent_Alone'].transform('median')
#     )
    
#     # Fill any remaining missing values with overall median
#     df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    
#     # Drop temporary column
#     df.drop(columns=['social_attend_bin'], inplace=True)
    
#     # Enhanced feature engineering
#     # Create interaction features
#     df['social_time_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
#     df['social_post_interaction'] = df['Social_event_attendance'] * df['Post_frequency']
#     df['friends_post_ratio'] = df['Friends_circle_size'] / (df['Post_frequency'] + 1)
    
#     # Create binned features
#     df['time_alone_binned'] = pd.cut(df['Time_spent_Alone'], bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
#     df['friends_size_binned'] = pd.cut(df['Friends_circle_size'], bins=3, labels=['Small', 'Medium', 'Large'])

#     # More sophisticated group-based imputation
#     df['social_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1e-6)
#     df['social_outside_interaction'] = df['Social_event_attendance'] * df['Going_outside']
    
#     # Polynomial features for key variables
#     df['Friends_circle_size_sq'] = df['Friends_circle_size'] ** 2
#     df['Post_frequency_log'] = np.log1p(df['Post_frequency'])
    
#     # Cluster-based features
#     from sklearn.cluster import KMeans
#     cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     df['social_cluster'] = kmeans.fit_predict(df[cluster_features])
    
#     # Time-based features
#     df['social_post_ratio'] = df['Social_event_attendance'] / (df['Post_frequency'] + 1)
#     df['social_friend_ratio'] = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1)
    
#     # More sophisticated binning
#     df['social_attend_binned'] = pd.qcut(df['Social_event_attendance'], 
#                                        q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
#                                        labels=False)
    
#     # One-hot encode categorical variables
#     categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'time_alone_binned', 'friends_size_binned']
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
#     return df


# class Config:
#     TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
#     TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
#     SUBMISSION_PATH = "/kaggle/input/playground-series-s5e7/sample_submission.csv"
    
#     # Base features
#     BASE_FEATURES = [
#         "Time_spent_Alone", "Social_event_attendance",
#         "Going_outside", "Friends_circle_size", "Post_frequency"
#     ]
    
#     LABEL_COLUMN = "Personality"
#     N_FOLDS = 5
#     RANDOM_STATE = 42

# XGB_PARAM_LIST = [
#     {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False},
#     {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.85, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False},
#     {"max_depth": 7, "learning_rate": 0.05, "n_estimators": 1200, "subsample": 0.8, "colsample_bytree": 0.6, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False},
#     {"max_depth": 5, "learning_rate": 0.15, "n_estimators": 300, "subsample": 1.0, "colsample_bytree": 0.95, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False},
#     {"max_depth": 9, "learning_rate": 0.02, "n_estimators": 2000, "subsample": 0.7, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False},
#     {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.75, "colsample_bytree": 0.75, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "use_label_encoder": False}   
# ]
# LGBM_PARAM_LIST = [
#     {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True},
#     {"num_leaves": 64, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True},
#     {"num_leaves": 128, "learning_rate": 0.02, "n_estimators": 2000, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True},
#     {"num_leaves": 48, "learning_rate": 0.07, "n_estimators": 800, "subsample": 0.95, "colsample_bytree": 0.75, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True},
#     {"num_leaves": 90, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.7, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True},
#     {"num_leaves": 110, "learning_rate": 0.025, "n_estimators": 1800, "subsample": 0.75, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": -1, "verbose": -1, "force_col_wise": True}
# ]
# CAT_PARAM_LIST = [
#     {"iterations": 500, "depth": 6, "learning_rate": 0.1, "random_seed": 42, "verbose": 0},
#     {"iterations": 1000, "depth": 8, "learning_rate": 0.05, "random_seed": 42, "verbose": 0},
#     {"iterations": 1500, "depth": 10, "learning_rate": 0.03, "random_seed": 42, "verbose": 0},
#     {"iterations": 2000, "depth": 7, "learning_rate": 0.02, "random_seed": 42, "verbose": 0},
#     {"iterations": 1200, "depth": 9, "learning_rate": 0.04, "random_seed": 42, "verbose": 0},
#     {"iterations": 800, "depth": 5, "learning_rate": 0.07, "random_seed": 42, "verbose": 0}
# ]






# LEARNERS = []

# # Add 6 XGBoost learners
# for i, params in enumerate(XGB_PARAM_LIST):
#     LEARNERS.append({
#         "name": f"xgb_{i+1}",
#         "Estimator": XGBClassifier,
#         "params": params
#     })

# # Add 6 LightGBM learners
# for i, params in enumerate(LGBM_PARAM_LIST):
#     LEARNERS.append({
#         "name": f"lgbm_{i+1}",
#         "Estimator": LGBMClassifier,
#         "params": params
#     })

# # Add 6 CatBoost learners
# for i, params in enumerate(CAT_PARAM_LIST):
#     LEARNERS.append({
#         "name": f"cat_{i+1}",
#         "Estimator": CatBoostClassifier,
#         "params": params
#     })





# def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
#     """Create time decay weights for model ensemble"""
#     positions = np.arange(n)
#     normalized = positions / (n - 1)
#     weights = decay ** (1.0 - normalized)
#     return weights * n / weights.sum()

# def load_data():
#     """Load and preprocess data"""
#     print("Loading data...")
    
#     # Load data
#     train_df = pd.read_csv(Config.TRAIN_PATH)
#     test_df = pd.read_csv(Config.TEST_PATH)
#     submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
#     print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
#     # Apply feature engineering
#     # train_df = feature_engineering(train_df)
#     # test_df = feature_engineering(test_df)
    
#     # Map target to binary values
#     train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})
    
#     # Get feature columns (exclude ID and target)
#     feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
#     Config.FEATURES = feature_cols
    
#     print(f"Processed shapes - Train: {train_df.shape}, Test: {test_df.shape}")
#     print(f"Number of features: {len(Config.FEATURES)}")
#     print(f"Target distribution:\n{train_df['Personality'].value_counts()}")
    
#     return train_df, test_df, submission_df


def optimize_threshold(y_true, y_pred_proba):
    """Find optimal threshold for classification"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.25
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


# def train_with_cv(train_df, test_df):
#     """Train models with cross-validation"""
#     X = train_df[Config.FEATURES]
#     y = train_df[Config.LABEL_COLUMN]
    
#     # Initialize arrays for out-of-fold predictions
#     oof_preds = np.zeros(len(train_df))
#     test_preds = np.zeros(len(test_df))
    
#     # Cross-validation
#     kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
#     fold_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#         print(f"\n--- Fold {fold + 1} ---")
        
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         # Train individual models
#         fold_test_preds = []
#         fold_val_preds = []
        
#         for learner in LEARNERS:
#             print(f"Training {learner['name']}...")
            
#             model = learner['Estimator'](**learner['params'])
            
#             # Fit model with proper early stopping based on model type
#             if learner['name'].startswith('xgb'):
#                 model.fit(
#                     X_train, y_train,
#                     eval_set=[(X_val, y_val)],
#                     early_stopping_rounds=100,
#                     verbose=False
#                 )
#             elif learner['name'].startswith('lgbm'):
#                 model.fit(
#                     X_train, y_train,
#                     eval_set=[(X_val, y_val)],
#                     eval_metric='logloss',
#                 )
#             elif learner['name'].startswith('cat'):
#                 model.fit(
#                     X_train, y_train,
#                     eval_set=(X_val, y_val),
#                     early_stopping_rounds=100,
#                     verbose=False
#                 )
#             else:  # RandomForest - no early stopping
#                 model.fit(X_train, y_train)
            
#             # Predictions
#             val_pred = model.predict_proba(X_val)[:, 1]
#             test_pred = model.predict_proba(test_df[Config.FEATURES])[:, 1]
            
#             fold_val_preds.append(val_pred)
#             fold_test_preds.append(test_pred)
        
#         # Ensemble predictions (simple average)
#         fold_val_pred = np.mean(fold_val_preds, axis=0)
#         fold_test_pred = np.mean(fold_test_preds, axis=0)
        
#         # Store predictions
#         oof_preds[val_idx] = fold_val_pred
#         test_preds += fold_test_pred / Config.N_FOLDS
        
#         # Calculate fold score
#         fold_score = roc_auc_score(y_val, fold_val_pred)
#         fold_scores.append(fold_score)
#         print(f"Fold {fold + 1} AUC: {fold_score:.4f}")
    
#     # Overall CV score
#     cv_score = roc_auc_score(y, oof_preds)
#     print(f"\nOverall CV AUC: {cv_score:.4f} (+/- {np.std(fold_scores):.4f})")
    
#     return oof_preds, test_preds


def train_final_ensemble(train_df, test_df):
    """Train final ensemble on full training data"""
    print("\nTraining final ensemble...")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[(l['name'], l['Estimator'](**l['params'])) for l in LEARNERS],
        voting='soft'
    )
    
    X = train_df[Config.FEATURES]
    y = train_df[Config.LABEL_COLUMN]
    
    # Train ensemble
    ensemble.fit(X, y)
    
    # Get predictions
    test_probs = ensemble.predict_proba(test_df[Config.FEATURES])[:, 1]
    
    return test_probs

def create_submission(test_probs, submission_df, threshold=0.5):
    """Create submission file"""
    # Convert probabilities to class predictions
    test_preds = (test_probs >= threshold).astype(int)
    
    # Map back to original labels
    submission_df["Personality"] = test_preds
    submission_df['Personality'] = submission_df['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
    
    # Save submission
    submission_df.to_csv("submission.csv", index=False)
    print("Submission created successfully!")
    
    print(f"\nSubmission distribution:")
    print(submission_df['Personality'].value_counts())




# from sklearn.linear_model import LogisticRegression

# def train_stacked_model(train_df, test_df):
#     X = train_df[Config.FEATURES]
#     y = train_df[Config.LABEL_COLUMN]
    
#     # Create out-of-fold predictions for stacking
#     oof_predictions = np.zeros((len(train_df), len(LEARNERS)))
#     test_predictions = np.zeros((len(test_df), len(LEARNERS)))
    
#     kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         for i, learner in enumerate(LEARNERS):
#             model = learner['Estimator'](**learner['params'])
#             model.fit(X_train, y_train)
#             oof_predictions[val_idx, i] = model.predict_proba(X_val)[:, 1]
#             test_predictions[:, i] += model.predict_proba(test_df[Config.FEATURES])[:, 1] / Config.N_FOLDS
    
#     # Train meta-model on out-of-fold predictions
#     meta_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1)
#     meta_model.fit(oof_predictions, y)
    
#     # Get final predictions
#     final_test_preds = meta_model.predict_proba(test_predictions)[:, 1]
    
#     return final_test_preds


# # =========================
# # Main Execution
# # =========================
# def main():
#     """Main execution function"""
#     print("=== Personality Classification Pipeline ===\n")
    
#     # Load data
#     train_df, test_df, submission_df = load_data()
    
#     # Method 1: Cross-validation approach
#     print("\n=== Method 1: Cross-Validation Ensemble ===")
#     oof_preds, test_preds_cv = train_with_cv(train_df, test_df)
    
#     # Find optimal threshold
#     optimal_threshold, best_score = optimize_threshold(train_df[Config.LABEL_COLUMN], oof_preds)
#     print(f"Optimal threshold: {optimal_threshold:.3f} (Accuracy: {best_score:.4f})")
    
#     # Method 2: Final ensemble approach
#     # print("\n=== Method 2: Final Ensemble ===")
#     # test_preds_final = train_final_ensemble(train_df, test_df)
#     finat_test_preds_stacked = train_stacked_model(train_df,test_df);

#     # Combine predictions (weighted average)
#     final_test_preds = test_preds_cv
    
#     # Create submission
#     create_submission(final_test_preds, submission_df, threshold=optimal_threshold)
    
#     print("\n=== Pipeline Complete ===")

# if __name__ == "__main__":
#     main()


# Submission distribution:
# Personality
# Extrovert    4612
# Introvert    1563
# Name: count, dtype: int64

# Overall CV AUC: 0.9692 (+/- 0.0010)
# Optimal threshold: 0.550 (Accuracy: 0.9687)
# #### 0.973279


# Submission distribution:
# Personality
# Extrovert    4627
# Introvert    1548
# Name: count, dtype: int64
# add Code
# ENSEMBLE COMPARISON:
# Hill Climbing AUC: 0.96463
# Weighted Average AUC: 0.96121
# ######0.976518

# Submission distribution:
# Personality
# Extrovert    4611
# Introvert    1564
# Name: count, dtype: int64
# Overall CV AUC: 0.9689 (+/- 0.0032)
# Optimal threshold: 0.560 (Accuracy: 0.9688)
# #####0.973279 

# Submission distribution:
# Personality
# Extrovert    4579
# Introvert    1596
# Name: count, dtype: int64
# Overall CV AUC: 0.9668 (+/- 0.0037)

# Optimal threshold: 0.820 (F1: 0.9693)


# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
    SUBMISSION_PATH = "/kaggle/input/playground-series-s5e7/sample_submission.csv"
    
    # Core features - updated to include categorical columns
    BASE_FEATURES = [
        "Time_spent_Alone", 
        "Social_event_attendance",
        "Going_outside", 
        "Friends_circle_size", 
        "Post_frequency",
        "Stage_fear",
        "Drained_after_socializing"
    ]
    
    CATEGORICAL_FEATURES = [
        "Stage_fear",
        "Drained_after_socializing"
    ]
    
    LABEL_COLUMN = "Personality"
    N_FOLDS = 20
    N_REPEATS = 3
    RANDOM_STATE = 42

# =========================
# Data Preprocessing
# =========================
def preprocess_data(df, label_encode_cols=None):
    """Handle missing values and encode categorical variables"""
    df = df.copy()
    
    if label_encode_cols:
        for col in label_encode_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values - fill with median for numeric, mode for categorical
    for col in df.columns:
        if col in Config.CATEGORICAL_FEATURES:
            if not df[col].empty:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
        elif df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

# =========================
# Model Configurations
# =========================
XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "random_state": Config.RANDOM_STATE,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "scale_pos_weight": 2.84,
    "enable_categorical": False
}

LGBM_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": Config.RANDOM_STATE,
    "verbose": -1,
    "is_unbalance": True
}

cb_params = {
    "border_count": 39,
    "colsample_bylevel": 0.19459088572914465,
    "depth": 2,
    "iterations": 1467,
    "l2_leaf_reg": 31.236169478676036,
    "learning_rate": 0.06852669420904771,
    "min_child_samples": 160,
    "random_state": 42,
    "random_strength": 0.8517786189616939,
    "scale_pos_weight": 1.1691394390533685,
    "subsample": 0.3192330024411618,
    "verbose": False,
}

xgb_params_two = {
    "colsample_bylevel": 0.8168489864941239,
    "colsample_bynode": 0.8850485490950061,
    "colsample_bytree": 0.8379339940113913,
    "gamma": 2.3977359439809276,
    "learning_rate": 0.0616974880921061,
    "max_depth": 6,  # Fixed: was 344 which is too large
    "max_leaves": 89,
    "min_child_weight": 10,
    "n_estimators": 696,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 1.849084818346014,
    "reg_lambda": 29.680324563362227,
    "subsample": 0.5902901569391961,
    "verbosity": 0,
    "enable_categorical": False,  # Changed to False since we're label encoding
    "use_label_encoder": False
}

hgb_params = {
    "l2_regularization": 28.13576008319012,
    "learning_rate": 0.1543598086529694,
    "max_depth": 10,  # Fixed: was 325 which is too large
    # "max_features": 0.323620656779567,
    "max_iter": 2490,
    "max_leaf_nodes": 216,
    "min_samples_leaf": 12,
    "random_state": 42,
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.6467443250209886,
    "learning_rate": 0.06547186748153115,
    "min_child_samples": 34,
    "min_child_weight": 0.24399244943904663,
    "n_estimators": 498,
    "n_jobs": -1,
    "num_leaves": 158,
    "random_state": 42,
    "reg_alpha": 6.568921253574134,
    "reg_lambda": 62.66165355751099,
    "subsample": 0.8,  # Fixed: was too small
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.8384834064170148,
    "learning_rate": 0.07006829797238343,
    "min_child_samples": 46,
    "min_child_weight": 0.7625394962666617,
    "n_estimators": 1887,
    "n_jobs": -1,
    "num_leaves": 341,
    "random_state": 42,
    "reg_alpha": 10.53082019937197,
    "reg_lambda": 67.44600065144685,
    "subsample": 0.4925008305336127,
    "verbose": -1
}

lgbm_dart_params = {
    "boosting_type": "dart",
    "colsample_bytree": 0.7592971191793424,
    "learning_rate": 0.046141766106846074,
    "min_child_samples": 18,
    "min_child_weight": 0.4740109054323218,
    "n_estimators": 4035,
    "n_jobs": -1,
    "num_leaves": 393,
    "random_state": 42,
    "reg_alpha": 48.016799341666605,
    "reg_lambda": 89.12860300833658,
    "subsample": 0.8,  # Fixed: was too small
    "verbose": -1
}

# Fixed LEARNERS configuration with unique names
LEARNERS = [
    {
        "name": "xgb_baseline",
        "Estimator": XGBClassifier,
        "params": XGB_PARAMS
    },
    {
        "name": "lgbm_baseline",
        "Estimator": LGBMClassifier,
        "params": LGBM_PARAMS
    },
    {
        "name": "xgb_optimized",
        "Estimator": XGBClassifier,
        "params": xgb_params_two
    },
    {
        "name": "lgbm_gbdt",
        "Estimator": LGBMClassifier,
        "params": lgbm_params
    },
    {
        "name": "lgbm_dart",
        "Estimator": LGBMClassifier,
        "params": lgbm_dart_params
    },
    {
        "name": "lgbm_goss",
        "Estimator": LGBMClassifier,
        "params": lgbm_goss_params
    },
    {
        "name": "catboost_optimized",
        "Estimator": CatBoostClassifier,
        "params": cb_params
    },
    {
        "name": "hgb_optimized",
        "Estimator": HistGradientBoostingClassifier,
        "params": hgb_params
    }
]

# =========================
# Missing Functions
# =========================
def optimize_threshold(y_true, y_pred):
    """Find optimal threshold for F1 score"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        score = f1_score(y_true, y_pred_binary)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def create_submission(test_preds, submission_df, threshold=0.5):
    """Create submission file"""
    submission_df = submission_df.copy()
    
    # Convert probabilities to binary predictions
    binary_preds = (test_preds >= threshold).astype(int)
    
    # Map back to original labels
    submission_df['Personality'] = binary_preds
    submission_df['Personality'] = submission_df['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    print(f"Submission saved to submission.csv")
    print(f"Prediction distribution:\n{submission_df['Personality'].value_counts()}")

# =========================
# Data Loading with Preprocessing
# =========================
def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Preprocess data
    train_df = preprocess_data(train_df, Config.CATEGORICAL_FEATURES)
    test_df = preprocess_data(test_df, Config.CATEGORICAL_FEATURES)
    
    # Map target to binary values
    train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})
    
    print(f"Number of features: {len(Config.BASE_FEATURES)}")
    print(f"Target distribution:\n{train_df['Personality'].value_counts()}")
    
    return train_df, test_df, submission_df

# =========================
# Training with CV (Fixed)
# =========================
def train_with_cv(train_df, test_df):
    """Train models with repeated stratified CV"""
    X = train_df[Config.BASE_FEATURES]
    y = train_df[Config.LABEL_COLUMN]
    X_test = test_df[Config.BASE_FEATURES]
    
    # Initialize arrays for predictions
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    # Repeated stratified KFold
    rskf = RepeatedStratifiedKFold(
        n_splits=Config.N_FOLDS,
        n_repeats=Config.N_REPEATS,
        random_state=Config.RANDOM_STATE
    )
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_test_preds = []
        
        for learner in LEARNERS:
            print(f"Training {learner['name']}...")
            model = learner['Estimator'](**learner['params'])
            
            # Fit with early stopping for applicable models
            if learner['name'].startswith('xgb'):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif learner['name'].startswith('lgbm'):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='logloss'
                )
            elif learner['name'].startswith('catboost'):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Get predictions
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            
            # Store predictions
            fold_test_preds.append(test_pred)
        
        # Average model predictions for this fold
        fold_test_pred = np.mean(fold_test_preds, axis=0)
        test_preds += fold_test_pred / (Config.N_FOLDS * Config.N_REPEATS)
        
        # Average validation predictions for this fold
        fold_val_preds = []
        for learner in LEARNERS:
            model = learner['Estimator'](**learner['params'])
            
            if learner['name'].startswith('xgb'):
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif learner['name'].startswith('lgbm'):
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss')
            elif learner['name'].startswith('catboost'):
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            
            val_pred = model.predict_proba(X_val)[:, 1]
            fold_val_preds.append(val_pred)
        
        # Average validation predictions
        fold_val_pred = np.mean(fold_val_preds, axis=0)
        oof_preds[val_idx] = fold_val_pred
        
        # Calculate fold score
        fold_score = roc_auc_score(y_val, fold_val_pred)
        fold_scores.append(fold_score)
        print(f"Fold {fold + 1} AUC: {fold_score:.4f}")
    
    # Overall CV score
    cv_score = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {cv_score:.4f} (+/- {np.std(fold_scores):.4f})")
    
    return oof_preds, test_preds

# =========================
# Main Execution
# =========================
def main():
    print("=== Personality Classification Pipeline ===\n")
    
    # Load and preprocess data
    train_df, test_df, submission_df = load_data()
    
    # Train with cross-validation
    print("\n=== Training with Repeated Stratified CV ===")
    oof_preds, test_preds = train_with_cv(train_df, test_df)
    
    # Find optimal threshold
    optimal_threshold, best_score = optimize_threshold(
        train_df[Config.LABEL_COLUMN], 
        oof_preds
    )
    print(f"\nOptimal threshold: {optimal_threshold:.3f} (F1: {best_score:.4f})")
    
    # Create submission
    create_submission(test_preds, submission_df, threshold=optimal_threshold)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()




