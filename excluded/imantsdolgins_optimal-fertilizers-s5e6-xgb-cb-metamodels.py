# --- SETUP AND IMPORTS ---
import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

# --- Environment Setup ---
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("âœ… All libraries imported and environment setup complete.")


# --- Verify Input Files ---
print("\n â˜‘ï¸� Available Input Files")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("-----------------------------------------------")


# --- Configuration --- 
class CFG:
    # --- File Paths ---
    TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
    ORIGINAL_PATH = '/kaggle/input/fertilizer-prediction/Fertilizer-Prediction.csv'
    TEST_PATH = "/kaggle/input/playground-series-s5e6/test.csv"
    SUBMISSION_PATH = "/kaggle/input/playground-series-s5e6/sample_submission.csv"

    # --- Model & CV Parameters ---
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # --- XGBoost Hyperparameters ---
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'max_depth': 16,
        'learning_rate': 0.01,
        'n_estimators': 7000,
        'reg_alpha': 3.0,
        'reg_lambda': 1.5,
        'gamma': 0.3,
        'subsample': 0.85,
        'colsample_bytree': 0.4,
        'min_child_weight': 5,
        'random_state': RANDOM_STATE,
        'eval_metric': 'mlogloss',
        'device': "cuda"
    }

    CAT_PARAMS = {
        'task_type': 'GPU',
        'loss_function': 'MultiClass',
        'iterations': 15000,
        'random_state': RANDOM_STATE,
        'learning_rate': 0.025864872617827245,
        'depth': 6,
        'l2_leaf_reg': 1.6075377486224298,
        'min_data_in_leaf': 47,
        'bagging_temperature': 0.00304800613681506174,
        'random_strength': 0.0010001403722718295,
        'border_count': 219
    }

    # --- Stacking Meta-Learner Parameters ---
    LR_META_PARAMS = {
        'C': 0.1,
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs',
        'n_jobs': -1
    }
    RF_META_PARAMS = {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_leaf': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
     }

# --- Simplified Meta-Learner Parameters ---

    LGBM_META_PARAMS_SIMPLE = {
        'objective': 'multiclass',
        'n_estimators': 30,
        'learning_rate': 0.05,
        'num_leaves': 7,
        'max_depth': 2,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    XGB_META_PARAMS_SIMPLE = {
        'objective': 'multi:softprob',
        'device': 'cuda',
        'n_estimators': 40,
        'learning_rate': 0.05,
        'max_depth': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': RANDOM_STATE
    }
    # LGBM_META_PARAMS = {
    #     'objective': 'multiclass',
    #     'n_estimators': 200,
    #     'learning_rate': 0.05,
    #     'num_leaves': 15,
    #     'max_depth': 4,
    #     'reg_alpha': 0.1,
    #     'reg_lambda': 0.1,
    #     'random_state': RANDOM_STATE,
    #     'n_jobs': -1
    # }
    # XGB_META_PARAMS = {
    #     'objective': 'multi:softprob',
    #     'device': 'cuda',
    #     'n_estimators': 300,
    #     'learning_rate': 0.05,
    #     'max_depth': 3,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'random_state': RANDOM_STATE
    # }
    
    # GBC_META_PARAMS = {
    #     'n_estimators': 150,
    #     'learning_rate': 0.05,
    #     'max_depth': 3,
    #     'subsample': 0.7,
    #     'min_samples_leaf': 10,
    #     'random_state': RANDOM_STATE
    # }

print("âœ… Configuration setup complete.")


# --- Helper Functions ---
def mapk(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    avg_precisions = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        if true_label in top_k_preds[i]:
            rank = np.where(top_k_preds[i] == true_label)[0][0] + 1
            avg_precisions.append(1 / rank)
        else:
            avg_precisions.append(0.0)
    return np.mean(avg_precisions)

def create_submission_file(test_preds: np.ndarray, target_le: LabelEncoder, file_name: str):
    print(f"\nğŸ”„ Generating submission file: {file_name}...")
    submission_df = pd.read_csv(CFG.SUBMISSION_PATH)
    top_preds_indices = np.argsort(test_preds, axis=1)[:, ::-1][:, :3]
    top_preds_names = target_le.inverse_transform(top_preds_indices.ravel()).reshape(top_preds_indices.shape)
    submission_df['Fertilizer Name'] = [' '.join(row) for row in top_preds_names]
    submission_df.to_csv(file_name, index=False)
    print(f"âœ… {file_name} created successfully!")

def train_evaluate_meta_model(model, model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, target_le: LabelEncoder):
    print(f"\n--- ğŸš€ Training {model_name} Meta-Learner ---")
    
    # --- Train the model ---
    model.fit(X_train, y_train)
    print(f"âœ… {model_name} Meta-Learner trained.")

    # --- Evaluate on OOF data ---
    oof_preds_stack = model.predict_proba(X_train)
    stacking_score = mapk(y_train, oof_preds_stack, k=3)
    print(f"ğŸ�† {model_name} Stacking OOF MAP@3 Score: {stacking_score:.6f}")

    # --- Predict on test data and create submission file ---
    final_predictions = model.predict_proba(X_test)
    submission_filename = f"submission_stacking_{model_name.lower().replace(' ', '_')}.csv"
    
    # Reuse existing helper function to create the file
    create_submission_file(final_predictions, target_le, submission_filename)

print("âœ… Helper functions defined.")


# --- Data Loading ---
print("ğŸ’¾ Loading all necessary data...")
df_train = pd.read_csv(CFG.TRAIN_PATH)
df_original = pd.read_csv(CFG.ORIGINAL_PATH)
df_test = pd.read_csv(CFG.TEST_PATH)

# --- Store test IDs for submission file ---
test_ids = df_test['id'].copy()

# --- Drop unnecessary 'id' columns ---
df_train.drop(columns=['id'], inplace=True, errors='ignore')
df_test.drop(columns=['id'], inplace=True, errors='ignore')

print("âœ… Data loaded successfully.")
print(f"âœ”ï¸�Train data shape: {df_train.shape}")
print(f"âœ”ï¸�Original data shape: {df_original.shape}")
print(f"âœ”ï¸�Test data shape: {df_test.shape}")


print("\nâš™ï¸� --- Starting Data Preparation Pipeline ---")

# --- 1. Separate Target Variables ---
target = df_train.pop('Fertilizer Name')
target_original = df_original.pop('Fertilizer Name')
print("âœ… Target variables separated.")

# --- 2. Unified Label Encoding for Features and Target ---
cat_cols = df_test.select_dtypes(include=['object']).columns.tolist()
print(f"â˜‘ï¸� Categorical features to be encoded: {cat_cols}")

# Feature Encoding
for col in cat_cols:
    le = LabelEncoder()
    combined_data = pd.concat([df_train[col], df_original[col], df_test[col]], axis=0).astype(str)
    le.fit(combined_data)
    df_train[col] = le.transform(df_train[col].astype(str))
    df_original[col] = le.transform(df_original[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))

# Target Encoding
target_le = LabelEncoder()
combined_target = pd.concat([target, target_original], axis=0)
target_le.fit(combined_target)
target = target_le.transform(target)
target_original = target_le.transform(target_original)

# Update the number of classes in the config
CFG.XGB_PARAMS['num_class'] = len(target_le.classes_)
print("âœ… All dataframes have been successfully label-encoded.")

# --- 3. Verification ---
print("\nğŸ”� --- Final Data Overview ---")
print(f"âœ”ï¸�Train data shape: {df_train.shape}")
print(f"âœ”ï¸�Original data shape: {df_original.shape}")
print(f"âœ”ï¸�Test data shape: {df_test.shape}")
print("\nğŸ”¢ Train data head (now fully numerical):")
display(df_train.head())


print("\nğŸš€ --- Training XGBoost Model ---")

# --- Initialize Prediction Storage ---
oof_preds_xgb = np.zeros((len(df_train), len(target_le.classes_)))
test_preds_xgb = np.zeros((len(df_test), len(target_le.classes_)))

# --- Cross-Validation Training Loop ---
sk_fold = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_STATE)
for fold, (train_idx, val_idx) in enumerate(sk_fold.split(df_train, target)):
    print(f"\nğŸš€ --- Fold {fold + 1}/{CFG.N_FOLDS} ---")
    X_train, y_train = df_train.iloc[train_idx], target[train_idx]
    X_val, y_val = df_train.iloc[val_idx], target[val_idx]
    
    # Augment training data with the original dataset
    X_train_aug = pd.concat([X_train, df_original], axis=0)
    y_train_aug = np.concatenate([y_train, target_original], axis=0)
    
    dtrain = xgb.DMatrix(X_train_aug, label=y_train_aug)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(df_test)
    
    model = xgb.train(
        CFG.XGB_PARAMS, dtrain,
        num_boost_round=CFG.XGB_PARAMS['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    oof_preds_xgb[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration)) / CFG.N_FOLDS
    fold_score = mapk(y_val, oof_preds_xgb[val_idx], k=3)
    print(f"  âœ”ï¸�  Fold {fold+1} MAP@3 Score: {fold_score:.6f}")

final_oof_score_xgb = mapk(target, oof_preds_xgb, k=3)
print(f"\n\nğŸ�† --- XGBoost Overall OOF MAP@3 Score: {final_oof_score_xgb:.6f} ---")


print("\nğŸš€ --- Training CatBoost Model ---")

# --- Initialize Prediction Storage ---
oof_preds_cat = np.zeros((len(df_train), len(target_le.classes_)))
test_preds_cat = np.zeros((len(df_test), len(target_le.classes_)))
cat_feature_indices = [df_train.columns.get_loc(col) for col in cat_cols]

# --- Cross-Validation Training Loop ---
sk_fold = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_STATE)
for fold, (train_idx, val_idx) in enumerate(sk_fold.split(df_train, target)):
    print(f"\nğŸš€ --- Fold {fold + 1}/{CFG.N_FOLDS} ---")
    X_train, y_train = df_train.iloc[train_idx], target[train_idx]
    X_val, y_val = df_train.iloc[val_idx], target[val_idx]

    # Augment training data with the original dataset
    X_train_aug = pd.concat([X_train, df_original], axis=0)
    y_train_aug = np.concatenate([y_train, target_original], axis=0)
    
    model = cat.CatBoostClassifier(**CFG.CAT_PARAMS)
    model.fit(X_train_aug, y_train_aug, eval_set=[(X_val, y_val)],
              cat_features=cat_feature_indices,
              early_stopping_rounds=100, verbose=500)
    
    oof_preds_cat[val_idx] = model.predict_proba(X_val)
    test_preds_cat += model.predict_proba(df_test) / CFG.N_FOLDS
    fold_score = mapk(y_val, oof_preds_cat[val_idx], k=3)
    print(f"  âœ”ï¸�  Fold {fold+1} MAP@3 Score: {fold_score:.6f}")

final_oof_score_cat = mapk(target, oof_preds_cat, k=3)
print(f"\n\nğŸ�† --- CatBoost Overall OOF MAP@3 Score: {final_oof_score_cat:.6f} ---")


# --- Create individual model submissions ---
create_submission_file(test_preds_xgb, target_le, "submission_xgb.csv")
create_submission_file(test_preds_cat, target_le, "submission_catboost.csv")


try:
    print("\nâœ… Preparing meta-features from base model predictions...")
    
    X_meta_train = np.hstack([oof_preds_xgb, oof_preds_cat])
    X_meta_test = np.hstack([test_preds_xgb, test_preds_cat])
    y_meta_train = target # The original, encoded target
    
    print(f"âœ”ï¸� Shape of Meta-Training Data (X_meta_train): {X_meta_train.shape}")
    print(f"âœ”ï¸� Shape of Meta-Test Data (X_meta_test): {X_meta_test.shape}")

    # --- Stacking Experiments ---
    
    # Experiment 1: Logistic Regression
    lr_model = LogisticRegression(**CFG.LR_META_PARAMS)
    train_evaluate_meta_model(lr_model, "Logistic Regression", X_meta_train, y_meta_train, X_meta_test, target_le)

    # Experiment 2: Random Forest
    rf_model = RandomForestClassifier(**CFG.RF_META_PARAMS)
    train_evaluate_meta_model(rf_model, "Random Forest", X_meta_train, y_meta_train, X_meta_test, target_le)
    
    # Experiment 3: LightGBM
    lgbm_model = lgb.LGBMClassifier(**CFG.LGBM_META_PARAMS_SIMPLE) #LGBM_META_PARAMS
    train_evaluate_meta_model(lgbm_model, "LightGBM", X_meta_train, y_meta_train, X_meta_test, target_le)

    # Experiment 4: Gradient Boosting Classifier
    xgb_model = XGBClassifier(**CFG.XGB_META_PARAMS_SIMPLE) #XGB_META_PARAMS
    train_evaluate_meta_model(xgb_model, "XGBoost", X_meta_train, y_meta_train, X_meta_test, target_le)
    
    # # Experiment 4: Gradient Boosting Classifier
    # gbc_model = GradientBoostingClassifier(**CFG.GBC_META_PARAMS)
    # train_evaluate_meta_model(gbc_model, "Gradient Boosting", X_meta_train, y_meta_train, X_meta_test, target_le)
    
    print("\nğŸ�‰ Stacking project complete!")

except NameError as e:
    print(f"\nâ�Œ FATAL ERROR: A required prediction variable is not in memory.")
    print("Please run (Model Training) for both XGBoost and CatBoost.")
    print(f"Details: {e}")


print("ğŸ§ª Experimenting with weighted blending...")

# Give more weight to the stronger XGBoost model
blended_preds_70_30 = (0.7 * test_preds_xgb) + (0.3 * test_preds_cat)
create_submission_file(blended_preds_70_30, target_le, "submission_blended_70_30.csv")

# Try another combination
blended_preds_80_20 = (0.8 * test_preds_xgb) + (0.2 * test_preds_cat)
create_submission_file(blended_preds_80_20, target_le, "submission_blended_80_20.csv")

