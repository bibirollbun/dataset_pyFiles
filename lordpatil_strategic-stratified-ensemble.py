import pandas as pd
import numpy as np
import warnings


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb


warnings.filterwarnings('ignore')



# Load competition data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Load original external dataset, as recommended in discussions
# The file might have a different name, adjust if necessary.
original_df = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
print("External 'Fertilizer Prediction.csv' dataset loaded successfully.")


def standardize_columns(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('Temparature', 'Temperature')
    df = df.rename(columns={'Phosphor_ous': 'Phosphorous'})
    return df


train_df = standardize_columns(train_df)
test_df = standardize_columns(test_df)
original_df = standardize_columns(original_df)


test_ids = test_df['id']


print("Augmenting training data with the external dataset (x4 times)...")
train_df = pd.concat([train_df] + [original_df] * 4, ignore_index=True)


train_df = train_df.drop('id', axis=1, errors='ignore')


def create_features(df):
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    
    # Nutrient Ratios
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
    
    # Total Nutrients
    df['total_nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    
    # Climate Aggregates
    df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100.0
    
    # Soil-Crop Interactions (very strong signal)
    df['soil_crop_interaction'] = df['Soil_Type'] + "_" + df['Crop_Type']
    
    return df


print("Creating new features for train and test sets...")
train_df = create_features(train_df)
test_df = create_features(test_df)


target = 'Fertilizer_Name'
le = LabelEncoder()
train_df[target] = le.fit_transform(train_df[target])
label_classes = le.classes_
print(f"Target variable '{target}' encoded.")


features = [col for col in test_df.columns if col not in ['id']]
categorical_features = ['Soil_Type', 'Crop_Type', 'soil_crop_interaction']


for col in categorical_features:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')
    
X = train_df[features]
y = train_df[target]
X_test = test_df[features]


print("\n--- Model Training ---")


# Define model parameters for GPU training
LGB_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(label_classes),
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 20,
    'max_depth': 5,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'device': 'gpu',  
}

XGB_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': len(label_classes),
    'eta': 0.03,
    'max_depth': 5,
    'seed': 42,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'tree_method': 'gpu_hist',  
    # 'nthread' is not needed for GPU
}

CAT_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'random_seed': 42,
    'verbose': 0,
    'cat_features': categorical_features,
    'task_type': 'GPU',  
}




# Setup Cross-validation
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)




# To store predictions
lgb_preds = np.zeros((len(X_test), len(label_classes)))
xgb_preds = np.zeros((len(X_test), len(label_classes)))
cat_preds = np.zeros((len(X_test), len(label_classes)))




# For XGBoost, we need to label encode categorical features
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in categorical_features:
    le_cat = LabelEncoder()
    # Fit on the combined data from X and X_test to handle unseen categories
    combined_series = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le_cat.fit(combined_series)
    X_xgb[col] = le_cat.transform(X[col].astype(str))
    X_test_xgb[col] = le_cat.transform(X_test[col].astype(str))


print(f"Starting GPU training with {NFOLDS}-fold stratified CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"===== Fold {fold+1} =====")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- LightGBM ---
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_preds += lgb_model.predict_proba(X_test) / NFOLDS

    # --- XGBoost ---
    print("Training XGBoost...")
    X_train_xgb, y_train_xgb = X_xgb.iloc[train_idx], y.iloc[train_idx]
    X_val_xgb, y_val_xgb = X_xgb.iloc[val_idx], y.iloc[val_idx]
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train_xgb, y_train_xgb,
                  eval_set=[(X_val_xgb, y_val_xgb)],
                  early_stopping_rounds=100,
                  verbose=False)
    xgb_preds += xgb_model.predict_proba(X_test_xgb) / NFOLDS
    
    # --- CatBoost ---
    print("Training CatBoost...")
    cat_model = ctb.CatBoostClassifier(**CAT_PARAMS)
    cat_model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=100,
                  use_best_model=True, # Recommended with early stopping
                  verbose=0)
    cat_preds += cat_model.predict_proba(X_test) / NFOLDS

print("Model training complete.")


print("\n--- Prediction and Submission ---")


print("Ensembling predictions from all models...")
ensemble_preds = (lgb_preds + xgb_preds + cat_preds) / 3.0


top_3_indices = np.argsort(ensemble_preds, axis=1)[:, ::-1][:, :3]


predicted_fertilizer_names_list = []
for row_indices in top_3_indices:
    names = le.inverse_transform(row_indices)
    predicted_fertilizer_names_list.append(" ".join(names))


submission_df = pd.DataFrame({'id': test_ids, 'Fertilizer Name': predicted_fertilizer_names_list})


print("\nSubmission DataFrame head:")
print(submission_df.head())


submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"\nSubmission file '{submission_file}' created successfully.")


print("\nScript finished.")

