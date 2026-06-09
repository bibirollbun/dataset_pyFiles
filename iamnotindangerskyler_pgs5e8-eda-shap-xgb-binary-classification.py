import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, classification_report, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import shap
from xgboost import XGBClassifier
import xgboost as xgb
import gc
import torch
from itertools import combinations


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Keep original ID columns
train_id = train['id'].copy()
test_id = test['id'].copy()

print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

print("\nðŸ“Š Missing values in training set:")
print(train.isna().sum())

print("\nðŸ“Š Missing values in test set:")
print(test.isna().sum())

print("\nðŸ“Š Statistical summary of training data:")
display(train.describe().T)

train_data = train.drop(columns=['id'])
test_data = test.drop(columns=['id'])

# Define categorical and target columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
target_column = 'y'

numeric_columns = [col for col in train_data.columns if col not in categorical_columns + [target_column]]

print(f"\n Number of numeric features: {len(numeric_columns)}")
print(f" Number of categorical features: {len(categorical_columns)}")
print(f" Target column: {target_column}")
print(f" Numeric columns: {numeric_columns}")
print(f" Categorical columns: {categorical_columns}")


# Check unique values in education column
print(" Unique values in 'education' (train_data):")
print(train_data['education'].value_counts(dropna=False))
print("\n Unique values in 'education' (test_data):")
print(test_data['education'].value_counts(dropna=False))

# Define one-hot encoding columns (excluding education)
one_hot_columns = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day']

# Define ordinal encoding for education
education_mapping = {
    'primary': 1,
    'secondary': 2,
    'tertiary': 3,
    'unknown': 0  # Treat unknown as a separate category (not part of ordinal scale)
}

# Define columns to scale
columns_to_scale = ['age', 'balance', 'campaign', 'pdays', 'previous', 'duration']

# Function to preprocess data (encoding + scaling)
def preprocess_data(train_df, test_df, one_hot_cols, education_map, scale_cols, target_col='y'):
    # Create copies to avoid modifying original data
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Handle missing values in education and create education_unknown column
    train_processed['education_unknown'] = train_processed['education'].isna() | (train_processed['education'] == 'unknown')
    test_processed['education_unknown'] = test_processed['education'].isna() | (test_processed['education'] == 'unknown')
    train_processed['education_unknown'] = train_processed['education_unknown'].astype(int)
    test_processed['education_unknown'] = test_processed['education_unknown'].astype(int)
    
    # Fill NaN in education with 'unknown' before mapping
    train_processed['education'] = train_processed['education'].fillna('unknown')
    test_processed['education'] = test_processed['education'].fillna('unknown')
    
    # Ordinal encoding for education
    def map_education(x):
        return education_map.get(x, 0)
    train_processed['education'] = train_processed['education'].apply(map_education)
    test_processed['education'] = test_processed['education'].apply(map_education)
    
    # One-hot encoding for specified columns
    train_processed = pd.get_dummies(train_processed, columns=one_hot_cols, prefix=one_hot_cols, drop_first=True)
    test_processed = pd.get_dummies(test_processed, columns=one_hot_cols, prefix=one_hot_cols, drop_first=True)
    
    # Ensure test set has the same one-hot encoded columns as train set (excluding target)
    feature_columns = [col for col in train_processed.columns if col != target_col]
    missing_cols = set(feature_columns) - set(test_processed.columns)
    for col in missing_cols:
        if any(prefix in col for prefix in one_hot_cols):
            test_processed[col] = 0
    
    # Reorder test columns to match train's feature columns (excluding target)
    test_processed = test_processed[feature_columns]
    
    # Min-Max scaling for specified numeric columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_processed[scale_cols] = scaler.fit_transform(train_processed[scale_cols])
    test_processed[scale_cols] = scaler.transform(test_processed[scale_cols])
    
    return train_processed, test_processed, scaler

# Apply preprocessing
train_data_processed, test_data_processed, scaler = preprocess_data(
    train_data, test_data, one_hot_columns, education_mapping, columns_to_scale, target_column
)


# Create copies for feature engineering
train_data_fe = train_data_processed.copy()
test_data_fe = test_data_processed.copy()

# Numeric columns (already scaled)
numeric_columns = ['age', 'balance', 'campaign', 'pdays', 'previous', 'duration']

# Categorical columns (after encoding, we'll use original categorical columns for interactions)
original_categorical_columns = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day', 'education']

# 1. Polynomial Features (2nd Degree)
print("ðŸ“Š Generating 2nd Degree Polynomial Features...")
# Squares
for col in numeric_columns:
    train_data_fe[f'{col}_Squared'] = train_data_fe[col] ** 2
    test_data_fe[f'{col}_Squared'] = test_data_fe[col] ** 2
    print(f"âœ“ {col}_Squared created")

# Cross products
for col1, col2 in combinations(numeric_columns, 2):
    feature_name = f'{col1}_{col2}_Interaction'
    train_data_fe[feature_name] = train_data_fe[col1] * train_data_fe[col2]
    test_data_fe[feature_name] = test_data_fe[col1] * test_data_fe[col2]
    print(f"âœ“ {feature_name} created")

# Example interaction: age * duration * campaign
train_data_fe['Age_Duration_Campaign_Interaction'] = train_data_fe['age'] * train_data_fe['duration'] * train_data_fe['campaign']
test_data_fe['Age_Duration_Campaign_Interaction'] = test_data_fe['age'] * test_data_fe['duration'] * test_data_fe['campaign']
print("âœ“ Age_Duration_Campaign_Interaction created")

# 2. Ratio Features
print("\nðŸ“Š Generating Ratio Features...")
epsilon = 1e-6  # To prevent division by zero
for col1, col2 in combinations(numeric_columns, 2):
    feature_name = f'{col1}_{col2}_Ratio'
    train_data_fe[feature_name] = train_data_fe[col1] / (train_data_fe[col2] + epsilon)
    test_data_fe[feature_name] = test_data_fe[col1] / (test_data_fe[col2] + epsilon)
    print(f"âœ“ {feature_name} created")

# 3. Categorical-Numerical Interactions
print("\nðŸ“Š Generating Categorical-Numerical Interaction Features...")
for cat_col in original_categorical_columns:
    if cat_col in train_data.columns:  # Use original categorical columns from train_data
        for num_col in numeric_columns:
            feature_name = f'{num_col}_by_{cat_col}'
            group_means = train_data.groupby(cat_col)[num_col].mean()
            train_data_fe[feature_name] = train_data[cat_col].map(group_means)
            test_data_fe[feature_name] = test_data[cat_col].map(group_means).fillna(group_means.mean())
            print(f"âœ“ {feature_name} created")

# 4. Categorical Combinations
print("\nðŸ“Š Generating Categorical Combination Features...")
train_data_fe['Job_Marital_Interaction'] = train_data['job'].astype(str) + "_" + train_data['marital'].astype(str)
test_data_fe['Job_Marital_Interaction'] = test_data['job'].astype(str) + "_" + test_data['marital'].astype(str)

# Label Encoding for Job_Marital_Interaction
le_interaction = LabelEncoder()
train_data_fe['Job_Marital_Interaction'] = le_interaction.fit_transform(train_data_fe['Job_Marital_Interaction'])
le_interaction.classes_ = np.append(le_interaction.classes_, 'unknown')
test_data_fe['Job_Marital_Interaction'] = test_data_fe['Job_Marital_Interaction'].map(
    lambda x: x if x in le_interaction.classes_[:-1] else 'unknown'
)
test_data_fe['Job_Marital_Interaction'] = le_interaction.transform(test_data_fe['Job_Marital_Interaction'])
print("âœ“ Job_Marital_Interaction created")

# 5. High Correlation Check and Removal
print("\nðŸ“Š High Correlation Check...")
print("Selecting features with 98.5% or higher correlation:")
new_numeric_columns = [col for col in train_data_fe.columns 
                       if train_data_fe[col].dtype in ['int64', 'float64'] 
                       and col not in ['id', 'y']]
correlation_matrix = train_data_fe[new_numeric_columns].corr()
high_corr_pairs = []
to_drop = set()
threshold = 0.985  # Correlation threshold

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) >= threshold:
            feat1, feat2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
            # Select the feature with the longer name to drop
            to_drop.add(feat2 if len(feat2) > len(feat1) else feat1)
            high_corr_pairs.append((feat1, feat2, corr_val))

if high_corr_pairs:
    existing_cols_to_drop = [col for col in to_drop if col in train_data_fe.columns]
    train_data_fe.drop(columns=existing_cols_to_drop, inplace=True)
    test_data_fe.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Removed highly correlated features: {existing_cols_to_drop}")
else:
    print("No feature pairs found with |correlation| >= 0.9.")

# 6. Logarithmic Transformations (for skewed columns)
print("\nðŸ“Š Generating Logarithmic Transformation Features...")
for col in numeric_columns:
    # Check skewness on original data (before scaling)
    if abs(skew(train_data[col].dropna())) > 0.5:  # Skewness threshold
        train_data_fe[f'Log_{col}'] = np.log1p(train_data[col].clip(lower=0))
        test_data_fe[f'Log_{col}'] = np.log1p(test_data[col].clip(lower=0))
        print(f"âœ“ Log_{col} created")

# Remaining features
remaining_features = [col for col in train_data_fe.columns 
                     if col not in ['id', 'y']]
print(f"\nRemaining feature count and names: {len(remaining_features)} {remaining_features}")

# Newly created columns
new_columns = [col for col in train_data_fe.columns if col not in train_data.columns]
print(f"\nNewly created columns: {new_columns}")

# Verify final datasets
print("\nðŸ“Š Training data after feature engineering:")
print(f"Shape: {train_data_fe.shape}")
print("\nFirst few rows of feature-engineered training data:")
display(train_data_fe.head())

print("\nðŸ“Š Test data after feature engineering:")
print(f"Shape: {test_data_fe.shape}")
print("\nFirst few rows of feature-engineered test data:")
display(test_data_fe.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import shap

# Print status message
print("Starting XGBoost Model Training with Fuzzy Predictions...")

# Display dataset information
print("train_data_fe shape:", train_data_fe.shape)

# Separate features and target
features = [col for col in train_data_fe.columns if col not in ['y', 'y_encoded']]
X = train_data_fe[features]
y = train_data_fe['y']
X_test = test_data_fe[features]

# Split data into train and test sets (0.2 test size) for feature selection
print("\nSplitting data for feature selection (0.2 test size)...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train an initial XGBoost model for SHAP feature importance
print("Training initial XGBoost model for SHAP feature selection...")
initial_model = XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    device='cuda',
    n_estimators=1000,  # Reduced for faster feature selection
    random_state=42,
    n_jobs=1
)
initial_model.fit(X_train, y_train)

# Calculate SHAP values
print("Calculating SHAP values for feature selection...")
explainer = shap.TreeExplainer(initial_model)
shap_values = explainer.shap_values(X_train)

# Compute mean absolute SHAP values for each feature
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': shap_importance
}).sort_values(by='importance', ascending=False)

# Select top 40 features
top_40_features = feature_importance['feature'].head(50).tolist()
print(f"Selected top 40 features: {top_40_features}")


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
from tqdm import tqdm

# Filter datasets to include only top 40 features
X = X[top_40_features]
X_test = X_test[top_40_features]
print(f"Shape of X after feature selection: {X.shape}")
print(f"Shape of X_test after feature selection: {X_test.shape}")

# Define CatBoost hyperparameters from provided Optuna trial
best_params = {
    'iterations': 2699,
    'depth': 7,
    'learning_rate': 0.03992667039524374,
    'subsample': 0.9014399242911059,
    'min_data_in_leaf': 41,
    'l2_leaf_reg': 5.2632940602541245,
    'task_type': 'CPU',  # Use CPU to avoid colsample_bylevel GPU issue
    'random_seed': 42,
    'eval_metric': 'Logloss',
    'bootstrap_type': 'Bernoulli',  # Support subsample
    'verbose': 0
}

# Perform 5-fold stratified cross-validation to generate OOF predictions
print("\nStarting 5-Fold Stratified K-Fold Cross-Validation for OOF Predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
mcc_scores = []
acc_scores = []
auc_scores = []
feature_importances = []
oof_probs = np.zeros(len(X))  # Array to store OOF predictions

for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Cross-Validation Folds")):
    print(f"\nTraining Fold {fold + 1}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train CatBoost model
    model = CatBoostClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    # Predict on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)

    # Store OOF predictions
    oof_probs[val_idx] = val_probs

    # Calculate metrics
    f1 = f1_score(y_val, val_preds)
    mcc = matthews_corrcoef(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)

    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    auc_scores.append(auc)

    # Store feature importance
    feature_importances.append(model.get_feature_importance())

    print(f"Fold {fold + 1} Results:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# Print cross-validation results
print("\nCross-Validation Results Summary:")
print(f"Average F1 Score: {np.mean(f1_scores):.4f} (Â±{np.std(f1_scores):.4f})")
print(f"Average MCC: {np.mean(mcc_scores):.4f} (Â±{np.std(mcc_scores):.4f})")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} (Â±{np.std(acc_scores):.4f})")
print(f"Average AUC-ROC: {np.mean(auc_scores):.4f} (Â±{np.std(auc_scores):.4f})")

# Add OOF predictions as a new feature to the training dataset
train_data_fe['oof_cb'] = oof_probs
print("\nAdded 'oof_cb' column to training dataset.")
print("First 5 rows of training dataset with 'oof_cb':")
print(train_data_fe[['oof_cb']].head())

# Save the updated training dataset with OOF predictions
train_data_fe.to_csv('train_data_with_oof_cb.csv', index=False)
print("\nTraining dataset with 'oof_cb' saved to 'train_data_with_oof_cb.csv'.")

# Train final model on full data
print("\nTraining Final CatBoost Model on Full Data...")
final_model = CatBoostClassifier(**best_params)
final_model.fit(X, y, verbose=False)

# Predict probabilities on test set
print("Making Fuzzy Predictions on Test Set...")
test_probs_cb = final_model.predict_proba(X_test)[:, 1]  # Probability for positive class

# Add test predictions to test dataset
test_data_fe = X_test.copy()  # Assume X_test is a DataFrame
test_data_fe['test_cb'] = test_probs_cb
print("\nAdded 'test_cb' column to test dataset.")
print("First 5 rows of test dataset with 'test_cb':")
print(test_data_fe[['test_cb']].head())

# Save updated test dataset
test_data_fe.to_csv('test_data_with_cb_predictions.csv', index=False)
print("\nTest dataset with 'test_cb' saved to 'test_data_with_cb_predictions.csv'.")


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
from tqdm import tqdm

# Filter datasets to include only top 40 features
X = X[top_40_features]
X_test = X_test[top_40_features]
print(f"Shape of X after feature selection: {X.shape}")
print(f"Shape of X_test after feature selection: {X_test.shape}")

# Define LightGBM hyperparameters from provided Optuna trial
best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device_type': 'gpu',  # Use GPU if available; change to 'cpu' if not
    'n_estimators': 2299,
    'max_depth': 12,
    'learning_rate': 0.0428676976128519,
    'subsample': 0.9931752867330768,
    'colsample_bytree': 0.5743629790143214,
    'min_child_weight': 5.772962476311002,
    'reg_alpha': 0.9860793622818897,
    'reg_lambda': 0.9725793102816362,
    'num_leaves': 82,
    'random_state': 42,
    'verbose': -1
}

# Perform 5-fold stratified cross-validation to generate OOF predictions
print("\nStarting 5-Fold Stratified K-Fold Cross-Validation for OOF Predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
mcc_scores = []
acc_scores = []
auc_scores = []
feature_importances = []
oof_probs_lgbm = np.zeros(len(X))  # Array to store OOF predictions

for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Cross-Validation Folds")):
    print(f"\nTraining Fold {fold + 1}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train LightGBM model
    model = LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Predict on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)

    # Store OOF predictions
    oof_probs_lgbm[val_idx] = val_probs

    # Calculate metrics
    f1 = f1_score(y_val, val_preds)
    mcc = matthews_corrcoef(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)

    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    auc_scores.append(auc)

    # Store feature importance
    feature_importances.append(model.feature_importances_)

    print(f"Fold {fold + 1} Results:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# Print cross-validation results
print("\nCross-Validation Results Summary:")
print(f"Average F1 Score: {np.mean(f1_scores):.4f} (Â±{np.std(f1_scores):.4f})")
print(f"Average MCC: {np.mean(mcc_scores):.4f} (Â±{np.std(mcc_scores):.4f})")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} (Â±{np.std(acc_scores):.4f})")
print(f"Average AUC-ROC: {np.mean(auc_scores):.4f} (Â±{np.std(auc_scores):.4f})")

# Add OOF predictions as a new feature to the training dataset
train_data_fe['oof_lgbm'] = oof_probs_lgbm
print("\nAdded 'oof_lgbm' column to training dataset.")
print("First 5 rows of training dataset with 'oof_lgbm':")
print(train_data_fe[['oof_lgbm']].head())

# Save the updated training dataset with OOF predictions
train_data_fe.to_csv('train_data_with_oof_lgbm.csv', index=False)
print("\nTraining dataset with 'oof_lgbm' saved to 'train_data_with_oof_lgbm.csv'.")

# Train final model on full data
print("\nTraining Final LightGBM Model on Full Data...")
final_model = LGBMClassifier(**best_params)
final_model.fit(X, y)

# Predict probabilities on test set
print("Making Fuzzy Predictions on Test Set...")
test_probs_lgbm = final_model.predict_proba(X_test)[:, 1]  # Probability for positive class

# Add test predictions to test dataset
test_data_fe = X_test.copy()  # Assume X_test is a DataFrame
test_data_fe['test_lgbm'] = test_probs_lgbm
print("\nAdded 'test_lgbm' column to test dataset.")
print("First 5 rows of test dataset with 'test_lgbm':")
print(test_data_fe[['test_lgbm']].head())

# Save updated test dataset
test_data_fe.to_csv('test_data_with_lgbm_predictions.csv', index=False)
print("\nTest dataset with 'test_lgbm' saved to 'test_data_with_lgbm_predictions.csv'.")


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

# Filter datasets to include only top 40 features
X = X[top_40_features]
X_test = X_test[top_40_features]
print(f"Shape of X after feature selection: {X.shape}")
print(f"Shape of X_test after feature selection: {X_test.shape}")

# Define XGBoost hyperparameters from provided trial
best_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',  # Use GPU if available; change to 'cpu' if not
    'n_estimators': 2500,
    'max_depth': 9,
    'learning_rate': 0.03890246184207177,
    'subsample': 0.9172074929597854,
    'colsample_bytree': 0.6782475433828932,
    'min_child_weight': 8,
    'gamma': 0.015406194625565472,
    'reg_alpha': 0.15620343000892567,
    'reg_lambda': 0.6158107348748302,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': 1,
    'enable_categorical': False
}

# Perform 5-fold stratified cross-validation to generate OOF predictions
print("\nStarting 5-Fold Stratified K-Fold Cross-Validation for OOF Predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
mcc_scores = []
acc_scores = []
auc_scores = []
feature_importances = []
oof_probs_xgb = np.zeros(len(X))  # Array to store OOF predictions

for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Cross-Validation Folds")):
    print(f"\nTraining Fold {fold + 1}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train XGBoost model
    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    # Predict on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)

    # Store OOF predictions
    oof_probs_xgb[val_idx] = val_probs

    # Calculate metrics
    f1 = f1_score(y_val, val_preds)
    mcc = matthews_corrcoef(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)

    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    auc_scores.append(auc)

    # Store feature importance
    feature_importances.append(model.feature_importances_)

    print(f"Fold {fold + 1} Results:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# Print cross-validation results
print("\nCross-Validation Results Summary:")
print(f"Average F1 Score: {np.mean(f1_scores):.4f} (Â±{np.std(f1_scores):.4f})")
print(f"Average MCC: {np.mean(mcc_scores):.4f} (Â±{np.std(mcc_scores):.4f})")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} (Â±{np.std(acc_scores):.4f})")
print(f"Average AUC-ROC: {np.mean(auc_scores):.4f} (Â±{np.std(auc_scores):.4f})")

# Add OOF predictions as a new feature to the training dataset
train_data_fe['oof_xgb'] = oof_probs_xgb
print("\nAdded 'oof_xgb' column to training dataset.")
print("First 5 rows of training dataset with 'oof_xgb':")
print(train_data_fe[['oof_xgb']].head())

# Save the updated training dataset with OOF predictions
train_data_fe.to_csv('train_data_with_oof_xgb.csv', index=False)
print("\nTraining dataset with 'oof_xgb' saved to 'train_data_with_oof_xgb.csv'.")

# Train final model on full data
print("\nTraining Final XGBoost Model on Full Data...")
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# Predict probabilities on test set
print("Making Fuzzy Predictions on Test Set...")
test_probs_xgb = final_model.predict_proba(X_test)[:, 1]  # Probability for positive class

# Add test predictions to test dataset
test_data_fe = X_test.copy()  # Assume X_test is a DataFrame
test_data_fe['test_xgb'] = test_probs_xgb
print("\nAdded 'test_xgb' column to test dataset.")
print("First 5 rows of test dataset with 'test_xgb':")
print(test_data_fe[['test_xgb']].head())

# Save updated test dataset
test_data_fe.to_csv('test_data_with_xgb_predictions.csv', index=False)
print("\nTest dataset with 'test_xgb' saved to 'test_data_with_xgb_predictions.csv'.")


import pandas as pd
import numpy as np

# Prepare training data with OOF predictions
train_data_fe['oof_cb'] = oof_probs
train_data_fe['oof_lgbm'] = oof_probs_lgbm
train_data_fe['oof_xgb'] = oof_probs_xgb
print("\nAdded OOF columns ('oof_cb', 'oof_lgbm', 'oof_xgb') to training dataset.")

# Prepare test data with final predictions
test_data_fe = X_test.copy()
test_data_fe['test_cb'] = test_probs_cb
test_data_fe['test_lgbm'] = test_probs_lgbm
test_data_fe['test_xgb'] = test_probs_xgb
print("\nAdded test prediction columns ('test_cb', 'test_lgbm', 'test_xgb') to test dataset.")

# Save datasets
train_data_fe.to_csv('/kaggle/working/train_data_with_oof_all.csv', index=False)
test_data_fe.to_csv('/kaggle/working/test_data_with_predictions_all.csv', index=False)
print("\nTraining dataset saved to '/kaggle/working/train_data_with_oof_all.csv'.")
print("\nTest dataset saved to '/kaggle/working/test_data_with_predictions_all.csv'.")

# Print first 5 rows of final train and test datasets
print("\nFirst 5 rows of training dataset (train_data_fe):")
print(train_data_fe.head())
print("\nFirst 5 rows of test dataset (test_data_fe):")
print(test_data_fe.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# XGBoost hyperparameters
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',  # Change to 'cpu' if GPU is not available
    'n_estimators': 2500,
    'max_depth': 9,
    'learning_rate': 0.03890246184207177,
    'subsample': 0.9172074929597854,
    'colsample_bytree': 0.6782475433828932,
    'min_child_weight': 8,
    'gamma': 0.015406194625565472,
    'reg_alpha': 0.15620343000892567,
    'reg_lambda': 0.6158107348748302,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': 1,
    'enable_categorical': False
}

# SHAP analysis with all features (excluding target 'y')
print("\nPerforming SHAP analysis with all features...")
# Ensure 'y' is not included in features
X_all = train_data_fe.drop(columns=['y'], errors='ignore').copy()  # Remove 'y' if present
y_all = y.copy()

# Train model with all features
print("Training XGBoost model with all features...")
model_all = XGBClassifier(**xgb_params)
model_all.fit(X_all, y_all)

# Calculate SHAP values
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model_all)
shap_values = explainer.shap_values(X_all.sample(1000, random_state=42))  # Sample for speed
shap_df = pd.DataFrame({
    'feature': X_all.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

# SHAP summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_all.sample(1000, random_state=42), feature_names=X_all.columns, show=False)
plt.savefig('/kaggle/working/shap_summary_all_features.png', bbox_inches='tight')
plt.close()
print("SHAP summary plot saved as '/kaggle/working/shap_summary_all_features.png'.")

# Save SHAP values
shap_df.to_csv('/kaggle/working/shap_values_all_features.csv', index=False)
print("\nSHAP values saved as '/kaggle/working/shap_values_all_features.csv'.")
print("Top 5 most important features (based on SHAP values):")
print(shap_df.head())

# Select top features
print("\nSelecting top features...")
top_features = shap_df['feature'].tolist()  # Features sorted by SHAP values
n_features_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
results = []

# Evaluate model with different numbers of features
print("\nTesting XGBoost model with different feature counts...")
for n_features in n_features_list:
    selected_features = top_features[:n_features]  # Top n features
    print(f"\nEvaluating with {n_features} features: {selected_features}")

    # Prepare data with selected features
    X_stack = train_data_fe[selected_features]
    y_stack = y

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_stack, y_stack), total=5, desc=f"CV Folds ({n_features} features)")):
        X_train, X_val = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
        y_train, y_val = y_stack.iloc[train_idx], y_stack.iloc[val_idx]

        # Train XGBoost model
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )

        # Calculate ROC-AUC
        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        auc_scores.append(auc)

    # Save results
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    results.append({
        'n_features': n_features,
        'features': selected_features,
        'mean_auc': mean_auc,
        'std_auc': std_auc
    })
    print(f"Results for {n_features} features: Mean AUC = {mean_auc:.4f} (Â±{std_auc:.4f})")

# Find the best feature combination
best_result = max(results, key=lambda x: x['mean_auc'])
best_n_features = best_result['n_features']
best_features = best_result['features']
print(f"\nBest feature combination: {best_n_features} features")
print(f"Features: {best_features}")
print(f"Mean AUC: {best_result['mean_auc']:.4f} (Â±{best_result['std_auc']:.4f})")

# Train final model with best features
print("\nTraining final XGBoost model with best features...")
X_stack_final = train_data_fe[best_features]
y_stack_final = y
final_model = XGBClassifier(**xgb_params)
final_model.fit(X_stack_final, y_stack_final)

# Make predictions on test set
print("\nMaking predictions on test set...")
# Map training features to test features
test_features = []
for f in best_features:
    if 'oof_' in f:
        test_f = f.replace('oof_', 'test_')
    else:
        test_f = f  # Keep non-oof features as is
    if test_f not in test_data_fe.columns:
        print(f"Warning: Feature '{test_f}' not found in test_data_fe. Available columns: {test_data_fe.columns}")
        raise KeyError(f"Feature '{test_f}' not in test_data_fe")
    test_features.append(test_f)

X_test_stack = test_data_fe[test_features]
test_probs = final_model.predict_proba(X_test_stack)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_id,
    'y': test_probs
})
submission.to_csv('/kaggle/working/submission_xgb_stacking.csv', index=False)
print("\nSubmission file saved as '/kaggle/working/submission_xgb_stacking.csv'.")
print("First 5 rows of submission file:")
print(submission.head())

