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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission_df=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
original_df=pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


print(train_df.shape)
print(test_df.shape)
print(original_df.shape)


print(train_df.columns)
print(original_df.columns)


## drop id from train df 
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

print(train_df.columns)
print(original_df.columns)


# Filter only the columns that are in train_df
common_columns = train_df.columns.intersection(original_df.columns)
original_df = original_df[common_columns]

# Concatenate and drop duplicates
combined_df = pd.concat([train_df, original_df], ignore_index=True)
combined_df.drop_duplicates(inplace=True)

print("Combined Data Shape (after removing duplicates):", combined_df.shape)



print(combined_df.columns)


target_col = 'Fertilizer Name'

# Categorical columns excluding the target
categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

# Now encode features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    test_df[col] = le.transform(test_df[col])  # same encoder!
    label_encoders[col] = le

# Encode the target separately
target_le = LabelEncoder()
target = target_le.fit_transform(combined_df[target_col])
label_encoders['target'] = target_le



for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=combined_df, x=col, order=combined_df[col].value_counts().index, palette='Set2')
    plt.title(f'Count Plot of {col}', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



numerical_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(6, 6))
    
    # Get value counts
    counts = combined_df[col].value_counts()
    
    # Create pie chart
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.tight_layout()
    plt.show()



for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(combined_df[col], kde=True, color='red', bins=30)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import LabelEncoder

# Fit on combined_df, transform both combined_df and test_df
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    test_df[col] = le.transform(test_df[col])  # Use same encoder!
    label_encoders[col] = le



target_col = 'Fertilizer Name'
features = combined_df.drop(columns=[target_col])
target_raw = combined_df[target_col]

from sklearn.preprocessing import LabelEncoder
target_le = LabelEncoder()
target = target_le.fit_transform(target_raw)

label_encoders['target'] = target_le




# ---- Step 4: Standardize Features Only ----
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# ---- Step 5: Combine Scaled Features and Encoded Target ----
combined_df_scaled = pd.concat([features_scaled, pd.Series(target, name=target_col)], axis=1)



test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)



for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(combined_df[col], kde=True)
    plt.title(f'Distribution of {col} after scaling')
    plt.show()



corr_matrix = combined_df_scaled.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Feature Correlation Matrix')
plt.show()


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd

params = {
    'booster': 'gbtree',
    'lambda': 0.0010064935583862765,
    'alpha': 3.1766322669357034,
    'colsample_bytree': 0.42885923660896164,
    'subsample': 0.9886300935357404,
    'learning_rate': 0.2987215639008042,
    'max_depth': 10,
    'min_child_weight': 6,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42,
    'num_class': len(target_le.classes_)
}

num_classes = params['num_class']
FOLDS = 20
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# MAP@3 metric
def map3(actual, predicted_proba, k=3):
    top_k_preds = np.argsort(predicted_proba, axis=1)[:, ::-1][:, :k]
    score = 0.0
    for i in range(len(actual)):
        if actual[i] in top_k_preds[i]:
            rank = np.where(top_k_preds[i] == actual[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(actual)

# Out-of-fold predictions for training data and aggregate predictions for test data
oof = np.zeros((len(X), num_classes))
pred_prob = np.zeros((len(test_df), num_classes))
xgb_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10, verbose=0)

    val_proba = model.predict_proba(x_valid)
    oof[valid_idx] = val_proba
    pred_prob += model.predict_proba(test_df)

    fold_map3 = map3(y_valid.values, val_proba)
    xgb_scores.append(fold_map3)
    print(f"Fold {fold} MAP@3: {fold_map3:.4f}")

print(f"\nAverage StratifiedKFold MAP@3: {np.mean(xgb_scores):.6f}")

# Final predictions for test set (average of folds)
pred_prob /= FOLDS
# Top 3 predictions
top_3_preds = np.argsort(pred_prob, axis=1)[:, ::-1][:, :3]

# Decode back to Fertilizer Names
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Build submission
submission = pd.DataFrame({
    'id': sample_submission_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission saved successfully.")



print(submission.head())
print(submission['Fertilizer Name'].head().tolist())


%%time

import os
import warnings
warnings.filterwarnings("ignore")

import gc 
import numpy as np
import pandas as pd
import random as python_random
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

# Set all seeds immediately
np.random.seed(42)
python_random.seed(42)

# --- Load Data ---
df_train_original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train_additional = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print("Original Train data loaded. Shape:", df_train_original.shape)
print("Additional Train data loaded. Shape:", df_train_additional.shape)
print("Test data loaded. Shape:", df_test.shape)

# --- Define features and target ---
TARGET = 'Fertilizer Name'
ID_COL = 'id'

# Separate features (X) and target (y) for original and additional datasets
X_original = df_train_original.drop([ID_COL, TARGET], axis=1).copy()
y_original = df_train_original[TARGET].copy()

X_additional = df_train_additional.drop(TARGET, axis=1).copy()
y_additional = df_train_additional[TARGET].copy()

X_test = df_test.drop(ID_COL, axis=1).copy()
test_ids = df_test[ID_COL].copy()


# --- Feature Engineering ---
print("\nPerforming Feature Engineering...")

original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Function to apply feature engineering
def apply_feature_engineering(df):
    df_copy = df.copy() 
    df_copy['Temp_Humidity_Interaction'] = df_copy['Temparature'] * df_copy['Humidity']
    df_copy['N_P_Ratio'] = df_copy['Nitrogen'] / (df_copy['Phosphorous'].replace(0, 1e-6))
    df_copy['K_P_Ratio'] = df_copy['Potassium'] / (df_copy['Phosphorous'].replace(0, 1e-6))
    df_copy['Soil_Crop_Combination'] = df_copy['Soil Type'].astype(str) + '_' + df_copy['Crop Type'].astype(str)

    # Binning numerical features (as strings for categorical handling)
    for col in original_numerical_cols:
        df_copy[f'{col}_Binned'] = df_copy[col].astype(str)

    return df_copy

# Apply FE to all datasets
X_original_fe = apply_feature_engineering(X_original)
X_additional_fe = apply_feature_engineering(X_additional)
X_test_fe = apply_feature_engineering(X_test)

print("Feature Engineering complete.")

# Define common feature lists after FE (must be consistent across all X's)
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
                      'Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination']
# Add binned columns, ensuring they are derived from original_numerical_cols consistently
categorical_features.extend([f'{col}_Binned' for col in original_numerical_cols])


# Polynomial Features (fit on original train data, transform all)
poly_features_to_transform = original_numerical_cols
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit on original training numerical features
X_original_poly_transformed = poly.fit_transform(X_original_fe[poly_features_to_transform])
X_additional_poly_transformed = poly.transform(X_additional_fe[poly_features_to_transform])
X_test_poly_transformed = poly.transform(X_test_fe[poly_features_to_transform])

poly_feature_names = poly.get_feature_names_out(poly_features_to_transform)

X_original_fe = X_original_fe.drop(columns=poly_features_to_transform)
X_additional_fe = X_additional_fe.drop(columns=poly_features_to_transform)
X_test_fe = X_test_fe.drop(columns=poly_features_to_transform)

X_original_fe = pd.concat([X_original_fe, pd.DataFrame(X_original_poly_transformed, columns=poly_feature_names, index=X_original_fe.index)], axis=1)
X_additional_fe = pd.concat([X_additional_fe, pd.DataFrame(X_additional_poly_transformed, columns=poly_feature_names, index=X_additional_fe.index)], axis=1)
X_test_fe = pd.concat([X_test_fe, pd.DataFrame(X_test_poly_transformed, columns=poly_feature_names, index=X_test_fe.index)], axis=1)

# Update numerical features list to include polynomial ones
numerical_features.extend(poly_feature_names)

# Ensure unique and consistent order for all feature lists
numerical_features = list(dict.fromkeys(numerical_features))
categorical_features = list(dict.fromkeys(categorical_features))
numerical_features = [f for f in numerical_features if f not in categorical_features]
all_features_ordered = numerical_features + categorical_features

X_original_fe = X_original_fe[all_features_ordered].copy()
X_additional_fe = X_additional_fe[all_features_ordered].copy()
X_test_fe = X_test_fe[all_features_ordered].copy()


# --- Convert categorical features to 'category' dtype for XGBoost ---
print("\nConverting categorical features to 'category' dtype for XGBoost internal handling...")
# Collect all unique categories across ALL datasets for consistent encoding
all_categories_union = {}
for col in categorical_features:
    if col in X_original_fe.columns:
        all_categories_union[col] = pd.concat([
            X_original_fe[col],
            X_additional_fe[col],
            X_test_fe[col]
        ], axis=0).astype(str).unique()
    else:
        print(f"Warning: Categorical column '{col}' not found after feature engineering. Skipping conversion.")

for col in categorical_features:
    if col in X_original_fe.columns:
        X_original_fe[col] = pd.Categorical(X_original_fe[col], categories=all_categories_union[col])
        X_additional_fe[col] = pd.Categorical(X_additional_fe[col], categories=all_categories_union[col])
        X_test_fe[col] = pd.Categorical(X_test_fe[col], categories=all_categories_union[col])

print("Categorical feature conversion complete.")
print("Processed X_original_fe shape:", X_original_fe.shape)
print("Processed X_additional_fe shape:", X_additional_fe.shape)
print("Processed X_test_fe shape:", X_test_fe.shape)


# --- Target Encoding ---
label_encoder = LabelEncoder()
# Fit on the union of all training targets
y_encoded_all_train = label_encoder.fit_transform(pd.concat([y_original, y_additional]).values)
y_original_encoded = label_encoder.transform(y_original.values)
y_additional_encoded = label_encoder.transform(y_additional.values) 

fertilizer_classes = label_encoder.classes_
print("\nTarget encoding complete. Fertilizer classes (order):", fertilizer_classes)
print(f"Number of classes: {len(fertilizer_classes)}")


# --- XGBoost Model Training with 5-Fold Cross-Validation ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_original_fe), len(fertilizer_classes)))
test_preds_list = []

print(f"\nStarting {N_SPLITS}-Fold Cross-Validation for XGBoost...")

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(fertilizer_classes),
    'eval_metric': 'mlogloss',
    'eta': 0.01,
    'max_depth': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'n_estimators': 10000,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'verbose': 0,
    'enable_categorical': True
}


for fold, (train_idx_original, val_idx_original) in enumerate(skf.split(X_original_fe, y_original_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    X_train_fold_original = X_original_fe.iloc[train_idx_original]
    y_train_fold_original = y_original_encoded[train_idx_original]

    X_val_fold = X_original_fe.iloc[val_idx_original]
    y_val_fold = y_original_encoded[val_idx_original]

    # Combine current fold's training data with the entire additional dataset
    X_train_final = pd.concat([X_train_fold_original, X_additional_fe], ignore_index=True)
    y_train_final = np.concatenate([y_train_fold_original, y_additional_encoded])

    print(f"Fold {fold+1} training data shape (original + additional): {X_train_final.shape}")
    print(f"Fold {fold+1} validation data shape (original only): {X_val_fold.shape}")

    model = XGBClassifier(**xgb_params)

    model.fit(X_train_final, y_train_final, 
              eval_set=[(X_val_fold, y_val_fold)], 
              verbose=1000
             )

    oof_preds[val_idx_original] = model.predict_proba(X_val_fold)
    test_preds_list.append(model.predict_proba(X_test_fe)) 

    del X_train_final, y_train_final, X_train_fold_original, y_train_fold_original, X_val_fold, y_val_fold, model
    gc.collect()

print("\nCross-validation complete.")


# --- MAP@3 Calculation (on OOF predictions) ---
def apk(actual, predicted, k=3):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

y_true_labels_for_map = [[label] for label in y_original.values]

oof_ranked_labels = []
for i in range(len(oof_preds)):
    top_3_indices = np.argsort(oof_preds[i])[-3:][::-1]
    oof_ranked_labels.append([fertilizer_classes[idx] for idx in top_3_indices])

print("\nCalculating OOF MAP@3 score...")
oof_map3_score = mapk(y_true_labels_for_map, oof_ranked_labels, k=3)
print(f"Overall OOF MAP@3: {oof_map3_score:.5f}")


# --- Generate Submission File ---
final_test_preds = np.mean(test_preds_list, axis=0)

test_ranked_labels = []
for i in range(len(final_test_preds)):
    top_3_indices = np.argsort(final_test_preds[i])[-3:][::-1]
    top_3_fertilizers = [fertilizer_classes[idx] for idx in top_3_indices]
    test_ranked_labels.append(" ".join(top_3_fertilizers))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': test_ranked_labels
})

submission_df.to_csv('submission_1.csv', index=False)
print(submission_df.head())


