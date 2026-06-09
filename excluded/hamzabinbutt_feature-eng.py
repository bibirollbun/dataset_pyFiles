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


# Paths
train_path = '/kaggle/input/playground-series-s5e6/train.csv'
submission_path = '/kaggle/input/playground-series-s5e6/sample_submission.csv'

# Load the data
df_train = pd.read_csv(train_path)
df_submission = pd.read_csv(submission_path)

# Quick preview
print("Train shape:", df_train.shape)

df_train.head()


# Drop the 'id' column
df_train.drop(columns='id', inplace=True)

# Check updated shapes
print("âœ… Train shape:", df_train.shape)


# Replace spaces with underscores in column names
df_train.columns = df_train.columns.str.replace(' ', '_')

# Re-identify numerical and categorical columns
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Print updated results
print("ğŸ“Š Numerical columns:")
print(num_cols)

print("\nğŸ”¤ Categorical columns:")
print(cat_cols)



# Define categorical variables (excluding the target 'Fertilizer_Name')
cat_vars = ['Soil_Type', 'Crop_Type']

# Define numerical variables
num_vars = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Loop through each categorical variable
for cat in cat_vars:
    # Calculate the proportion of each category
    proportions = df_train[cat].value_counts(normalize=True)
    
    # Map the proportion back to the DataFrame
    df_train[f'{cat}_prop'] = df_train[cat].map(proportions)
    
    # Multiply each numeric variable by the proportion
    for num in num_vars:
        new_col = f'{num}_x_{cat}_prop'
        df_train[new_col] = df_train[num] * df_train[f'{cat}_prop']

# Drop temporary proportion columns (optional)
df_train.drop(columns=[f'{cat}_prop' for cat in cat_vars], inplace=True)

# View the new columns
df_train[[col for col in df_train.columns if '_x_' in col]].head()

# Check updated shapes
print("âœ… Train shape:", df_train.shape)


# Check for missing values in each column
print(df_train.isnull().sum())



import numpy as np

# ğŸ“Š Numerical columns
num_vars = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# ğŸ”� Loop over each numeric column
for col in num_vars:
    # Get the 10 most frequent values (peaks)
    top_10_values = df_train[col].value_counts().nlargest(10).index.tolist()

    # Create 10 new columns: difference with each top value
    for i, top_val in enumerate(top_10_values, 1):
        new_col_name = f"{col}_diff_from_peak_{i}"
        df_train[new_col_name] = np.abs(df_train[col] - top_val)

# Check updated shapes
print("âœ… Train shape:", df_train.shape)


# Define your categorical and numerical variables
cat_vars = ['Soil_Type', 'Crop_Type']
num_vars = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Step 1: Calculate proportion of each category
soil_prop = df_train['Soil_Type'].value_counts(normalize=True)
crop_prop = df_train['Crop_Type'].value_counts(normalize=True)

# Step 2: Map proportions to original dataframe
df_train['Soil_prop'] = df_train['Soil_Type'].map(soil_prop)
df_train['Crop_prop'] = df_train['Crop_Type'].map(crop_prop)

# Step 3: Multiply the two proportions to get a combined proportion
df_train['SoilCrop_prop'] = df_train['Soil_prop'] * df_train['Crop_prop']

# Step 4: Multiply each numerical variable with the combined proportion
for num in num_vars:
    df_train[f'{num}_x_SoilCrop_prop'] = df_train[num] * df_train['SoilCrop_prop']

# Optional: Drop intermediate proportion columns
df_train.drop(columns=['Soil_prop', 'Crop_prop', 'SoilCrop_prop'], inplace=True)

# Show results
print("âœ… Created columns:", [f'{num}_x_SoilCrop_prop' for num in num_vars])
print("âœ… df_train shape:", df_train.shape)



# Assuming df_train contains the numeric columns:
# 'Nitrogen', 'Phosphorous', 'Potassium'

# Create ratio features
df_train['N_to_P'] = df_train['Nitrogen'] / (df_train['Phosphorous'] + 1e-5)
df_train['N_to_K'] = df_train['Nitrogen'] / (df_train['Potassium'] + 1e-5)
df_train['P_to_K'] = df_train['Phosphorous'] / (df_train['Potassium'] + 1e-5)

print("âœ… df_train shape:", df_train.shape)



# Create ratio features
df_train['Temp_to_Humid'] = df_train['Temparature'] / (df_train['Humidity'] + 1e-5)
df_train['Temp_to_Moist'] = df_train['Temparature'] / (df_train['Moisture'] + 1e-5)
df_train['Humid_to_Moist'] = df_train['Humidity'] / (df_train['Moisture'] + 1e-5)

print("âœ… df_train shape:", df_train.shape)



# Categorical variables
cat_vars = ['Soil_Type', 'Crop_Type']

# Numerical variables
num_vars = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Iterate over categorical variables
for cat in cat_vars:
    # Create binary indicator columns (one-hot encoding without dropping any column)
    dummies = pd.get_dummies(df_train[cat], prefix=cat)
    
    # Add these binary columns to df_train
    df_train = pd.concat([df_train, dummies], axis=1)
    
    # Multiply each dummy with each numerical variable
    for dummy_col in dummies.columns:
        for num_col in num_vars:
            new_col = f'{num_col}_x_{dummy_col}'
            df_train[new_col] = df_train[num_col] * df_train[dummy_col]

# Check how many new columns were created
print("âœ… Shape after adding all features:", df_train.shape)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (excluding target)
cat_cols = ['Soil_Type', 'Crop_Type']

# Encode features
feature_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    feature_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
df_train['Fertilizer_Name'] = label_encoder.fit_transform(df_train['Fertilizer_Name'])

target_col='Fertilizer_Name'
# âœ… Prepare features (X) and target (y)
X = df_train.drop(target_col, axis=1)
y = df_train[target_col]

# âœ… Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# âœ… Print shapes
print("âœ… Data split complete:")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape:   {y_val.shape}")



df_train


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.neighbors import KNeighborsClassifier as cuKNN
import cudf

# âš™ï¸� MAP@3 Evaluation Metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits, used = 0.0, 0.0, set()
        for i, pred in enumerate(p):
            if pred in a and pred not in used:
                hits += 1.0
                score += hits / (i + 1.0)
                used.add(pred)
        return score / min(len(a), k)
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

def map3_score(y_true, y_proba):
    top3 = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    return mapk(y_true, top3)

# ğŸ§  Define models
models = {
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        objective='multi:softprob', eval_metric='mlogloss',
        tree_method='hist', device='cuda'),

    'LightGBM': LGBMClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        objective='multiclass', random_state=42),

    'CatBoost': CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=6,
        loss_function='MultiClass', verbose=0, task_type='GPU')}

# ğŸ“Š Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_all = []
oof_preds_dict = {name: np.zeros((len(X_train), len(np.unique(y_train)))) for name in models}

for model_name, model in models.items():
    print(f"\nğŸ”� Training {model_name}")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if model_name.startswith('cuML'):
            X_tr_cudf = cudf.DataFrame.from_pandas(X_tr)
            X_vl_cudf = cudf.DataFrame.from_pandas(X_vl)
            y_tr_cudf = cudf.Series(y_tr.values)
            y_vl_cudf = cudf.Series(y_vl.values)

            model.fit(X_tr_cudf, y_tr_cudf)

            if hasattr(model, "predict_proba"):
                y_tr_pred = model.predict_proba(X_tr_cudf).to_numpy()
                y_vl_pred = model.predict_proba(X_vl_cudf).to_numpy()
            else:
                def to_proba(labels, n_classes):
                    one_hot = np.zeros((len(labels), n_classes))
                    for i, val in enumerate(labels):
                        one_hot[i, int(val)] = 1
                    return one_hot
                y_tr_pred = to_proba(model.predict(X_tr_cudf).to_numpy(), len(np.unique(y_train)))
                y_vl_pred = to_proba(model.predict(X_vl_cudf).to_numpy(), len(np.unique(y_train)))
        else:
            model.fit(X_tr, y_tr)
            y_tr_pred = model.predict_proba(X_tr)
            y_vl_pred = model.predict_proba(X_vl)

        # Save OOF predictions for ensemble
        oof_preds_dict[model_name][val_idx] = y_vl_pred

        train_map3 = map3_score(y_tr, y_tr_pred)
        val_map3 = map3_score(y_vl, y_vl_pred)

        print(f"  ğŸ“˜ Fold {fold}: Train MAP@3={train_map3:.4f} | Val MAP@3={val_map3:.4f}")
        fold_results.append({
            'Model': model_name,
            'Fold': fold,
            'Train_MAP@3': round(train_map3, 4),
            'Val_MAP@3': round(val_map3, 4)
        })

    results_all.extend(fold_results)

# ğŸ”„ Ensemble: Average OOF Probabilities
ensemble_oof_pred = np.mean(list(oof_preds_dict.values()), axis=0)
ensemble_map3 = map3_score(y_train, ensemble_oof_pred)

# ğŸ“ˆ Results Summary
results_df = pd.DataFrame(results_all)
print("\nğŸ“Š Summary of MAP@3 across models and folds:")
print(results_df.groupby("Model")[["Train_MAP@3", "Val_MAP@3"]].mean().round(4))

print(f"\nğŸŒŸ Ensemble OOF MAP@3 Score: {ensemble_map3:.4f}")



from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder

# ğŸ§® 1. Compute overfitting gap per model
results_df["Gap"] = results_df["Train_MAP@3"] - results_df["Val_MAP@3"]

# ğŸ“Š 2. Average gap per model
avg_gaps = results_df.groupby("Model")["Gap"].mean().abs()

# ğŸ”¢ 3. Inverse gap weights (smaller gap â‡’ higher weight)
inv_gap = 1 / avg_gaps
weights = inv_gap / inv_gap.sum()
weights = weights.to_dict()

print("\nğŸ“� Computed model weights (based on inverse MAP@3 gap):")
for k, v in weights.items():
    print(f"  {k}: {v:.4f}")

# ğŸ§ª 4. Weighted OOF prediction
ensemble_weighted_oof = sum(weights[model] * oof_preds_dict[model] for model in models)
weighted_ensemble_map3 = map3_score(y_train, ensemble_weighted_oof)
print(f"\nğŸŒŸ Weighted Ensemble OOF MAP@3 Score: {weighted_ensemble_map3:.4f}")



from sklearn.linear_model import RidgeCV
from sklearn.multiclass import OneVsRestClassifier

# âœ… Stack base model OOF predictions column-wise to form meta-features
X_meta = np.hstack([oof_preds_dict[model] for model in models])

# âœ… Meta-target is just the original training labels
y_meta = y_train.values

# ğŸ§  Ridge Meta-Model with OneVsRest
meta_model = OneVsRestClassifier(RidgeCV(alphas=np.logspace(-3, 3, 10)))
meta_model.fit(X_meta, y_meta)

# âœ… Get class-wise regression outputs
ensemble_ridge_oof = meta_model.predict(X_meta)

# â›”ï¸� This gives hard labels (not usable for softmax)
# What we want instead is `.predict()` from each estimator to build raw scores for each class

# âœ… Better: Use .predict from each estimator to get continuous outputs manually
ridge_preds = np.column_stack([
    est.predict(X_meta) for est in meta_model.estimators_
])

# âœ… Apply softmax to convert to probability-like scores
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

ensemble_ridge_oof = softmax(ridge_preds)

# ğŸ�¯ Final Ridge Meta-Model Score
print(f"ğŸ�¯ Ridge Meta-Model OOF MAP@3 Score: {map3_score(y_meta, ensemble_ridge_oof):.4f}")


