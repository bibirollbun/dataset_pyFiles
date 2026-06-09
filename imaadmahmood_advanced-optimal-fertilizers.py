import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ---------------------------
# Verify files in input directory (optional)
# ---------------------------
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e6'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ---------------------------
# Load data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# ---------------------------
# Create Soil_Crop_Interaction in train and test before concatenation
# ---------------------------
train['Soil_Crop_Interaction'] = train['Soil Type'].astype(str) + '_' + train['Crop Type'].astype(str)
test['Soil_Crop_Interaction'] = test['Soil Type'].astype(str) + '_' + test['Crop Type'].astype(str)

# ---------------------------
# Combine train and test for feature engineering
# ---------------------------
train['is_train'] = 1
test['is_train'] = 0
test['Fertilizer'] = -1  # placeholder for target column

combined = pd.concat([train, test], axis=0, ignore_index=True)

# ---------------------------
# Label Encoding for target (encoded fertilizer codes)
# ---------------------------
le_fert = LabelEncoder()
combined.loc[combined['is_train'] == 1, 'Fertilizer'] = le_fert.fit_transform(combined.loc[combined['is_train'] == 1, 'Fertilizer'])

# ---------------------------
# Feature Engineering
# ---------------------------
combined['N_P_ratio'] = combined['Nitrogen'] / (combined['Phosphorous'] + 1e-5)
combined['N_K_ratio'] = combined['Nitrogen'] / (combined['Potassium'] + 1e-5)
combined['P_K_ratio'] = combined['Phosphorous'] / (combined['Potassium'] + 1e-5)
combined['NPK_sum'] = combined['Nitrogen'] + combined['Phosphorous'] + combined['Potassium']

combined['Temp_Humidity'] = combined['Temparature'] * combined['Humidity']

combined['N_Temp'] = combined['Nitrogen'] * combined['Temparature']
combined['Moisture_Humidity'] = combined['Moisture'] * combined['Humidity']
combined['log_Nitrogen'] = np.log1p(combined['Nitrogen'])
combined['log_Phosphorous'] = np.log1p(combined['Phosphorous'])
combined['log_Potassium'] = np.log1p(combined['Potassium'])

for col in ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']:
    freq = combined[col].value_counts() / len(combined)
    combined[f'{col}_freq'] = combined[col].map(freq)

# Group means from train only
for cat_col in ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']:
    for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
        group_mean = train.groupby(cat_col)[nutrient].mean()
        combined[f'{cat_col}_{nutrient}_mean'] = combined[cat_col].map(group_mean)

# Binning continuous features
for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture', 'Temparature', 'Humidity']:
    combined[f'{col}_bin'] = pd.qcut(combined[col], 10, duplicates='drop').astype(str)

# ---------------------------
# Encoding categorical features
# ---------------------------
for col in ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

for col in [f'{col}_bin' for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture', 'Temparature', 'Humidity']]:
    combined[col] = LabelEncoder().fit_transform(combined[col])

# ---------------------------
# Target Encoding with KFold
# ---------------------------
def target_encode(trn_series=None, tst_series=None, target=None, n_splits=5, shuffle=True, random_state=42):
    assert len(trn_series) == len(target)
    oof = pd.Series(np.nan, index=trn_series.index)
    tst_mean = pd.Series(0, index=tst_series.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_idx, val_idx in skf.split(trn_series, target):
        trn_fold, val_fold = trn_series.iloc[train_idx], trn_series.iloc[val_idx]
        target_fold = target.iloc[train_idx]
        means = target_fold.groupby(trn_fold).mean()
        oof.iloc[val_idx] = val_fold.map(means)
    means = target.groupby(trn_series).mean()
    tst_mean = tst_series.map(means)
    return oof.fillna(target.mean()), tst_mean.fillna(target.mean())

cat_cols_to_te = ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']
for col in cat_cols_to_te:
    trn_series = combined.loc[combined['is_train'] == 1, col]
    tst_series = combined.loc[combined['is_train'] == 0, col]
    target = combined.loc[combined['is_train'] == 1, 'Fertilizer']
    oof_te, tst_te = target_encode(trn_series=trn_series, tst_series=tst_series, target=target, n_splits=5)
    combined.loc[combined['is_train'] == 1, f'{col}_te'] = oof_te
    combined.loc[combined['is_train'] == 0, f'{col}_te'] = tst_te

# ---------------------------
# Prepare final datasets
# ---------------------------
features = [
    'Nitrogen', 'Phosphorous', 'Potassium', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum',
    'Temparature', 'Humidity', 'Moisture', 'Temp_Humidity',
    'Soil Type', 'Crop Type', 'Soil_Crop_Interaction',
    'Soil Type_freq', 'Crop Type_freq', 'Soil_Crop_Interaction_freq',
    'Soil Type_te', 'Crop Type_te', 'Soil_Crop_Interaction_te',
    'Soil Type_Nitrogen_mean', 'Soil Type_Phosphorous_mean', 'Soil Type_Potassium_mean',
    'Crop Type_Nitrogen_mean', 'Crop Type_Phosphorous_mean', 'Crop Type_Potassium_mean',
    'Soil_Crop_Interaction_Nitrogen_mean', 'Soil_Crop_Interaction_Phosphorous_mean', 'Soil_Crop_Interaction_Potassium_mean',
    'N_Temp', 'Moisture_Humidity',
    'log_Nitrogen', 'log_Phosphorous', 'log_Potassium',
    'Nitrogen_bin', 'Phosphorous_bin', 'Potassium_bin', 'Moisture_bin', 'Temparature_bin', 'Humidity_bin'
]

X_train = combined.loc[combined['is_train'] == 1, features]
y_train = combined.loc[combined['is_train'] == 1, 'Fertilizer'].astype(int)
X_test = combined.loc[combined['is_train'] == 0, features]

# ---------------------------
# LightGBM Training with callbacks for early stopping
# ---------------------------
SEED = 42
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros((len(X_train), len(le_fert.classes_)))
test_preds = np.zeros((len(X_test), len(le_fert.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    unique_classes = np.unique(y_tr)
    if len(unique_classes) < 2:
        print(f"Fold {fold+1} training set has only one class: {unique_classes}. Skipping this fold.")
        continue

    model = LGBMClassifier(
        objective='multiclass',
        learning_rate=0.02,
        num_leaves=127,
        max_depth=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=SEED,
        device='gpu'  # Remove if no GPU available
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
    )

    oof_preds[val_idx, :] = model.predict_proba(X_val, num_iteration=model.best_iteration_)
    test_preds += model.predict_proba(X_test, num_iteration=model.best_iteration_) / NFOLDS

# ---------------------------
# OOF MAP@3 Evaluation
# ---------------------------
def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            score += 1.0
    return score / len(actual)

oof_top3 = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
oof_top3_labels = le_fert.inverse_transform(oof_top3.flatten()).reshape(oof_top3.shape)
actual_labels = le_fert.inverse_transform(y_train)

oof_map3 = mapk(actual_labels, oof_top3_labels.tolist(), k=3)
print(f"OOF Validation MAP@3: {oof_map3:.5f}")

# ---------------------------
# Map encoded labels back to original Fertilizer Names for submission
# ---------------------------
# Use combined DataFrame to get mapping from encoded label to Fertilizer Name string
fertilizer_map = combined.loc[combined['is_train'] == 1, ['Fertilizer', 'Fertilizer Name']].drop_duplicates().set_index('Fertilizer')['Fertilizer Name'].to_dict()

num_classes = len(le_fert.classes_)
top_k = min(3, num_classes)

test_topk = np.argsort(test_preds, axis=1)[:, -top_k:][:, ::-1]

def map_to_fertilizer_names(encoded_array, mapping):
    n_samples, top_k = encoded_array.shape
    names = []
    for i in range(n_samples):
        row_names = [mapping[int(code)] for code in encoded_array[i]]
        names.append(' '.join(row_names))
    return names

fertilizer_names = map_to_fertilizer_names(test_topk, fertilizer_map)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': fertilizer_names
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


