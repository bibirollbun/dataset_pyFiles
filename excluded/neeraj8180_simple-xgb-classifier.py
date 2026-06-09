!pip install -q catboost lightgbm shap


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
import shap

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

target_col = 'Personality'
id_col = 'id'

X_train = train.drop([target_col], axis=1)
y_train_raw = train[target_col]
X_test = test.copy()

label_encoder_target = LabelEncoder()
y_train = label_encoder_target.fit_transform(y_train_raw)
print('Target classes:', label_encoder_target.classes_)

num_pos = sum(y_train == 1)
num_neg = sum(y_train == 0)
imbalance_ratio = num_neg / num_pos
print(f"Positives: {num_pos}, Negatives: {num_neg}, Ratio: {imbalance_ratio:.2f}")

scale_pos_weight = imbalance_ratio

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
if id_col in num_cols:
    num_cols.remove(id_col)




filled_train = X_train.copy()
filled_test = X_test.copy()

for col in num_cols:
    mean_val = filled_train[col].mean()
    filled_train[col] = filled_train[col].fillna(mean_val)
    filled_test[col] = filled_test[col].fillna(mean_val)

for col in cat_cols:
    mode_val = filled_train[col].mode()[0]
    filled_train[col] = filled_train[col].fillna(mode_val)
    filled_test[col] = filled_test[col].fillna(mode_val)

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([filled_train[col], filled_test[col]], axis=0).astype(str)
    le.fit(combined)
    filled_train[col] = le.transform(filled_train[col].astype(str))
    filled_test[col] = le.transform(filled_test[col].astype(str))

from itertools import combinations

interaction_pairs = list(combinations(num_cols, 2))
for f1, f2 in interaction_pairs:
    new_col = f"{f1}_x_{f2}"
    filled_train[new_col] = filled_train[f1] * filled_train[f2]
    filled_test[new_col] = filled_test[f1] * filled_test[f2]

for col in cat_cols:
    freq = filled_train[col].value_counts()
    filled_train[f"{col}_freq"] = filled_train[col].map(freq)
    filled_test[f"{col}_freq"] = filled_test[col].map(freq)
features = [c for c in filled_train.columns if c != id_col]
X_train_final = filled_train[features]
X_test_final = filled_test[features]
print(f"Total features after engineering: {len(features)}")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print("\nTraining LightGBM for SHAP feature selection...")
lgbm_shap_model = LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
lgbm_shap_model.fit(X_tr, y_tr)

explainer = shap.TreeExplainer(lgbm_shap_model)
shap_values = explainer.shap_values(X_tr)[1]
mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    'feature': X_tr.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values(by='mean_abs_shap', ascending=False)

top_n = 50
top_features = feature_importance['feature'].iloc[:top_n].tolist()
print("\nTop features selected:\n", top_features)
X_tr_sel = X_tr[top_features]
X_val_sel = X_val[top_features]
X_test_sel = X_test_final[top_features]
print("\nTraining first layer (CatBoost)...")
cat_model = CatBoostRegressor(
    random_state=42,
    verbose=0
)
cat_model.fit(X_tr_sel, y_tr)

cat_val_preds = cat_model.predict(X_val_sel)
train_residuals = y_tr - cat_model.predict(X_tr_sel)
print("\nTraining second layer (LightGBM on residuals)...")
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_tr_sel, train_residuals)

lgbm_val_residual_preds = lgbm_model.predict(X_val_sel)
final_val_preds = cat_val_preds + lgbm_val_residual_preds
final_val_preds_binary = (final_val_preds > 0.5).astype(int)

val_acc = accuracy_score(y_val, final_val_preds_binary)
print(f"\nResidual Stacking Validation Accuracy: {val_acc:.4f}\n")
print(classification_report(y_val, final_val_preds_binary, target_names=label_encoder_target.classes_))
print("\nStarting Pseudo-labeling...")

cat_test_preds = cat_model.predict(X_test_sel)
test_residual_preds = lgbm_model.predict(X_test_sel)
final_test_preds_raw = cat_test_preds + test_residual_preds
test_proba = final_test_preds_raw

confident_mask = (test_proba > 0.99) | (test_proba < 0.01)
pseudo_X = X_test_sel[confident_mask]
pseudo_y = (test_proba[confident_mask] > 0.5).astype(int)

print(f"Using {len(pseudo_y)} pseudo-labeled samples.")

X_pseudo_extended = pd.concat([X_train_final[top_features], pseudo_X])
y_pseudo_extended = np.concatenate([y_train, pseudo_y])
final_model = LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    class_weight='balanced',
    random_state=42
)
final_model.fit(X_pseudo_extended, y_pseudo_extended)
test_preds_numeric = final_model.predict(X_test_final[top_features])
test_preds = label_encoder_target.inverse_transform(test_preds_numeric)
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: test_preds
})
submission.to_csv('submission_imbalance_handled_corrected.csv', index=False)
print("\nsubmission_imbalance_handled_corrected.csv generated!")





