import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
tc_smiles = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')


train = train.merge(tc_smiles, on='SMILES', how='left')
test = test.merge(tc_smiles, on='SMILES', how='left')


target_cols = ['Density', 'Tc', 'Tg', 'FFV', 'Rg']
train[target_cols] = train[target_cols].fillna(train[target_cols].median())


tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=256)
tfidf_train = tfidf.fit_transform(train['SMILES'].fillna('')).toarray()
tfidf_test = tfidf.transform(test['SMILES'].fillna('')).toarray()

tfidf_train_df = pd.DataFrame(tfidf_train, columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])])
tfidf_test_df = pd.DataFrame(tfidf_test, columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])])

X = tfidf_train_df.copy()
X_test = tfidf_test_df.copy()
y = train[target_cols]

X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_val = rf.predict(X_val)
rf_test_preds = rf.predict(X_test)


lgb_val = np.zeros_like(y_val)
lgb_test_preds = np.zeros((X_test.shape[0], len(target_cols)))
for i, col in enumerate(target_cols):
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train[col])
    lgb_val[:, i] = model.predict(X_val)
    lgb_test_preds[:, i] = model.predict(X_test)


cat_val = np.zeros_like(y_val)
cat_test_preds = np.zeros((X_test.shape[0], len(target_cols)))
for i, col in enumerate(target_cols):
    print(f"Training CatBoost for {col}...")
    cat = CatBoostRegressor(iterations=200, depth=6, verbose=100, random_seed=42)
    cat.fit(X_train, y_train[col])
    cat_val[:, i] = cat.predict(X_val)
    cat_test_preds[:, i] = cat.predict(X_test)


print("Random Forest MAE:", dict(zip(target_cols, mean_absolute_error(y_val, rf_val, multioutput='raw_values'))))
print("LightGBM MAE:", dict(zip(target_cols, mean_absolute_error(y_val, lgb_val, multioutput='raw_values'))))
print("CatBoost MAE:", dict(zip(target_cols, mean_absolute_error(y_val, cat_val, multioutput='raw_values'))))


val_ensemble = (rf_val + lgb_val + cat_val) / 3
mae_ensemble = mean_absolute_error(y_val, val_ensemble, multioutput='raw_values')
print("Ensemble Average MAE:", dict(zip(target_cols, mae_ensemble)))


mae_scores = {}
for i, col in enumerate(target_cols):
    y_true = y_val[col].values
    y_pred = val_ensemble[:, i]
    mae = mean_absolute_error(y_true, y_pred)
    val_range = y_true.max() - y_true.min()
    n = len(y_true)
    mae_scores[col] = {'mae': mae, 'range': val_range, 'n': n}

weights = {}
denom = sum((1 / np.sqrt(v['n'])) / v['range'] for v in mae_scores.values())
for col, v in mae_scores.items():
    weights[col] = ((1 / np.sqrt(v['n'])) / v['range']) / denom

wmae = sum(weights[col] * mae_scores[col]['mae'] for col in target_cols)
print(f"\nğŸ”¥ Competition Metric â€” Weighted MAE (wMAE): {wmae:.5f}")


final_preds = (rf_test_preds + lgb_test_preds + cat_test_preds) / 3


submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
submission[target_cols] = final_preds
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved as 'submission.csv'")



import os
print("âœ… Files in output folder:")
print(os.listdir("."))



print(train.columns)


print(tc_smiles.columns)

