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


import os, gc, warnings, numpy as np, pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# CONFIG

SEED = 42
FOLDS = 7
USE_GPU = False  # Set True if GPU available
INPUT_DIR = '/kaggle/input/playground-series-s5e10'
TRAIN_PATH = f'{INPUT_DIR}/train.csv'
TEST_PATH  = f'{INPUT_DIR}/test.csv'
TARGET = 'accident_risk'
ID_COL = 'id'

np.random.seed(SEED)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "xgb_ultra_meta"


# LOAD DATA
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y = train[TARGET].copy()
train_ids = train[ID_COL].copy()
test_ids  = test[ID_COL].copy()

train = train.drop(columns=[TARGET, ID_COL])
test  = test.drop(columns=[ID_COL])

print(f"Train: {train.shape} | Test: {test.shape}")

# Stratification bins
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')


def create_ultra_meta_features(df, is_train=True):
    X = df.copy()
    
    # === BASE RISK COMPONENTS ===
    X['risk_curve']      = 0.30 * X['curvature']
    X['risk_night']      = 0.20 * (X['lighting'] == 'night').astype(int)
    X['risk_bad_weather']= 0.15 * (~X['weather'].isin(['clear'])).astype(int)
    X['risk_high_speed'] = 0.25 * (X['speed_limit'] >= 80).astype(int)
    X['risk_accidents']  = 0.18 * (X['num_reported_accidents'] > 3).astype(int)
    
    X['risk_total_base'] = (X['risk_curve'] + X['risk_night'] + X['risk_bad_weather'] + 
                            X['risk_high_speed'] + X['risk_accidents'])

    # === INTERACTIONS ===
    X['risk_night_curve']     = X['risk_night'] * X['risk_curve'] * 1.4
    X['risk_weather_speed']   = X['risk_bad_weather'] * X['risk_high_speed'] * 1.3
    X['risk_triple_threat']   = X['risk_night'] * X['risk_bad_weather'] * X['risk_high_speed']
    X['risk_speed_curve']     = X['risk_high_speed'] * X['risk_curve']

    # === NON‑LINEAR ===
    X['risk_total_sq']   = X['risk_total_base'] ** 2
    X['risk_total_sqrt'] = np.sqrt(np.abs(X['risk_total_base']))
    X['risk_curve_sq']   = X['risk_curve'] ** 2

    # === GRANULAR CONDITIONS ===
    X['risk_poor_visibility'] = 0.28 * (
        (X['lighting'].isin(['night', 'dawn', 'dusk'])) | 
        (X['weather'].isin(['fog', 'rain', 'snow']))
    ).astype(int)
    
    X['risk_severe_weather'] = 0.22 * X['weather'].isin(['snow', 'fog']).astype(int)
    X['risk_wet_road']       = 0.14 * X['weather'].isin(['rain', 'snow']).astype(int)

    # === SPEED ZONES ===
    X['risk_very_high_speed'] = 0.30 * (X['speed_limit'] >= 90).astype(int)
    X['risk_moderate_speed']  = 0.12 * ((X['speed_limit'] >= 50) & (X['speed_limit'] < 80)).astype(int)

    # === ACCIDENT HISTORY ===
    X['risk_high_accidents']     = 0.20 * (X['num_reported_accidents'] > 6).astype(int)
    X['risk_accident_rate']      = X['num_reported_accidents'] / (X['num_reported_accidents'].max() + 1)

    # === CURVATURE ZONES (quantile‑based) ===
    if is_train:
        curve_q75 = X['curvature'].quantile(0.75)
        curve_q25 = X['curvature'].quantile(0.25)
    else:
        curve_q75 = 0.75  # will be replaced in pipeline
        curve_q25 = 0.25
    
    X['risk_sharp_curve']    = 0.38 * (X['curvature'] > curve_q75).astype(int)
    X['risk_moderate_curve'] = 0.22 * ((X['curvature'] > curve_q25) & (X['curvature'] <= curve_q75)).astype(int)

    # === RATIO FEATURES ===
    X['curve_per_speed']     = X['curvature'] / (X['speed_limit'] + 1)
    X['accidents_per_curve'] = X['num_reported_accidents'] / (X['curvature'] + 1)

    # === FINAL ENHANCED SCORE ===
    X['risk_enhanced_total'] = (
        X['risk_total_base'] + X['risk_poor_visibility'] + X['risk_severe_weather'] +
        X['risk_very_high_speed'] + X['risk_high_accidents'] + X['risk_sharp_curve']
    )

    return X, (curve_q75, curve_q25) if is_train else X


print("\nEngineering ultra meta features...")
X_train = train.copy()
X_test  = test.copy()

X_train, (q75, q25) = create_ultra_meta_features(X_train, is_train=True)
X_test = create_ultra_meta_features(X_test, is_train=False)[0]

# Apply same quantiles to test
X_test['risk_sharp_curve']    = 0.38 * (X_test['curvature'] > q75).astype(int)
X_test['risk_moderate_curve'] = 0.22 * ((X_test['curvature'] > q25) & (X_test['curvature'] <= q75)).astype(int)

print(f"Features: {X_train.shape[1]} (+{X_train.shape[1] - train.shape[1]})")

# Label encode
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))


BEST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
    'max_bin': 518,
    'learning_rate': 0.0185,
    'max_depth': 7,
    'min_child_weight': 6,
    'subsample': 0.804,
    'colsample_bytree': 0.632,
    'colsample_bylevel': 0.826,
    'colsample_bynode': 0.846,
    'reg_alpha': 0.156,
    'reg_lambda': 0.972,
    'gamma': 0.0048,
    'random_state': SEED,
    'n_jobs': -1
}


print("\nTraining XGBoost with 7‑fold Stratified CV...")
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []
feat_imp_dict = {}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bins), 1):
    print(f"  Fold {fold}/{FOLDS}")
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dvalid = xgb.DMatrix(X_val, y_val)
    dtest  = xgb.DMatrix(X_test)
    
    model = xgb.train(
        BEST_PARAMS,
        dtrain,
        num_boost_round=12000,
        evals=[(dvalid, 'val')],
        early_stopping_rounds=250,
        verbose_eval=False
    )
    
    # Importance
    imp = model.get_score(importance_type='gain')
    for k, v in imp.items():
        feat_imp_dict.setdefault(k, []).append(v)
    
    oof_preds[val_idx] = model.predict(dvalid)
    test_preds += model.predict(dtest) / FOLDS
    
    rmse = mean_squared_error(y_val, oof_preds[val_idx], squared=False)
    fold_scores.append(rmse)
    print(f"    RMSE: {rmse:.6f}")

test_preds /= FOLDS
cv_rmse = mean_squared_error(y, oof_preds, squared=False)
cv_std  = np.std(fold_scores)

print(f"\nFINAL CV RMSE: {cv_rmse:.6f} (±{cv_std:.6f})")


# Average importance
avg_imp = {k: np.mean(v) for k, v in feat_imp_dict.items()}
imp_df = pd.DataFrame({
    'feature': list(avg_imp.keys()),
    'importance': list(avg_imp.values())
}).sort_values('importance', ascending=False)

feature_names = X_train.columns
imp_df['name'] = imp_df['feature'].apply(lambda x: feature_names[int(x[1:])] if x.startswith('f') else x)
top20 = imp_df.head(20)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Bar
ax1.barh(range(len(top20)), top20['importance'], color='#00d4d4')
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['name'], fontsize=9)
ax1.set_title('Top 20 Features (Gain)')
ax1.invert_yaxis()

# Treemap
ax2.axis('off')
ax2.set_title('Treemap (Top 15)')
x0 = y0 = 0
for _, row in top20.head(15).iterrows():
    w = row['importance'] / top20['importance'].sum() * 50
    h = 0.8
    color = plt.cm.coolwarm(row['importance'] / top20['importance'].max())
    ax2.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor='white'))
    if w > 2:
        ax2.text(x0 + w/2, y0 + h/2, row['name'][:12], ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    x0 += w
    if x0 > 10:
        x0 = 0
        y0 += h

# Cumulative
cum = np.cumsum(top20.head(30)['importance']) / imp_df['importance'].sum() * 100
ax3.plot(cum, marker='o', color='#d63031')
ax3.axhline(80, color='red', linestyle='--', label='80%')
ax3.axhline(90, color='orange', linestyle='--', label='90%')
ax3.set_title('Cumulative Importance')
ax3.legend()

fig.suptitle(f'XGBoost Ultra | CV RMSE: {cv_rmse:.6f}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'feat_imp_ultra_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.show()


# Submission
sub = pd.DataFrame({ID_COL: test_ids, TARGET: test_preds.clip(0, 1)})
sub.to_csv('submission.csv', index=False)
print("submission.csv saved!")

# OOF & Test
np.save(f'oof_{model_name}_{timestamp}.npy', oof_preds)
np.save(f'test_{model_name}_{timestamp}.npy', test_preds)

# Summary
with open(f'summary_{model_name}_{timestamp}.txt', 'w') as f:
    f.write(f"CV RMSE: {cv_rmse:.6f} (±{cv_std:.6f})\n")
    f.write(f"Folds: {fold_scores}\n")
    f.write(f"Features: {X_train.shape[1]}\n")
    f.write(f"Top 5: {', '.join(top20['name'].head(5).tolist())}\n")

print(f"All files saved with timestamp: {timestamp}")
gc.collect()

