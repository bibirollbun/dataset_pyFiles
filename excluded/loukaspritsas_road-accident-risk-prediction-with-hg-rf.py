
# ========= Imports =========
import os, glob, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE

# Basic models from sklearn (fast to run)
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# Encoders + preprocessing
from sklearn.preprocessing import OrdinalEncoder

# ========= CONFIG =========
TARGET_COL = "accident_risk"
ID_COL     = "id"
N_SPLITS   = 5      # K-Fold splits
RANDOM_SEED = 42
VERBOSE = True
PREFERRED_DIR_NAME = "predict" 




# ========= Data Loading =========

def find_kaggle_comp_dir():
    base = '/kaggle/input'
    if not os.path.exists(base):
        return None
    candidates = []
    for d in glob.glob(os.path.join(base, '*')):
        if os.path.isdir(d):
            has_train = len(glob.glob(os.path.join(d, 'train.csv'))) > 0
            has_test  = len(glob.glob(os.path.join(d, 'test.csv'))) > 0
            if has_train and has_test:
                candidates.append(d)
    if candidates:
        cand_pref = [d for d in candidates if PREFERRED_DIR_NAME.lower() in os.path.basename(d).lower()]
        return cand_pref[0] if cand_pref else candidates[0]
    return None

comp_dir = find_kaggle_comp_dir()

if comp_dir is not None:
    if VERBOSE: print(f"Found competition dir: {comp_dir}")
    train_path = os.path.join(comp_dir, 'train.csv')
    test_path  = os.path.join(comp_dir, 'test.csv')
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
else:
    if VERBOSE: print("Competition files not found. Using a small synthetic dataset so the notebook can run.")
    rng = np.random.RandomState(RANDOM_SEED)
    n_train, n_test = 1200, 800
    dates = pd.date_range('2021-01-01', periods=n_train+n_test, freq='H')
    cat1  = rng.choice(['A','B','C'], size=n_train+n_test)
    cat2  = rng.choice(['Urban','Rural'], size=n_train+n_test)
    num1  = rng.normal(0,1,size=n_train+n_test)
    num2  = rng.gamma(2.0,1.0,size=n_train+n_test)
    # target with some signal
    target = 0.3*(cat1=='B').astype(float) + 0.6*(cat2=='Urban').astype(float) + 0.5*num1 + 0.2*np.log1p(num2) + rng.normal(0,0.3,size=n_train+n_test)

    df = pd.DataFrame({
        'id': np.arange(n_train+n_test),
        'timestamp': dates.astype(str),
        'weather_cat': cat1,
        'area_cat': cat2,
        'speed_mean': num1,
        'traffic_index': num2,
        'target': target
    })
    train = df.iloc[:n_train].copy()
    test  = df.iloc[n_train:].drop(columns=['target']).copy()

# show shapes
print("Train shape:", train.shape)
print("Test shape :", test.shape)



# ========= Column Detection =========
if TARGET_COL is None:
    only_in_train = [c for c in train.columns if c not in test.columns]
    # don't treat typical meta columns as target
    bad_target_names = set(['fold', 'kfold', 'split', 'subset'])
    candidates = [c for c in only_in_train if c.lower() not in bad_target_names]
    if len(candidates) == 0:
        TARGET_COL = train.columns[-1]
    else:
        TARGET_COL = candidates[0]

if ID_COL is None:
    for guess in ['id','ID','Id','record_id']:
        if guess in test.columns:
            ID_COL = guess
            break
    if ID_COL is None:
        both = [c for c in train.columns if c in test.columns]
        id_like = None
        for c in both:
            # quick check: many unique values & no missing
            if train[c].isna().sum() == 0 and train[c].nunique() > 0.9*len(train):
                id_like = c
                break
        ID_COL = id_like if id_like is not None else both[0]

print("TARGET_COL:", TARGET_COL)
print("ID_COL    :", ID_COL)

# ========= Basic date parsing =========
def enrich_dates(df):
    for c in list(df.columns):
        if df[c].dtype == 'object':
            sample = df[c].dropna().astype(str).head(50)
            parse_ok = 0
            for x in sample:
                try:
                    _ = pd.to_datetime(x, errors='raise')
                    parse_ok += 1
                except:
                    pass
            if parse_ok > 0.8*len(sample) and len(sample)>0:
                # convert
                dt = pd.to_datetime(df[c], errors='coerce')
                df[c+'_year'] = dt.dt.year
                df[c+'_month'] = dt.dt.month
                df[c+'_day'] = dt.dt.day
                df[c+'_hour'] = dt.dt.hour
                # drop original text col (keeps it simple)
                df.drop(columns=[c], inplace=True)
    return df

train = enrich_dates(train)
test  = enrich_dates(test)

# ========= Identify numeric / categorical =========
# Simple rule: object and 'category' dtypes treated as categorical.
cat_cols = [c for c in train.columns if train[c].dtype == 'object' or str(train[c].dtype)=='category']
num_cols = [c for c in train.columns if c not in cat_cols+[TARGET_COL]]

# ========= Fill missing =========
# For numeric, we fill with median. For categorical, we fill with "missing".
for c in num_cols:
    if train[c].isna().any():
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

for c in cat_cols:
    train[c] = train[c].fillna("missing")
    test[c]  = test[c].fillna("missing")

# ========= Encode categoricals with a simple OrdinalEncoder =========
if len(cat_cols) > 0:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_cols] = enc.fit_transform(train[cat_cols])
    test[cat_cols]  = enc.transform(test[cat_cols])

# final feature list (exclude target)
features = [c for c in train.columns if c != TARGET_COL]
print(f"Using {len(features)} features.")



# ========= CV Training =========
X = train[features].copy()
y = train[TARGET_COL].values
X_test = test[features].copy()

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_m1 = np.zeros(len(train))
oof_m2 = np.zeros(len(train))
pred_m1 = np.zeros(len(test))
pred_m2 = np.zeros(len(test))

fold_mae_m1 = []
fold_mae_m2 = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    # Model 1: HistGradientBoostingRegressor (fast + good baseline)
    m1 = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    m1.fit(X_tr, y_tr)
    p1_val = m1.predict(X_val)
    p1_test = m1.predict(X_test)
    oof_m1[val_idx] = p1_val
    pred_m1 += p1_test / N_SPLITS
    mae1 = MAE(y_val, p1_val)
    fold_mae_m1.append(mae1)
    if VERBOSE: print(f"[M1] Fold {fold} MAE = {mae1:.6f}")

    # Model 2: RandomForestRegressor (very basic)
    m2 = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    m2.fit(X_tr, y_tr)
    p2_val = m2.predict(X_val)
    p2_test = m2.predict(X_test)
    oof_m2[val_idx] = p2_val
    pred_m2 += p2_test / N_SPLITS
    mae2 = MAE(y_val, p2_val)
    fold_mae_m2.append(mae2)
    if VERBOSE: print(f"[M2] Fold {fold} MAE = {mae2:.6f}")

# Print CV summaries
print("\nSummary:")
print(f" Model 1 CV mae: {np.mean(fold_mae_m1):.6f} ± {np.std(fold_mae_m1):.6f}")
print(f" Model 2 CV mae: {np.mean(fold_mae_m2):.6f} ± {np.std(fold_mae_m2):.6f}")



# ========= Simple blend (alpha for Model1 vs Model2) =========
# We search alpha in [0,1] step 0.05 to minimize OOF MAE.
alphas = np.linspace(0,1,21)
best_alpha = None
best_mae = 1e9
for a in alphas:
    blend = a*oof_m1 + (1-a)*oof_m2
    m = MAE(train[TARGET_COL].values, blend)
    if m < best_mae:
        best_mae = m
        best_alpha = a

print(f"Best blend alpha (M1 weight): {best_alpha:.3f} | CV mae = {best_mae:.6f}")

# Blended test predictions
pred_blend = best_alpha*pred_m1 + (1-best_alpha)*pred_m2



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

X = train[features].values
y = train[TARGET_COL].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))

for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    
    oof_preds[val_idx] = preds

# final RMSE
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print("OOF RMSE:", rmse)


# ========= Choose the final strategy =========
cv1 = np.mean(fold_mae_m1)
cv2 = np.mean(fold_mae_m2)
best_cv = min(cv1, cv2, best_mae)

if best_cv == best_mae:
    strategy = "blend"
    final_pred = pred_blend
elif best_cv == cv1:
    strategy = "m1"
    final_pred = pred_m1
else:
    strategy = "m2"
    final_pred = pred_m2

print(f"Chosen strategy = {strategy}")

# ========= Build submission =========
sub = pd.DataFrame({
    ID_COL: test[ID_COL].values if ID_COL in test.columns else np.arange(len(test)),
    TARGET_COL: final_pred
})

# Sort by ID to be nice (if ID is sortable)
try:
    sub = sub.sort_values(by=ID_COL)
except Exception:
    pass


sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}")
print(sub.head())

