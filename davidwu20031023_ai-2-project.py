# %% Cell 0 — 安裝所需套件
!pip install polars tqdm keras torch colorama lightgbm xgboost catboost
!pip install mord


# ===== Cell 1 ── 基本 import 與全域設定 =====
import os, gc, warnings, numpy as np, pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# sklearn
from sklearn.model_selection   import train_test_split, GridSearchCV
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base              import clone
from sklearn.metrics           import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score, cohen_kappa_score
)

# models
from xgboost          import XGBRegressor, XGBClassifier
from catboost         import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None



# ===== Cell 2 ── 讀取 train.csv，處理 Season，刪除 PCIAT-PCIAT_01~20，去除 sii 缺失 =====
DATA_DIR = '/kaggle/input/child-mind-institute-problematic-internet-use'
train = pd.read_csv(f'{DATA_DIR}/train.csv')

# Season → 數值
season_map = {'Spring':0, 'Summer':1, 'Fall':2, 'Winter':3}
for c in train.columns:
    if c.endswith('Season'):
        train[c] = train[c].map(season_map)

# 刪除 PCIAT-PCIAT_01~20
drop_pciat = [c for c in train.columns if c.startswith('PCIAT-PCIAT_') and c != 'PCIAT-PCIAT_Total']
train.drop(columns=drop_pciat, inplace=True)

# 丟掉 sii 缺失
train = train.dropna(subset=['sii']).reset_index(drop=True)
print("clean train shape:", train.shape)



# ===== Cell 3 ── 時序特徵萃取函式（與之前相同）=====
def process_file(dirname):
    df = pd.read_parquet(os.path.join(dirname,'part-0.parquet'))
    df.drop(columns='step', errors='ignore', inplace=True)
    feats = np.hstack([
        df.mean().values, df.std().values,
        df.min().values,  df.max().values,
        (df[['X','Y','Z']]!=0).mean().values,
        [(df['enmo']<0.01).mean()]
    ])
    cols = df.columns.tolist()
    names = ([f"{c}_mean" for c in cols]+[f"{c}_std" for c in cols]+
             [f"{c}_min"  for c in cols]+[f"{c}_max" for c in cols]+
             [f"{c}_active_ratio" for c in ['X','Y','Z']] + ['enmo_still_ratio'])
    sid = os.path.basename(dirname).split('=')[1]
    return feats, names, sid

def load_time_series(base):
    subdirs=[os.path.join(base,d) for d in os.listdir(base)]
    res=[]
    with ThreadPoolExecutor() as ex:
        for r in tqdm(ex.map(process_file, subdirs), total=len(subdirs), desc='TS'):
            res.append(r)
    feats, names, ids = zip(*res)
    df = pd.DataFrame(np.vstack(feats), columns=names[0]); df['id']=ids
    return df



# ===== Cell 4 ── 合併時序特徵 =====
train_ts = load_time_series(f'{DATA_DIR}/series_train.parquet')
train = train.merge(train_ts,on='id',how='left')
print("merged shape:", train.shape)



# ===== Cell 5 ── 切 labels，分 80/20，先不做任何變換 =====
y_clf = train['sii'].astype(int)
y_reg = train['PCIAT-PCIAT_Total'].astype(float)
X_raw = train.drop(columns=['sii','PCIAT-PCIAT_Total','id'])

X_tr_raw, X_val_raw, y_tr_clf, y_val_clf, y_tr_reg, y_val_reg = train_test_split(
    X_raw, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf)
print("split shapes:", X_tr_raw.shape, X_val_raw.shape)



# ===== Cell 6 ── 改良的前處理與特徵篩選 (修正版) =====
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# 1) 將數值 / 二元 / 類別 列出
num_cols = [c for c in X_tr_raw.columns
            if X_tr_raw[c].dtype in (np.float64, np.int64) and
               len(X_tr_raw[c].dropna().unique()) > 2]

bin_cols = [c for c in X_tr_raw.columns
            if set(X_tr_raw[c].dropna().unique()) <= {0, 1}]

cat_cols = [c for c in X_tr_raw.columns if c not in num_cols + bin_cols]

print(f"Numeric: {len(num_cols)}, Binary: {len(bin_cols)}, Categorical: {len(cat_cols)}")

# 2) ColumnTransformer
numeric_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler())
])

binary_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent'))     # 0/1 缺失 → 補眾數
])

categorical_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),    # 類別缺失 → 補眾數
    ('one', OneHotEncoder(handle_unknown='ignore',
                          sparse=False,
                          dtype=np.float32))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('bin', binary_pipe,  bin_cols),
    ('cat', categorical_pipe, cat_cols)
])

# 3) fit_transform / transform
X_tr_prep = preprocessor.fit_transform(X_tr_raw)
X_val_prep = preprocessor.transform(X_val_raw)

# 4) SelectKBest 取前 k 個相關特徵
k = 9
selector = SelectKBest(score_func=f_regression, k=min(k, X_tr_prep.shape[1]))
selector.fit(X_tr_prep, y_tr_reg)
mask = selector.get_support()

# 取得對應特徵名稱
def get_feature_names(ct):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder':
            continue
        if isinstance(trans, Pipeline):
            last = trans.steps[-1][1]
            if isinstance(last, OneHotEncoder):
                names.extend(last.get_feature_names_out(cols))
            else:
                names.extend(cols)
        elif isinstance(trans, OneHotEncoder):
            names.extend(trans.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return names

all_feat_names = get_feature_names(preprocessor)
selected_names = [f for f, keep in zip(all_feat_names, mask) if keep]

print(f"\nSelectKBest 保留 {len(selected_names)} 個特徵：")
for feat in selected_names:
    print(" -", feat)

# 最終輸出給模型
X_train = selector.transform(X_tr_prep)
X_val   = selector.transform(X_val_prep)



# ===== Cell 7 ── 共用前處理 (imputer+scaler 已內嵌於 X_tr / X_val) =====
# 將已篩選後的矩陣作為模型輸入，不再額外 scaler



# ===== Cell 8 ── 模型超參數配置不變 =====
param_reg = {
    'xgb': {'model': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'grid' : {'model__learning_rate':[0.03,0.1],
                      'model__n_estimators':[300,400]}},
    'cat': {'model': CatBoostRegressor(verbose=0, loss_function='RMSE', random_state=42),
            'grid' : {'model__depth':[6,8]}},
    'ada': {'model': AdaBoostRegressor(random_state=42),
            'grid' : {'model__n_estimators':[100,200],
                      'model__learning_rate':[0.05,0.1]}}
}

param_clf = {
    'xgb': {'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'grid' : {'model__learning_rate':[0.03,0.1],
                      'model__n_estimators':[300,400]}},
    'cat': {'model': CatBoostClassifier(verbose=0, random_state=42),
            'grid' : {'model__depth':[6,8]}},
    'ada': {'model': AdaBoostClassifier(random_state=42),
            'grid' : {'model__n_estimators':[100,200],
                      'model__learning_rate':[0.05,0.1]}}
}



# ===== Cell 9 ── 回歸模型 GridSearchCV + 自動閾值搜索 + 報告 =====
def raw2sii(v, th):  # th = (t1, t2, t3)
    return np.digitize(v, bins=th)

labels = [0, 1, 2, 3]

for name, spec in param_reg.items():
    pipe = Pipeline([('model', spec['model'])])
    gs = GridSearchCV(pipe,
                      spec['grid'],
                      cv=5,
                      scoring='neg_root_mean_squared_error',
                      n_jobs=-1)
    gs.fit(X_train, y_tr_reg)   # 使用 y_tr_reg 而非 y_train_reg
    best = gs.best_estimator_
    y_pred = best.predict(X_val)

    print(f"\n=== {name.upper()} 回歸最佳參數 ===\n{gs.best_params_}\n")
    print(f"[{name.upper()} Regression] "
          f"MSE={mean_squared_error(y_val_reg, y_pred):.4f}  "
          f"MAE={mean_absolute_error(y_val_reg, y_pred):.4f}  "
          f"R²={r2_score(y_val_reg, y_pred):.4f}")

    # --- 自動閾值搜尋 (QWK) ---
    mins, maxs = y_pred.min(), y_pred.max()
    cand = np.linspace(mins, maxs, 50)
    best_qwk, best_th = -1, None

    for t1 in cand:
        for t2 in cand:
            if t2 <= t1:
                continue
            for t3 in cand:
                if t3 <= t2:
                    continue
                mk = raw2sii(y_pred, (t1, t2, t3))
                qwk = cohen_kappa_score(y_val_clf, mk, weights='quadratic')
                if qwk > best_qwk:
                    best_qwk, best_th = qwk, (t1, t2, t3)

    mapped = raw2sii(y_pred, best_th)
    print(f"\n[{name.upper()} 最佳閾值] "
          f"{tuple(round(x, 3) for x in best_th)} → QWK={best_qwk:.4f}")
    print(classification_report(y_val_clf, mapped, labels=labels, digits=4))
    print(f"  Accuracy={accuracy_score(y_val_clf, mapped):.4f}")



# ===== Cell 10 ── 分類模型 GridSearchCV + 報告 =====
for name, spec in param_clf.items():
    pipe = Pipeline([('model', spec['model'])])
    gs = GridSearchCV(pipe,
                      spec['grid'],
                      cv=5,
                      scoring='accuracy',
                      n_jobs=-1)
    gs.fit(X_train, y_tr_clf)  # 使用 y_tr_clf 而非 y_train_clf
    best = gs.best_estimator_
    y_pred = best.predict(X_val)
    print(f"\n=== {name.upper()} 分類最佳參數 ===\n{gs.best_params_}\n")
    print(classification_report(y_val_clf, y_pred, labels=labels, digits=4))
    print(f"  Accuracy={accuracy_score(y_val_clf, y_pred):.4f}")


# ===== Cell 11 ── （可選）清理記憶體 =====
del train, train_ts, X_raw, X_tr_raw, X_val_raw
gc.collect()


