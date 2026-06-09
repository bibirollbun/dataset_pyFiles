# !pip -q install catboost shap lightgbm --upgrade

import os, sys, warnings, math, gc, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

import shap
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DATA_DIR = Path('/kaggle/input/playground-series-s5e10')
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH  = DATA_DIR / 'test.csv'

N_ROWS_TRAIN = None  

pd.set_option('display.max_columns', 200)
print("âœ… Imports ready. Seed =", SEED)


train = pd.read_csv(TRAIN_PATH, nrows=N_ROWS_TRAIN)
test  = pd.read_csv(TEST_PATH)


print(train.shape, test.shape)
display(train.head())
display(train.describe(include='all').T.head(20))


TARGET = 'accident_risk'
IDCOL  = 'id'

cat_cols  = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present','public_road','holiday','school_season']
num_cols  = ['num_lanes','curvature','speed_limit','num_reported_accidents']

# æ£€æŸ¥åˆ—æ˜¯å�¦é½�å…¨ï¼ˆå®¹é”™ï¼šä¸�å�Œç‰ˆæœ¬å­—æ®µå¤§å°�å†™æˆ–ç¼ºå¤±æ—¶æ��ç¤ºï¼‰
for col in [IDCOL] + cat_cols + bool_cols + num_cols + [TARGET]:
    if col not in train.columns and col != TARGET:  # TARGET ä»…åœ¨ train
        print(f"âš ï¸� Column {col} not in train/testã€‚")

print("âœ… Data loaded.")


# ç»Ÿä¸€å¸ƒå°”ï¼šæ�¥å�— True/Falseã€�'TRUE'/'FALSE'ã€�1/0
def to_bool01(s: pd.Series):
    if s.dtype == 'bool':
        return s.astype('int8')
    if s.dtype.name.startswith('int') or s.dtype.name.startswith('uint'):
        return s.astype('int8').clip(0,1)
    if s.dtype.name.startswith('float'):
        return s.fillna(0).astype('int8').clip(0,1)
    # å…¶ä»–æƒ…å†µï¼šå­—ç¬¦ä¸²
    mapping = {'TRUE':1,'True':1,'true':1,'FALSE':0,'False':0,'false':0,'T':1,'F':0,'Y':1,'N':0,'Yes':1,'No':0}
    return s.map(mapping).fillna(0).astype('int8')

# ç±»å�‹è½¬æ�¢
for c in cat_cols:
    if c in train.columns:
        train[c] = train[c].astype('category')
    if c in test.columns:
        test[c]  = test[c].astype('category')

for c in bool_cols:
    if c in train.columns:
        train[c] = to_bool01(train[c])
    if c in test.columns:
        test[c]  = to_bool01(test[c])

# ç¼ºå¤±å€¼ç»Ÿè®¡
def missing_report(df, name):
    miss = df.isna().mean().sort_values(ascending=False)
    print(f"\nğŸ”� Missing ratio in {name}:")
    display(miss[miss>0].to_frame('missing_ratio'))

missing_report(train, 'train')
missing_report(test, 'test')

# ç®€å�•å¡«è¡¥ï¼šæ•°å€¼ä¸­ä½�æ•°ï¼Œç±»åˆ«ä¼—æ•°
for c in num_cols:
    if c in train.columns:
        median = train[c].median()
        train[c] = train[c].fillna(median)
    if c in test.columns:
        test[c]  = test[c].fillna(median)

for c in cat_cols:
    if c in train.columns:
        mode = train[c].mode(dropna=True)
        mode = mode.iloc[0] if not mode.empty else 'unknown'
        train[c] = train[c].cat.add_categories(['unknown']).fillna(mode)
    if c in test.columns:
        test[c] = test[c].cat.add_categories(['unknown']).fillna(mode)

print("âœ… Dtypes converted & missing imputed.")
train.dtypes


train.head()


train.speed_limit.min()


def sanity_check_and_clip(df: pd.DataFrame, is_train=False):
    # è´Ÿå€¼/æ— æ•ˆå€¼æ£€æŸ¥ä¸�æˆªæ–­ï¼ˆæ ¹æ�®å¸¸è¯†ï¼‰
    if 'num_lanes' in df: 
        df['num_lanes'] = df['num_lanes'].clip(lower=1, upper=12)
    if 'curvature' in df:
        df['curvature'] = df['curvature'].clip(lower=0)  # ä¸�å…�è®¸è´Ÿæ›²ç�‡
    if 'speed_limit' in df:
        df['speed_limit'] = df['speed_limit'].clip(lower=10, upper=200)
    if 'num_reported_accidents' in df:
        df['num_reported_accidents'] = df['num_reported_accidents'].clip(lower=0)
    if is_train and TARGET in df:
        df[TARGET] = df[TARGET].clip(0,1)
    return df

train = sanity_check_and_clip(train, is_train=True)
test  = sanity_check_and_clip(test,  is_train=False)

print("âœ… Sanity check done.")
train[num_cols + ([TARGET] if TARGET in train else [])].describe()



# ç›®æ ‡åˆ†å¸ƒ
plt.figure(figsize=(5,3))
sns.histplot(train[TARGET], bins=20, kde=True)
plt.title('Target Distribution (accident_risk)')
plt.show()


# æ•°å€¼ç‰¹å¾�ä¸�ç›®æ ‡å…³ç³»ï¼ˆæ•£ç‚¹+ä½�esså›�å½’ï¼‰
for c in ['speed_limit','curvature','num_lanes','num_reported_accidents']:
    if c in train.columns:
        plt.figure(figsize=(5,3))
        sns.scatterplot(x=train[c], y=train[TARGET], alpha=0.7)
        sns.regplot(x=train[c], y=train[TARGET], scatter=False, lowess=True)
        plt.title(f'{c} vs accident_risk')
        plt.show()

print("âœ… Basic EDA done.")


# ç±»åˆ«åˆ†ç»„å�‡å€¼
for c in cat_cols:
    if c in train.columns:
        grp = train.groupby(c)[TARGET].mean().sort_values(ascending=False)
        print(f"\nğŸ“Š Mean target by {c}:")
        display(grp.to_frame('mean_accident_risk'))



# è®­ç»ƒ/æµ‹è¯•åˆ†å¸ƒå¯¹æ¯”ï¼ˆç®€å�•å�¯è§†ï¼‰
def dist_compare(train_s, test_s, title, bins=20):
    plt.figure(figsize=(5,3))
    sns.kdeplot(train_s, label='train', fill=True, alpha=0.3)
    sns.kdeplot(test_s,  label='test',  fill=True, alpha=0.3)
    plt.legend(); plt.title(title); plt.show()

for c in num_cols:
    if c in test.columns:
        dist_compare(train[c], test[c], f'Distribution: {c}')

# ç±»åˆ«åˆ†å¸ƒå¯¹æ¯”ï¼ˆtopç±»ï¼‰
for c in cat_cols:
    if c in test.columns:
        plt.figure(figsize=(6,3))
        t_counts = train[c].value_counts(normalize=True)
        s_counts = test[c].value_counts(normalize=True)
        comp = pd.concat([t_counts, s_counts], axis=1, keys=['train','test']).fillna(0)
        comp.plot(kind='bar', figsize=(6,3))
        plt.title(f'Category distribution: {c}')
        plt.show()

print("âœ… Drift checks (visual) done.")



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # è®¡æ•°å�‹å�˜æ�¢
    if 'num_reported_accidents' in out:
        out['num_reported_accidents_log1p'] = np.log1p(out['num_reported_accidents'])
    # å¤šé¡¹å¼�é¡¹
    if 'curvature' in out:
        out['curvature_sq'] = out['curvature']**2
    if 'speed_limit' in out:
        out['speed_limit_sq'] = out['speed_limit']**2
    # ç›´è§‰äº¤äº’é¡¹
    if set(['speed_limit','curvature']).issubset(out.columns):
        out['danger_index'] = out['speed_limit'] * out['curvature']
    # ç±»åˆ«äº¤äº’ï¼ˆä½�ç»´ one-hot ä¸‹çš„æ˜¾å¼�äº¤äº’ï¼‰
    if set(['lighting','weather']).issubset(out.columns):
        out['lighting_weather'] = out['lighting'].astype(str) + '|' + out['weather'].astype(str)
        out['lighting_weather'] = out['lighting_weather'].astype('category')
    if set(['road_type','speed_limit']).issubset(out.columns):
        out['road_speed'] = out['road_type'].astype(str) + '|' + pd.cut(out['speed_limit'], bins=[0,40,60,80,120,200], include_lowest=True).astype(str)
        out['road_speed'] = out['road_speed'].astype('category')
    return out

train_fe = add_features(train)
test_fe  = add_features(test)

# æ›´æ–°åˆ—æ¸…å�•
cat_cols_ext  = cat_cols + [c for c in ['lighting_weather','road_speed'] if c in train_fe.columns]
num_cols_ext  = num_cols + [c for c in ['num_reported_accidents_log1p','curvature_sq','speed_limit_sq','danger_index'] if c in train_fe.columns]
bool_cols_ext = bool_cols

print("ğŸ“Œ Feature columns (extended):")
print("Categorical:", cat_cols_ext)
print("Boolean:", bool_cols_ext)
print("Numeric:", num_cols_ext)


train_fe


# å›ºå®šéªŒè¯�é›†ï¼ˆ20%ï¼‰ï¼Œå¯¹ y åˆ†æ¡¶å��å�š stratify
y = train_fe[TARGET].values
bins = np.minimum(9, (y * 10).astype(int))  # 0-1 -> 10æ¡¶

train_idx, valid_idx = train_test_split(
    np.arange(len(train_fe)),
    test_size=0.2,
    random_state=SEED,
    stratify=bins
)

X_train = train_fe.iloc[train_idx].reset_index(drop=True)
y_train = train_fe.iloc[train_idx][TARGET].reset_index(drop=True)
X_valid = train_fe.iloc[valid_idx].reset_index(drop=True)
y_valid = train_fe.iloc[valid_idx][TARGET].reset_index(drop=True)

print(f"Train/Valid split: {X_train.shape} / {X_valid.shape}")

# KæŠ˜åˆ†å±‚å·¥å…·ï¼ˆå��ç»­CVç”¨ï¼‰
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("âœ… Split ready.")



y_pred_mean = np.full_like(y_valid, fill_value=y_train.mean(), dtype=float)
rmse_mean = mean_squared_error(y_valid, y_pred_mean, squared=False)
print(f"Baseline-0 (global mean) RMSE on holdout: {rmse_mean:.5f}")


# One-Hot + Ridge å›�å½’
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
pre = ColumnTransformer(
    transformers=[
        ('ohe', ohe, cat_cols_ext + ['lighting_weather','road_speed'] if 'lighting_weather' in train_fe and 'road_speed' in train_fe else cat_cols_ext),
        ('pass_bool', 'passthrough', bool_cols_ext),
        ('pass_num', 'passthrough', num_cols_ext),
    ],
    remainder='drop'
)

ridge = Ridge(alpha=1.0, random_state=SEED)

pipe_ridge = Pipeline(steps=[('pre', pre), ('ridge', ridge)])

pipe_ridge.fit(X_train, y_train)
pred_valid_ridge = pipe_ridge.predict(X_valid)
rmse_ridge = mean_squared_error(y_valid, pred_valid_ridge, squared=False)
print(f"Baseline-1 Ridge RMSE (holdout): {rmse_ridge:.5f}")


features_cb = cat_cols_ext + bool_cols_ext + num_cols_ext
cat_idx = [features_cb.index(c) for c in cat_cols_ext]

train_pool = Pool(X_train[features_cb], label=y_train, cat_features=cat_idx)
valid_pool = Pool(X_valid[features_cb], label=y_valid, cat_features=cat_idx)

cb_params = dict(
    loss_function='RMSE',
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    iterations=500,
    od_type='Iter',
    od_wait=200,
    verbose=False
)

model_cb = CatBoostRegressor(**cb_params)
model_cb.fit(train_pool, eval_set=valid_pool, verbose=False)

pred_valid_cb = model_cb.predict(valid_pool)
rmse_cb = mean_squared_error(y_valid, pred_valid_cb, squared=False)
print(f"Baseline-2A CatBoost RMSE (holdout): {rmse_cb:.5f}")



# ä½¿ç”¨ä¸� Ridge ç›¸å�Œçš„é¢„å¤„ç�†ï¼ˆOne-Hotï¼‰ï¼Œå†�å–‚ç»™ LightGBM
Xtr_enc = pipe_ridge.named_steps['pre'].fit_transform(X_train)
Xva_enc = pipe_ridge.named_steps['pre'].transform(X_valid)

lgb_train = lgb.Dataset(Xtr_enc, label=y_train)
lgb_valid = lgb.Dataset(Xva_enc, label=y_valid, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'lambda_l2': 2.0,
    'seed': SEED,
    'verbose': -1
}

model_lgb = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train','valid'],
    # early_stopping_rounds=200,
    # verbose_eval=False
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False), # å°† early_stopping_rounds æ”¾å…¥å›�è°ƒ
        lgb.log_evaluation(period=0) # ä½¿ç”¨ log_evaluation æ›¿ä»£ verbose_eval=False
    ]
)

pred_valid_lgb = model_lgb.predict(Xva_enc, num_iteration=model_lgb.best_iteration)
rmse_lgb = mean_squared_error(y_valid, pred_valid_lgb, squared=False)
print(f"Baseline-2B LightGBM RMSE (holdout): {rmse_lgb:.5f}")


def cv_catboost(df: pd.DataFrame, features: List[str], cat_cols: List[str], y: np.ndarray, n_splits=5) -> Tuple[np.ndarray, float, List[CatBoostRegressor]]:
    cat_idx = [features.index(c) for c in cat_cols]
    bins = np.minimum(9, (y * 10).astype(int))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof = np.zeros(len(df))
    models = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df, bins), 1):
        trn_pool = Pool(df.iloc[trn_idx][features], label=y[trn_idx], cat_features=cat_idx)
        val_pool = Pool(df.iloc[val_idx][features], label=y[val_idx], cat_features=cat_idx)

        model = CatBoostRegressor(**cb_params)
        model.fit(trn_pool, eval_set=val_pool, verbose=False)
        oof[val_idx] = model.predict(val_pool)
        models.append(model)
        print(f"Fold {fold} RMSE:", mean_squared_error(y[val_idx], oof[val_idx], squared=False))
    cv_rmse = mean_squared_error(y, oof, squared=False)
    return oof, cv_rmse, models

oof_cb, cv_rmse_cb, models_cb = cv_catboost(train_fe, features_cb, cat_cols_ext, y=y, n_splits=5)
print(f"CV RMSE (CatBoost): {cv_rmse_cb:.5f}")


# 1) Gain-based importance
imp = pd.DataFrame({
    'feature': features_cb,
    'importance_gain': model_cb.get_feature_importance(train_pool, type='FeatureImportance')
}).sort_values('importance_gain', ascending=False)
display(imp.head(20))

plt.figure(figsize=(6,6))
sns.barplot(data=imp.head(20), x='importance_gain', y='feature')
plt.title('CatBoost Feature Importance (Gain, top 20)')
plt.show()

# 2) SHAPï¼ˆå¯¹ holdout ä¸Šå°‘é‡�æ ·æœ¬ï¼Œé�¿å…�å¤ªæ…¢ï¼‰
explainer = shap.TreeExplainer(model_cb)
sample_idx = np.random.choice(len(X_valid), size=min(200, len(X_valid)), replace=False)
X_valid_sample = X_valid.iloc[sample_idx][features_cb]
shap_values = explainer.shap_values(X_valid_sample)

shap.summary_plot(shap_values, X_valid_sample, plot_type='dot', show=True)



from lightgbm.callback import early_stopping
model_lgb_sklearn = lgb.LGBMRegressor(**lgb_params)
model_lgb_sklearn.fit(
    Xtr_enc,
    y_train,
    eval_set=[(Xva_enc, y_valid)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=200, verbose=False)]
)

# ç”¨ç¼–ç �å��çš„éªŒè¯�é›†å�š permutation importance ï¼ˆæ›´ç›´è§‚åœ°çœ‹å¯¹ RMSE çš„å½±å“�ï¼‰
result = permutation_importance(
    estimator=model_lgb_sklearn,
    X=Xva_enc,
    y=y_valid,
    scoring='neg_root_mean_squared_error',
    n_repeats=10,
    random_state=SEED
)

# ä»� One-Hot è�·å�–ç‰¹å¾�å��
ohe_fitted: OneHotEncoder = pipe_ridge.named_steps['pre'].named_transformers_['ohe']
onehot_feature_names = list(ohe_fitted.get_feature_names_out())

bool_feature_names = bool_cols_ext
num_feature_names  = num_cols_ext
all_feature_names  = onehot_feature_names + bool_feature_names + num_feature_names

pi_df = pd.DataFrame({
    'feature': all_feature_names,
    'import_mean': result.importances_mean,
    'import_std': result.importances_std
}).sort_values('import_mean', ascending=False)

display(pi_df.head(25))

plt.figure(figsize=(6,7))
sns.barplot(data=pi_df.head(25), x='import_mean', y='feature')
plt.title('Permutation Importance (LightGBM, top 25)')
plt.show()



# é€‰æ‹© Top-K çš„ One-Hot ç‰¹å¾�ï¼Œè§‚å¯Ÿæ€§èƒ½å�˜åŒ–
K = 30  # å�¯ä»¥è°ƒï¼ˆå°�æ ·æœ¬æ—¶ä¸�è¦�å¤ªå¤§ï¼‰
topk_feats = set(pi_df.head(K)['feature'].tolist())

# å°†ç¼–ç �å��çš„çŸ©é˜µæˆªå�–åˆ° Top-K åˆ—ï¼ˆæŒ‰å��ç§°ç­›åˆ—ç´¢å¼•ï¼‰
name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
keep_idx = [name_to_idx[n] for n in topk_feats if n in name_to_idx]

Xtr_topk = Xtr_enc[:, keep_idx]
Xva_topk = Xva_enc[:, keep_idx]

lgb_params_fs = lgb_params.copy()
lgb_params_fs['lambda_l2'] = 3.0  # ç•¥åŠ å¼ºæ­£åˆ™
model_lgb_fs = lgb.train(
    lgb_params_fs,
    lgb.Dataset(Xtr_topk, label=y_train),
    num_boost_round=5000,
    valid_sets=[lgb.Dataset(Xva_topk, label=y_valid)],
    valid_names=['valid'],
    # early_stopping_rounds=200,
    # verbose_eval=False
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False), # å°† early_stopping_rounds æ”¾å…¥å›�è°ƒ
        lgb.log_evaluation(period=0) # ä½¿ç”¨ log_evaluation æ›¿ä»£ verbose_eval=False
    ]
)
pred_valid_lgb_fs = model_lgb_fs.predict(Xva_topk, num_iteration=model_lgb_fs.best_iteration)
rmse_lgb_fs = mean_squared_error(y_valid, pred_valid_lgb_fs, squared=False)
print(f"LightGBM + FeatureSelection Top-{K} RMSE: {rmse_lgb_fs:.5f}")



# ä½ å�¯ä»¥åœ¨è¿™é‡Œå¿«é€Ÿè¯•éªŒæ›´å¤šäº¤äº’ï¼ˆç¤ºä¾‹ï¼‰
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # äº¤äº’ï¼štime_of_day Ã— lighting
    if set(['time_of_day','lighting']).issubset(out.columns):
        out['tod_light'] = out['time_of_day'].astype(str) + '|' + out['lighting'].astype(str)
        out['tod_light'] = out['tod_light'].astype('category')
    # äº¤äº’ï¼špublic_road Ã— speed_limitï¼ˆå¸ƒå°” Ã— è¿�ç»­ï¼‰
    if set(['public_road','speed_limit']).issubset(out.columns):
        out['is_public_speed'] = out['public_road'] * out['speed_limit']
    return out

train_fx = add_interactions(train_fe)
test_fx  = add_interactions(test_fe)

# ä½¿ç”¨ CatBoost é‡�æ–°éªŒè¯�ï¼ˆholdoutï¼‰
features_cb_fx = [c for c in features_cb]  # æ‹·è´�
for c in ['tod_light','is_public_speed']:
    if c in train_fx.columns and c not in features_cb_fx:
        features_cb_fx.append(c)

cat_cols_fx = cat_cols_ext + [c for c in ['tod_light'] if c in train_fx.columns]

train_pool_fx = Pool(train_fx[features_cb_fx].iloc[train_idx], label=y_train, cat_features=[features_cb_fx.index(c) for c in cat_cols_fx if c in features_cb_fx])
valid_pool_fx = Pool(train_fx[features_cb_fx].iloc[valid_idx], label=y_valid, cat_features=[features_cb_fx.index(c) for c in cat_cols_fx if c in features_cb_fx])

model_cb_fx = CatBoostRegressor(**cb_params)
model_cb_fx.fit(train_pool_fx, eval_set=valid_pool_fx, verbose=False)

pred_valid_cb_fx = model_cb_fx.predict(valid_pool_fx)
rmse_cb_fx = mean_squared_error(y_valid, pred_valid_cb_fx, squared=False)
print(f"CatBoost + extra interactions RMSE (holdout): {rmse_cb_fx:.5f}")


# ç”¨å…¨è®­ç»ƒé›†æ‹Ÿå�ˆ + é¢„æµ‹æµ‹è¯•é›†
best_features = features_cb_fx  # æ�¢æˆ�ä½ æœ€å¥½çš„ç‰¹å¾�é›†å�ˆ
best_cat_cols = cat_cols_fx

full_pool = Pool(train_fx[best_features], label=train_fx[TARGET], cat_features=[best_features.index(c) for c in best_cat_cols if c in best_features])
test_pool = Pool(test_fx[best_features], cat_features=[best_features.index(c) for c in best_cat_cols if c in best_features])

final_model = CatBoostRegressor(**cb_params)
final_model.fit(full_pool, verbose=False)

test_pred = final_model.predict(test_pool)
test_pred = np.clip(test_pred, 0, 1)

sub = pd.DataFrame({IDCOL: test[IDCOL], 'accident_risk': test_pred})
save_path = '/kaggle/working/submission.csv'
sub.to_csv(save_path, index=False)
print(f"âœ… submission.csv saved at: {save_path}")
display(sub.head())



sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

