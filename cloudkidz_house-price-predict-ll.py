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


!pip install -q geohash2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.base import clone
import lightgbm as lgb
from tqdm import tqdm

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import geohash2 as geohash

train_path = '/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv'
test_path = '/kaggle/input/prediction-interval-competition-ii-house-price/test.csv'
sam_path = '/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv'


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sam_path)

print(train_df.shape)
print(train_df.columns)
train_df.head()


missing_values = train_df.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)


def zoning_group_classify(z):
    if pd.isna(z): return 'other'
    z = z.upper()
    if 'SF' in z: return 'SF'
    elif 'MR' in z: return 'MR'
    elif 'NC' in z: return 'NC'
    elif 'P' in z: return 'P'
    elif 'HR' in z or 'IG' in z: return 'other'
    return 'other'

def encode_dataset(df, is_train=True, top_cities=None, top_supermarket=None, top_sale_warning=None):
    
    encoded = pd.DataFrame()
    confirmed_one_hot = ['join_status', 'condition', 'stories', 'grade', 'fbsmt_grade', 'present_use']

    direct_add_cols = [
        'id', 'sale_price', 'join_year', 'latitude', 'longitude',
        'area', 'land_val', 'imp_val', 'year_built', 'year_reno',
        'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt',
        'beds', 'garb_sqft', 'gara_sqft', 'golf', 'greenbelt',
        'bath_full', 'bath_3qtr', 'bath_half', 'wfnt', 'noise_traffic',
        'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
        'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm',
        'view_otherwater', 'view_other'
        #'subdivision','sale_nbr'æ²’æœ‰å�šé€™å€‹ ç”¨æ„�ä¸�å¤§
    ]

    onehot_df = pd.get_dummies(df[confirmed_one_hot], drop_first=False)
    encoded = pd.concat([encoded, onehot_df], axis=1)

    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    encoded['sale_year'] = df['sale_date'].dt.year
    encoded['sale_month'] = df['sale_date'].dt.month
    encoded['sale_season'] = ((encoded['sale_month'] - 1) // 3 + 1)

    for col in direct_add_cols:
        if col in df.columns:
            encoded[col] = df[col]

    if is_train:
        top_cities = df['city'].value_counts().nlargest(10).index.tolist()
        top_supermarket = df['submarket'].value_counts().nlargest(10).index.tolist()
        top_sale_warning = df['sale_warning'].value_counts().nlargest(15).index.tolist()

    encoded['city_simplified'] = df['city'].apply(lambda x: x if x in top_cities else 'other')
    encoded['submarket_simplified'] = df['submarket'].apply(lambda x: x if x in top_supermarket else 'other')
    encoded['sale_warning_simplified'] = df['sale_warning'].apply(lambda x: x if x in top_sale_warning else 'other')

    city_dummy = pd.get_dummies(encoded['city_simplified'], prefix='city', drop_first=False)
    submarket_dummy = pd.get_dummies(encoded['submarket_simplified'], prefix='submarket', drop_first=False)
    sale_warning_dummy = pd.get_dummies(encoded['sale_warning_simplified'], prefix='sale_warning', drop_first=False)
    encoded = pd.concat([encoded, city_dummy, submarket_dummy, sale_warning_dummy], axis=1)

    encoded['zoning_group'] = df['zoning'].apply(zoning_group_classify)
    zoning_dummy = pd.get_dummies(encoded['zoning_group'], prefix='zoning_group', drop_first=False)
    encoded = pd.concat([encoded, zoning_dummy], axis=1)
    encoded.drop(columns=['zoning_group', 'city_simplified', 'submarket_simplified', 'sale_warning_simplified'], inplace=True)

    encoded['age'] = encoded['sale_year'] - encoded['year_built']
    encoded['renovated'] = np.where(encoded['year_reno'] > 0, 1, 0)
    encoded['years_since_reno'] = np.where(encoded['renovated'], encoded['sale_year'] - encoded['year_reno'], 0)
    encoded['total_baths'] = encoded['bath_full'] + 0.75 * encoded['bath_3qtr'] + 0.5 * encoded['bath_half']
    encoded['total_value'] = encoded['land_val'] + encoded['imp_val']
    encoded['living_area'] = encoded['sqft'] + encoded['sqft_fbsmt']


    encoded["floor_ratio"] = np.where(
    encoded["sqft_lot"] == 0,
    0,
    encoded["sqft"] / encoded["sqft_lot"]
    )

    encoded["is_large_house"] = (encoded["sqft"] > 3000).astype(int)
    encoded["is_recent_reno"] = (encoded["years_since_reno"] <= 5).astype(int)
    encoded["bath_per_bed"] = encoded["total_baths"] / encoded["beds"]
    encoded["bath_per_bed"] = encoded["bath_per_bed"].replace([np.inf, -np.inf], 0).fillna(0)
    
    return encoded, top_cities, top_supermarket, top_sale_warning

'''def pca_train_test(encoded, feature, scaler=None, pca=None, kmeans=None):
    """
    å¦‚æ�œå‚³å…¥ scaler/pca/kmeans == None â†’ åœ¨ df ä¸Š fit
    å�¦å‰‡å�ªå�š transform / predict
    å›�å‚³ (encoded, scaler, pca, kmeans)
    """
    # 1. æ¨™æº–åŒ–
    if scaler is None:
        scaler = StandardScaler().fit(encoded[feature])
    X_scaled = scaler.transform(encoded[feature])

    # 2. PCA
    if pca is None:
        pca = PCA(n_components=3, random_state=42).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # 3. KMeans
    if kmeans is None:
        kmeans = KMeans(n_clusters=10, random_state=42).fit(X_pca)
    encoded["pca_region_cluster"] = kmeans.predict(X_pca)

    # 4. Oneâ€‘hot
    dummies = pd.get_dummies(encoded["pca_region_cluster"], prefix="pca_region")
    encoded = pd.concat([encoded, dummies], axis=1)
    encoded.drop(columns=["pca_region_cluster"], inplace=True)

    return encoded, scaler, pca, kmeans'''



def clean_features(df, log_cols=None, clip_cols=None, add_log_target=False, log_target_col="sale_price_log"):
    """
    å°� df å�šå°�æ•¸åŒ– / clipã€‚
    Parameters
    ----------
    df : pd.DataFrame
    log_cols : list[str]  è¦� log1p çš„æ¬„ä½�ã€‚é �è¨­å¸¸è¦‹æ•¸å€¼é•·å°¾æ¬„ã€‚
    clip_cols : list[str] è¦� clip çš„æ¬„ä½�ï¼ˆå�¯ Noneï¼‰ã€‚
    add_log_target : bool  æ˜¯å�¦å�¦å¤–ç”¢ç”Ÿ log ç‰ˆç›®æ¨™æ¬„ï¼ˆä¸�è¦†å¯«å�Ÿæ¬„ä½�ï¼‰ã€‚
    log_target_col : str   æ–°å¢�ç›®æ¨™æ¬„å��ç¨±ã€‚
    Returns
    -------
    cleaned_df : pd.DataFrame
    """
    df = df.copy()  # ä¸�æ±™æŸ“å‘¼å�«ç«¯

    if log_cols is None:
        log_cols = ['land_val', 'imp_val', 'sqft_lot',
                    'garb_sqft', 'floor_ratio', 'total_value']
    if clip_cols is None:
        clip_cols = ['land_val', 'imp_val', 'sqft_lot']

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    for col in clip_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=1_000_000)

    if add_log_target and 'sale_price' in df.columns:
        df[log_target_col] = np.log1p(df['sale_price'])

    return df

def add_geohash(df, prec=6):
    df = df.copy()
    df['gh6'] = df.apply(
        lambda r: geohash.encode(r.latitude, r.longitude, precision=prec),
        axis=1
    )
    return df



# base features
train_encoded, top_cities, top_supermarket, top_sale_warning = encode_dataset(train_df, is_train=True)
test_encoded , _ , _ , _  = encode_dataset(test_df , is_train=False,
                                           top_cities=top_cities,
                                           top_supermarket=top_supermarket,
                                           top_sale_warning=top_sale_warning)

'''#  PCA + KMeans
pca_features = ['latitude', 'longitude', 'sqft', 'area', 'total_value', 'imp_val']
train_encoded, scaler, pca, kmeans = pca_train_test(train_encoded, pca_features)
test_encoded , _,     _,   _       = pca_train_test(test_encoded , pca_features,
                                                    scaler=scaler, pca=pca, kmeans=kmeans)'''

# geohash 6 
train_encoded = add_geohash(train_encoded)
test_encoded  = add_geohash(test_encoded)

# ---------- TRAIN ----------
gh_train_dum = pd.get_dummies(train_encoded['gh6'], prefix='gh6', drop_first=False)
train_encoded = pd.concat([train_encoded, gh_train_dum], axis=1)
train_encoded.drop(columns=['gh6'], inplace=True)    # å�Ÿ gh6 é¡�åˆ¥æ¬„å�¯ä¸Ÿ

# ---------- TEST ----------
gh_test_dum = pd.get_dummies(test_encoded['gh6'], prefix='gh6', drop_first=False)
test_encoded = pd.concat([test_encoded, gh_test_dum], axis=1)
test_encoded.drop(columns=['gh6'], inplace=True)

train_encoded = clean_features(train_encoded, add_log_target=True)
test_encoded  = clean_features(test_encoded , add_log_target=False)

train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)


train_encoded.dtypes.value_counts()


X = train_encoded.drop(columns=['sale_price', 'sale_price_log', 'id'])
y = train_encoded['sale_price_log']
y_raw = np.expm1(y)


def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)
    return width + (2 / alpha) * (below + above)

def oof_and_hill_climb_two_weights(X, y, model_lower, model_upper, alpha=0.1, n_splits=5, seed=42, steps=100):

    y_raw = np.expm1(y)

    oof_lowers = np.zeros(len(X))
    oof_uppers = np.zeros(len(X))

    fold_lower_models = []
    fold_upper_models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lower_model = clone(model_lower)
        upper_model = clone(model_upper)

        callbacks = [
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=0)
        ]


        lower_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=callbacks
        )

        upper_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=callbacks
        )

        #oof_lowers[val_idx] = lower_model.predict(X_val)
        #oof_uppers[val_idx] = upper_model.predict(X_val)

        # ï¼� lower
        oof_lowers[val_idx] = np.expm1(   # ğŸ‘ˆ é‚„å�Ÿ
        lower_model.predict(X_val, num_iteration=lower_model.best_iteration_)
        )
        # ï¼� upper
        oof_uppers[val_idx] = np.expm1(   # ğŸ‘ˆ é‚„å�Ÿ
        upper_model.predict(X_val, num_iteration=upper_model.best_iteration_)
        )

        fold_lower_models.append(lower_model)
        fold_upper_models.append(upper_model)


    # åˆ�å§‹åŒ–é›™æ¬Šé‡�
    current_w1 = 0.4  # ä¸‹é™� weight
    current_w2 = 0.6  # ä¸Šé™� weight

    best_score = np.inf
    best_weights = (current_w1, current_w2)

    '''for step in range(steps):
        # å¾®èª¿ perturbationï¼Œè®“ weight æœ‰éš¨æ©Ÿæ€§ï¼ˆé�¿å…�å�¡ä½�ï¼‰
        perturb1 = np.random.dirichlet([9])[0] - 0.9
        perturb2 = np.random.dirichlet([9])[0] - 0.9

        w1 = np.clip(current_w1 + 0.1 * perturb1, 0, 1)
        w2 = np.clip(current_w2 + 0.1 * perturb2, 0, 1)

        # é›™æ¬Šé‡�çµ„å�ˆ
        lower_combined = w1 * oof_lowers + (1 - w1) * oof_uppers
        upper_combined = w2 * oof_uppers + (1 - w2) * oof_lowers

        # ä¿®æ­£ï¼šç¢ºä¿�ä¸Šä¸‹é™�æ–¹å�‘æ­£ç¢ºï¼ˆé˜²æ­¢é �æ¸¬ç¯„åœ�éŒ¯ä½�ï¼‰
        lower_combined, upper_combined = np.minimum(lower_combined, upper_combined), np.maximum(lower_combined, upper_combined)

        score = np.mean(winkler_score(y_raw, lower_combined, upper_combined, alpha))

        if score < best_score:
            best_score = score
            best_weights = (w1, w2)
            current_w1, current_w2 = w1, w2
            print(f"[Step {step}] âœ… Improved Score: {best_score:.2f} (w1: {w1:.4f}, w2: {w2:.4f})")'''
    


    grid = np.linspace(0, 1, 401)          # 0.0025 æ­¥
    best_score = np.inf
    best_weights = (0, 0)
    best_cov = 0

    y_raw = np.expm1(y)                    # é‚„å�Ÿå–®ä½�ä¸€æ¬¡å°±å¥½

    for w1 in grid:
        for w2 in grid:
            if w1 > w2:                       # ä¿�æŒ�ä¸‹é™�â‰¤ä¸Šé™�
                continue
            low  = w1 * oof_lowers + (1 - w1) * oof_uppers
            high = w2 * oof_uppers + (1 - w2) * oof_lowers

            score = winkler_score(y_raw, low, high, alpha).mean()
            if score < best_score:
                best_score = score
                best_weights = (w1, w2)
                best_cov = ((y_raw >= low) & (y_raw <= high)).mean()

    print(f"[Grid] best Winkler {best_score:.0f}  "
          f"w1={best_weights[0]:.3f}  w2={best_weights[1]:.3f}  "
          f"cov={best_cov:.3f}")

    return (oof_lowers, oof_uppers, 
            best_weights, best_score , 
            fold_lower_models, fold_upper_models)


models = {
    "lower": lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.05,
        device="cpu",
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        random_state=42
    ),
    "upper": lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.95,
        device="cpu",
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        random_state=42
    )
}


oof_lowers, oof_uppers, best_weights, best_score , fold_lower_models, fold_upper_models= oof_and_hill_climb_two_weights(
    X, y,
    model_lower=models["lower"],
    model_upper=models["upper"],
    alpha=0.1, 
    n_splits=5,
    steps=100
)

w1_opt, w2_opt = best_weights
print(f"OOF Winkler  = {best_score:.0f}")
print(f"best weight      = w1 {w1_opt:.3f} / w2 {w2_opt:.3f}")


#å¡«è£œç¼ºæ¼�æ¬„ä½�ï¼ˆå°�é½Šè¨“ç·´é›†æ¬„ä½�ï¼‰
missing_cols = set(X.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

# ç¢ºä¿�æ¬„ä½�é †åº�ä¸€è‡´
test_encoded = test_encoded[X.columns]



test_encoded = test_encoded[X.columns]

# ---------- 5â€‘fold ----------
pred_low_raw  = np.zeros(len(test_encoded))
pred_high_raw = np.zeros(len(test_encoded))

for lo, up in zip(fold_lower_models, fold_upper_models):
    pred_low_raw  += np.expm1(lo.predict(test_encoded,  num_iteration=lo.best_iteration_))
    pred_high_raw += np.expm1(up.predict(test_encoded, num_iteration=up.best_iteration_))

pred_low_raw  /= len(fold_lower_models)
pred_high_raw /= len(fold_upper_models)

test_lower = w1_opt * pred_low_raw  + (1 - w1_opt) * pred_high_raw
test_upper = w2_opt * pred_high_raw + (1 - w2_opt) * pred_low_raw
test_lower, test_upper = np.minimum(test_lower, test_upper), np.maximum(test_lower, test_upper)


submission_df = pd.read_csv(sam_path)
submission_df.head()
test_encoded['id'] = test_df['id']

submission_df = pd.DataFrame({
    'id': test_encoded['id'], 
    'pi_lower': test_lower,
    'pi_upper': test_upper
})

submission_df.to_csv("submission.csv", index=False)
print("âœ… Submission file created successfully.")

