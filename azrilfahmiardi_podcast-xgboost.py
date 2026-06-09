import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.api.types import CategoricalDtype
from sklearn.impute import SimpleImputer
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
from scipy.stats import skew 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import gc


def clean(df):

    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    df = df.drop('Episode_Title', axis=1)

    return df

def encode(df):
    # The nominative (unordered) categorical features
    features_nom = [
    'Podcast_Name',
    'Genre',
    'Publication_Day',
    ]
    
    # The ordinal (ordered) categorical features
    ordered_levels = {
        'Episode_Sentiment': ['Negative', 'Neutral', 'Positive'],
        'Publication_Time': ['Morning', 'Afternoon', 'Evening', 'Night']
    }

    ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()}

    
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        if "None" not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories("None")
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    
    return df

def impute_upgraded(df):
    for name in df.select_dtypes("number").columns:
        df[name] = df[name].fillna(df[name].median())
    for name in df.select_dtypes("category").columns:
        df[name] = df[name].fillna(df[name].mode().iloc[0])
    return df

def load_data():

    # data_dir = 'input/'
    data_dir = Path("/kaggle/input/playground-series-s5e4/")

    df_train = pd.read_csv(data_dir / 'train.csv', index_col="id")
    df_test = pd.read_csv(data_dir / 'test.csv', index_col="id")
    df_org = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')

    # Gabungkan train, test, dan org untuk preprocessing
    df = pd.concat([df_train, df_test, df_org])

    # Preprocessing
    df = clean(df)
    df = encode(df)
    df = impute_upgraded(df)

    # Reform splits
    df_train = df.loc[df_train.index.union(df_org.index), :]
    df_test = df.loc[df_test.index, :]

    return df_train, df_test





def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X

cluster_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Episode_Num"
]

def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new

def pca_inspired(df):
    X = pd.DataFrame()
    X["TopFeaturesCombined"] = df.Episode_Length_minutes * df.Host_Popularity_percentage * df.Episode_Num
    X["GuestImpact"] = df.Guest_Popularity_percentage * df.Episode_Length_minutes
    X["HostImpact"] = df.Host_Popularity_percentage * df.Episode_Length_minutes
    X["ContentDensity"] = df.Episode_Length_minutes / (df.Number_of_Ads + 1)
    return X

def create_features_categorical(df):
    df['is_weekend'] = df['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    return df


    
def elm(df_train, df_test=None):
    df = df_train.copy()
    ELM = {}
    for k in range(3):
        col_name = f'ELm_r{k}'
        df[col_name] = df['Episode_Length_minutes'].round(k)
        ELM[col_name] = df[col_name]
    return pd.DataFrame(ELM, index=df.index)

# reference : https://www.kaggle.com/code/masayakawamata/single-xgboost-add-selected-features/notebook
selected_comb = [
    # 2-interaction
    ['Episode_Length_minutes', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Number_of_Ads'],    
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Host_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Podcast_Name'],
    ['Episode_Num', 'Podcast_Name'],  
    ['Guest_Popularity_percentage', 'Podcast_Name'],
    ['ELm_r1', 'Episode_Num'],
    ['ELm_r1', 'Host_Popularity_percentage'], 
    ['ELm_r1', 'Guest_Popularity_percentage'],
    ['ELm_r2', 'Episode_Num'],
    ['ELm_r2', 'Episode_Sentiment'],
    ['ELm_r2', 'Publication_Day'],


    # 3-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],

    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],   
    ['ELm_r1', 'Number_of_Ads', 'Episode_Sentiment'],
    ['ELm_r2', 'Number_of_Ads', 'Podcast_Name'],

    # 4-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Genre'],    
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Genre'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Time', 'Podcast_Name'],

    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time', 'Podcast_Name'],
]

def combination_features(df, df_test=None):
    X = df.copy()
    encoded_columns = []

    # Process combinations
    for comb in selected_comb:
        name = '_'.join(comb)
        
        if len(comb) == 2:
            X[name] = X[comb[0]].astype(str) + '_' + X[comb[1]].astype(str)
        elif len(comb) == 3:
            X[name] = (X[comb[0]].astype(str) + '_' +
                      X[comb[1]].astype(str) + '_' +
                      X[comb[2]].astype(str))
        elif len(comb) == 4:
            X[name] = (X[comb[0]].astype(str) + '_' +
                      X[comb[1]].astype(str) + '_' +
                      X[comb[2]].astype(str) + '_' +
                      X[comb[3]].astype(str))
        
        encoded_columns.append(name)
    
    # Convert to categorical
    X[encoded_columns] = X[encoded_columns].astype('category')
    
    # Handle test data if provided
    if df_test is not None:
        X_test = df_test.copy()
        
        # Apply same transformations to test data
        for comb in selected_comb:
            name = '_'.join(comb)
            
            if len(comb) == 2:
                X_test[name] = X_test[comb[0]].astype(str) + '_' + X_test[comb[1]].astype(str)
            elif len(comb) == 3:
                X_test[name] = (X_test[comb[0]].astype(str) + '_' +
                              X_test[comb[1]].astype(str) + '_' +
                              X_test[comb[2]].astype(str))
            elif len(comb) == 4:
                X_test[name] = (X_test[comb[0]].astype(str) + '_' +
                              X_test[comb[1]].astype(str) + '_' +
                              X_test[comb[2]].astype(str) + '_' +
                              X_test[comb[3]].astype(str))
        
        # Convert test data to categorical
        X_test[encoded_columns] = X_test[encoded_columns].astype('category')
        
        return X[encoded_columns], X_test[encoded_columns]
    
    return X[encoded_columns]
    
def get_combination_feature_names(df=None):
    encoded_columns = []
    for comb in selected_comb:
        name = '_'.join(comb)
        encoded_columns.append(name)
    
    return encoded_columns

# reference : https://www.kaggle.com/code/masayakawamata/single-xgboost-add-selected-features
def target_encode(df_train, df_val, col, target, stats='mean', prefix='TE'):
    df_val = df_val.copy()
    agg = df_train.groupby(col)[target].agg(stats)    
    if isinstance(stats, (list, tuple)):
        for s in stats:
            colname = f"{prefix}_{col}_{s}"
            df_val[colname] = df_val[col].map(agg[s]).astype(float)
            df_val[colname].fillna(agg[s].mean(), inplace=True)
    else:
        suffix = stats if isinstance(stats, str) else stats.__name__
        colname = f"{prefix}_{col}_{suffix}"
        df_val[colname] = df_val[col].map(agg).astype(float)
        df_val[colname].fillna(agg.mean(), inplace=True)
    return df_val

# reference: https://www.kaggle.com/code/act18l/say-goodbye-to-ordinalencoder
class OrderedTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cat_cols=None, n_splits=5, smoothing=0):
        self.cat_cols   = cat_cols
        self.n_splits   = n_splits
        self.smoothing  = smoothing       # 0 = no smoothing
        self.maps_      = {}              # per‑fold maps
        self.global_map = {}              # fit on full data for test set

    def _make_fold_map(self, X_col, y):
        means = y.groupby(X_col, dropna=False).mean()
        if self.smoothing > 0:
            counts = y.groupby(X_col, dropna=False).count()
            smooth = (counts * means + self.smoothing * y.mean()) / (counts + self.smoothing)
            means  = smooth
        return {k: r for r, k in enumerate(means.sort_values().index)}

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        kf = KFold(self.n_splits, shuffle=True, random_state=42)
        self.maps_ = {col: [None]*self.n_splits for col in self.cat_cols}

        for fold, (tr_idx, _) in enumerate(kf.split(X)):
            X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
            for col in self.cat_cols:
                self.maps_[col][fold] = self._make_fold_map(X_tr[col], y_tr)

        for col in self.cat_cols:
            self.global_map[col] = self._make_fold_map(X[col], y)

        return self

    def transform(self, X, y=None, fold=None):
        """
        • During CV pass fold index to use fold‑specific maps (leak‑free).
        • At inference time (fold=None) uses global map.
        """
        X = X.copy()
        tgt_maps = {col: (self.global_map[col] if fold is None else self.maps_[col][fold])
                    for col in self.cat_cols}
        for col, mapping in tgt_maps.items():
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("Listening_Time_minutes")

    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("Listening_Time_minutes")
        X = pd.concat([X, X_test])

    #feature engineering
    X = create_features_categorical(X)
    X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    X = X.join(pca_inspired(X))
    X = label_encode(X)
    X = X.join(elm(X))
    X = X.join(combination_features(X))
    
    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    
    if df_test is not None:
        return X, X_test
    else:
        return X


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def score_stacking_Kfold(X, y, X_test, base_models, meta_model):

    X = X.copy()
    X_test = X_test.copy()

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. Buat meta-features untuk training dan validasi
        X_train_meta = np.zeros((len(X_train), len(base_models)))
        X_val_meta = np.zeros((len(X_val), len(base_models)))
        X_test_meta = np.zeros((len(X_test), len(base_models)))

        for i, model in enumerate(base_models):
            model.fit(X_train, y_train)
            X_train_meta[:, i] = model.predict(X_train)
            X_val_meta[:, i] = model.predict(X_val)
            X_test_meta[:, i] += model.predict(X_test) / n_splits  # Rata-rata nanti

        # 2. Train meta-model
        meta_model.fit(X_train_meta, y_train)

        # 3. Predict dan hitung skor di validation
        val_pred = meta_model.predict(X_val_meta)
        score = rmse(y_val, val_pred)
        scores.append(score)

        # 4. Predict untuk test set
        test_preds += meta_model.predict(X_test_meta) / n_splits

        print(f"Fold {fold + 1} RMSE: {score:.4f}")

    print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}')
    print(f'Max RMSE score: {np.max(scores):.5f}')
    print(f'Min RMSE score: {np.min(scores):.5f}')

    return test_preds


df_train, df_test = load_data()
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:,'Listening_Time_minutes']


#Model

xgb_model = XGBRegressor(
    random_state = 0,
    n_estimators = 565,
    max_depth= 14,
    learning_rate = 0.04222221,
    subsample= 0.8,
    colsample_bytree = 0.8,   
    n_jobs= -1 
)

xgb_model_2 = XGBRegressor(
    random_state = 0,
    n_estimators = 565,
    max_depth=12,
    learning_rate=0.05858702616823876,
    subsample=0.9356075676850377,
    colsample_bytree=0.7895819265828284,
    gamma=1.7903575391246762
)

lgbm_model = LGBMRegressor(
    random_state = 0,
    n_iter=1000,
    max_depth=-1,
    num_leaves=1024,
    colsample_bytree=0.7,
    learning_rate=0.03,
    objective='l2',
    verbosity=-1,
    max_bin=1024,
)

random_forest_model = RandomForestRegressor(
    random_state=0,
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1
)


# preparation

train = X_train.copy()
train = train.join(y_train)
test = X_test.copy()
TARGET = 'Listening_Time_minutes'
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads']
encoded_columns = get_combination_feature_names()
encode_stats = ['mean']
FEATURES = NUMS + CATS + encoded_columns


warnings.simplefilter('ignore')


FOLDS          = 10
outer_kf       = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof            = np.zeros(len(train))
pred           = np.zeros(len(test))

for fold, (tr_idx, vl_idx) in enumerate(outer_kf.split(train), 1):
    print(f"--- Fold {fold} / {FOLDS} ---")

    X_tr_raw = train.loc[tr_idx, FEATURES].reset_index(drop=True)
    y_tr     = train.loc[tr_idx, TARGET].reset_index(drop=True)

    X_vl_raw = train.loc[vl_idx, FEATURES].reset_index(drop=True)
    y_vl     = train.loc[vl_idx, TARGET].reset_index(drop=True)

    X_ts_raw = test[FEATURES].copy()

    X_tr, X_vl, X_ts = X_tr_raw.copy(), X_vl_raw.copy(), X_ts_raw.copy()

    inner_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for _, (in_tr_idx, in_vl_idx) in enumerate(inner_kf.split(X_tr_raw), 1):
        in_tr = pd.concat([X_tr_raw.loc[in_tr_idx], y_tr.loc[in_tr_idx]], axis=1)
        in_vl = X_tr_raw.loc[in_vl_idx].reset_index(drop=True)

        for col in encoded_columns:
            for stat in encode_stats:
                te_tmp = target_encode(
                    in_tr, in_vl.copy(),
                    col, TARGET,
                    stats=stat, prefix="TE"
                )
                te_col = f"TE_{col}_{stat}"
                X_tr.loc[in_vl_idx, te_col] = te_tmp[te_col].values

    tr_with_y = pd.concat([X_tr_raw, y_tr], axis=1)
    for col in encoded_columns:
        for stat in encode_stats:
            te_col = f"TE_{col}_{stat}"
            X_vl = target_encode(tr_with_y, X_vl,      col, TARGET,
                                  stats=stat, prefix="TE")
            X_ts = target_encode(tr_with_y, X_ts,      col, TARGET,
                                  stats=stat, prefix="TE")

    X_tr.drop(encoded_columns, axis=1, inplace=True)
    X_vl.drop(encoded_columns, axis=1, inplace=True)
    X_ts.drop(encoded_columns, axis=1, inplace=True)    

    enc = OrderedTargetEncoder(
        cat_cols=CATS,
        n_splits=FOLDS,
        smoothing=20
    ).fit(X_tr, y_tr)

    X_tr[CATS] = enc.transform(X_tr[CATS], fold=None)[CATS]
    X_vl[CATS] = enc.transform(X_vl[CATS], fold=None)[CATS]
    X_ts[CATS] = enc.transform(X_ts[CATS], fold=None)[CATS]
    
    model = XGBRegressor(
        tree_method='hist',
        max_depth=14,
        colsample_bytree=0.5,
        subsample=0.9,
        n_estimators=50_000,
        learning_rate=0.02,
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=150,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=500
    )

    oof[vl_idx]  = model.predict(X_vl)
    pred        += model.predict(X_ts)

    del X_tr_raw, X_vl_raw, X_ts_raw, X_tr, X_vl, X_ts, y_tr, y_vl
    if fold != FOLDS:
        del model
    gc.collect()

pred /= FOLDS
rmse = np.sqrt(mean_squared_error(train[TARGET], oof))
print(f"Final OOF RMSE (XGB): {rmse:.5f}")


def make_submisson():
    output = pd.DataFrame({'id': X_test.index, 'Listening_Time_minutes': pred})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")


make_submisson()

