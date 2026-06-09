import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# other similar competitations: 
# https://www.kaggle.com/code/sheemamasood/jane-street-real-time-market-data-forecasting
# https://www.kaggle.com/competitions/millennium-statistical-modeling-quant-challenge


SEED = 42
N_SPLITS = 5

def safe_div(a, b):
    return a / (b + 1e-9)


def load_data():
    # train
    train = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")

    # test
    test_df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv")
    revealed_targets = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/example_test_files/revealed_targets.csv")

    return train, test_df, revealed_targets


#data inspection
train, test_df, revealed_targets = load_data()
train.dropna(subset=['target'], inplace=True)
#train.head()


train.head()


# df = train.groupby(['stock_id', 'date_id']).last().reset_index()
# df.head()


def data_inspection(train: pd.DataFrame, test_df: pd.DataFrame):
    print(f"train shape: {train.shape}")
    print(f"test shape: {test_df.shape}")
    print('\nnumber of stocks: ')
    print(train['stock_id'].nunique())
    print('\nnumber of days: ')
    print(train['date_id'].nunique())
    
    print("\ndataset columns:")
    print(train.columns.tolist())
    
    print("\ncheck nulls:")
    missing_train = train.isnull().sum().sort_values(ascending=False)
    print(missing_train)

    # quick stats for key numeric cols
    key_cols = ["imbalance_size", "matched_size", "bid_price", "ask_price", "wap", "target", 'date_id',"seconds_in_bucket", 'bid_size','ask_size']
    print("\ntrain numeric describe:")
    print(train[key_cols].describe(include="all").T)

data_inspection(train, test_df)


import matplotlib.pyplot as plt

def plot_stock_on_day(df, stock_id, date_id):
    """
    Plot price evolution for a single stock on a single day.
    """
    sub = df[(df["stock_id"] == stock_id) & (df["date_id"] == date_id)].copy()
    sub = sub.sort_values("seconds_in_bucket", ascending=True)

    # ---------- PRICE PLOT ----------
    plt.figure(figsize=(12, 6))

    #plt.plot(sub["seconds_in_bucket"], sub["bid_price"], label="Bid", alpha=0.7)
    #plt.plot(sub["seconds_in_bucket"], sub["ask_price"], label="Ask", alpha=0.7)
    #plt.plot(sub["seconds_in_bucket"], sub["wap"], label="WAP", linestyle="--")
    plt.plot(sub["seconds_in_bucket"], sub["reference_price"], label="Reference", linestyle=":")
    plt.plot(sub["seconds_in_bucket"], sub["near_price"], label="Near", linestyle="-.")
    plt.plot(sub["seconds_in_bucket"], sub["far_price"], label="Far", linestyle="-.")

    plt.title(f"Stock {stock_id} | Date {date_id} — Price evolution")
    plt.xlabel("Seconds in bucket")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_stock_on_day(train, 0, 0)


def plot_price_relationships(df, stock_id, date_id):
    sub = df[(df.stock_id == stock_id) & (df.date_id == date_id)].copy()
    sub["mid_price"] = 0.5 * (sub.bid_price + sub.ask_price)

    plt.figure(figsize=(5, 5))
    plt.scatter(sub["wap"], sub["reference_price"], alpha=0.4)
    plt.xlabel("WAP")
    plt.ylabel("Reference price")
    plt.title("WAP vs Reference")
    plt.grid(True)
    plt.show()

plot_price_relationships(train, 0, 0)


import matplotlib.pyplot as plt

def plot_target(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df['target'], bins=50, color='skyblue')
    plt.title('Target Variable Distribution')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

plot_target(train)


import seaborn as sns

def plot_corr(df):
    plt.figure(figsize=(16, 12))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title("Feature Correlation with Target")
    plt.show()

plot_corr(train)


sns.countplot(x='imbalance_buy_sell_flag', data=train)


def plot_target_by_time_bucket(df_train):
    df = df_train.copy()
    df['bucket'] = pd.cut(df['seconds_in_bucket'], bins=[0, 200, 400, 600])
    df.groupby('bucket')['target'].mean().plot(kind='bar')
    plt.title('Average Target Value by Time Bucket')
    plt.ylabel('Mean Target')
    plt.show()

plot_target_by_time_bucket(train)


def plot_hist_kde(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['target'], bins=100, kde=True, color='darkblue')
    plt.title('Distribution of Target (Price Movement)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

plot_hist_kde(train)


def check_box_plot_by_seconds(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='seconds_in_bucket', y='target', data=df, palette='Blues')
    plt.title('Target Distribution Across Auction Timeline')
    plt.xlabel('Seconds in Bucket')
    plt.ylabel('Target')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

check_box_plot_by_seconds(train)


class FeatureEngineer:
    
    def __init__(self, lags=(1, 2, 3, 5, 10), roll_window=(3, 5, 10), add_market_agg=True):
        self.lags = lags
        self.roll_window = roll_window
        self.add_market_agg = add_market_agg

    def _add_basic_features(self, df_X: pd.DataFrame) -> pd.DataFrame:
        # book prices
        df = df_X.copy()
        df['mid_price'] = 0.5 * (df['bid_price'] + df['ask_price'])
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_rel'] = safe_div(df['spread'], df['mid_price'])
        
        df['size_imbalance'] = safe_div(df['bid_size'] - df['ask_size'], df['bid_size'] + df['ask_size'])
        df['book_size_imbalance'] = safe_div(df['bid_size'], df['ask_size'])

        df['total_auction_sizes'] = df['imbalance_size'] + df['matched_size']
        df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
        df['imbalance_ratio'] = safe_div(df['imbalance_size'], df['total_auction_sizes'])
        df['matches_ratio'] = safe_div(df['matched_size'], df['total_auction_sizes'])

        # add missing prices flag
        # 'imbalance_size', 'matched_size': missingness is not informative in this dataset
        for c in [ 'reference_price', 'far_price', 'near_price', 'bid_price', 'ask_price', 'wap']:
            # Missing values occur because a specific market condition holds or the certain auction stage hasn’t been reached
            df[f'{c}_isna'] = df[c].isna().astype('int8')

        #fill na
        df['reference_price_filled'] = df['reference_price'].fillna(df['wap'])
        df['wap_filled'] = df['wap'].fillna(df['reference_price'])
        df['far_price_filled'] = df['far_price'].fillna(df['reference_price_filled'])
        df['near_price_filled'] = df['near_price'].fillna(df['reference_price_filled'])

        # get price diffs
        df['ref_wap_diff'] = df['reference_price_filled'] - df['wap_filled']
        df['mid_wap_diff'] = df['mid_price'] - df['wap_filled']
        df['near_far_diff'] = df['near_price_filled'] - df['far_price_filled']
        df['near_ref_diff'] = df['near_price_filled'] - df['reference_price_filled']
        df['far_ref_diff'] = df['far_price_filled'] - df['reference_price_filled']

        # time based features
        s = df['seconds_in_bucket'].astype('float32')
        # smooth the time onto a circle, so it is continuous and cyclincal
        # good for tree-based model; sin, cos - Where are we on the auction clock?
        df['sec_sin'] = np.sin(2 * np.pi * s / 600.0)
        df["sec_cos"] = np.cos(2 * np.pi * s / 600.0)

        return df


    def _add_market_agg_features(self, df_X: pd.DataFrame) -> pd.DataFrame:
        df = df_X.copy()
        key = ['date_id', 'seconds_in_bucket']
        cols = ['wap_filled', 'spread', 'size_imbalance', 'signed_imbalance', 'imbalance_ratio', 'matches_ratio']

        for c in cols:
            df[f'market_{c}_median'] = df.groupby(key)[c].transform("median")
            df[f'market_{c}_mean'] = df.groupby(key)[c].transform("mean")

        df['wap_vs_mkt'] = df['wap_filled'] - df["market_wap_filled_median"]
        df['spread_vs_mkt'] = df['spread'] - df["market_spread_median"]
        return df
    
    def _add_lag_and_rolling_features(self, df_X: pd.DataFrame) -> pd.DataFrame:
        df = df_X.copy()
        # why mergesort?
        # Python default to "quicksort": fast, but NOT stable, If two rows compare equal on the sort keys
        # for this data, the original order matters, to calculate the lag/rolling features
        df = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"], kind="mergesort")
        key = ['date_id', 'seconds_in_bucket']
        grouped = df.groupby(key)

        # lag prices
        prices = ['wap_filled', 'reference_price_filled', 'mid_price', 'spread', 'far_price', 'near_price', 'signed_imbalance']
        for lag in self.lags:
            for p in prices:
                df[f'{p}_lag{lag}'] = grouped[p].shift(lag)

        # returns
        for lag in self.lags:
            df[f'wap_ret_window{lag}'] = safe_div(df['wap_filled'] - df[f'wap_filled_lag{lag}'], df[f'wap_filled_lag{lag}'])
            df[f'spread_chg_lag{lag}'] = df['spread'] - df[f'spread_lag{lag}']
            df[f"imb_chg_lag{lag}"] = df["signed_imbalance"] - df[f"signed_imbalance_lag{lag}"]

        roll_cols = ['wap_filled', 'reference_price_filled', 'mid_price', 'spread', 'size_imbalance', 'near_price', 'signed_imbalance']
        for w in self.roll_window:
            for c in roll_cols:
                # why shift 1 - a feature at time t must not use information from time t itself 
                # — otherwise you leak the present into the past
                df[f'{c}_roll{w}_mean'] = grouped[c].transform(lambda s: s.shift(1).rolling(w).mean())

        return df
            

    def _add_histroical_agg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sec_daily = (
        df.groupby(["stock_id", "date_id", "seconds_in_bucket"], sort=False)
          .agg(
              spread_mean=("spread", "mean"),
              matched_mean=("matched_size", "mean"),
              abs_imb_mean=("imbalance_size", "mean"),
              wap_mean=("wap_filled", "mean"),
          )
          .reset_index()
          .sort_values(["stock_id", "seconds_in_bucket", "date_id"])
        )
        grp = sec_daily.groupby(["stock_id", "seconds_in_bucket"], sort=False)
        # expanding mean/std up to previous day
        for col in ["spread_mean", "matched_mean", "abs_imb_mean", "wap_mean"]:
            sec_daily[f"{col}_hist_mean"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
            sec_daily[f"{col}_hist_std"]  = grp[col].transform(lambda s: s.shift(1).expanding().std())

        # merge these baselines back
        df = df.merge(
            sec_daily[[
                "stock_id", "date_id", "seconds_in_bucket",
                "spread_mean_hist_mean", "spread_mean_hist_std",
                "matched_mean_hist_mean", "matched_mean_hist_std",
                "abs_imb_mean_hist_mean", "abs_imb_mean_hist_std",
                "wap_mean_hist_mean", "wap_mean_hist_std",
            ]],
            on=["stock_id", "date_id", "seconds_in_bucket"],
            how="left"
        )

        # --- Surprise / z-score style features
        df["spread_surprise"] = df["spread"] - df["spread_mean_hist_mean"]
        df["matched_size_surprise"] = df["matched_size"] - df["matched_mean_hist_mean"]
        df["abs_imb_surprise"] = df["imbalance_size"] - df["abs_imb_mean_hist_mean"]
        df["wap_surprise"] = df["wap_filled"] - df["wap_mean_hist_mean"]

        return df

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        # Without .copy(), Pandas is sometimes working with a view (shared memory, changes may affect the original)
        # and sometimes with a copy (separate memory), 
        # and you don’t control which one you get
        # With copy(), No ambiguity, and cost is tiny compared to groupby/rolling/modeling
        df = df_X.copy()
        df = self._add_basic_features(df)
        df = self._add_market_agg_features(df)
        df = self._add_lag_and_rolling_features(df)
        df = self._add_histroical_agg_features(df)
        return df


#fe = FeatureEngineer()
#df_fe = fe.transform(df_X)
#df_fe.head()


import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import early_stopping

def train_model(X, y):
    models = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # LightGBM model
        model_lgb = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.05)
        model_lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                      callbacks=[early_stopping(stopping_rounds=50)])

        # XGBoost model
        model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05)
        model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

        # CatBoost model
        model_cbt = cbt.CatBoostRegressor(loss_function='MAE', iterations=1000, learning_rate=0.05, verbose=0)
        model_cbt.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

        models.append((model_lgb, model_xgb, model_cbt))

    return models


# ----------------------------
# Training
# ----------------------------

import lightgbm as lgb

def train_models_cv(train_df: pd.DataFrame, feature_cols, target_col="target", group_col="date_id",
                  n_splits=N_SPLITS, seed=SEED):
    train_df = train_df.sort_values(["date_id", "seconds_in_bucket", 'stock_id'])
    X = train_df[feature_cols]
    y = train_df[target_col].astype("float32")
    return train_model(X, y)


# TODO: do we really need this?
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if col == "row_id":
            continue
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"[mem] {start_mem:.1f} MB -> {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduced)")
    return df


train, test_df, revealed_targets = load_data()
data_inspection(train, test_df)

# Optional: reduce memory early
train = reduce_mem_usage(train)
test_df = reduce_mem_usage(test_df)

# Feature engineering
fe = FeatureEngineer()


train.dropna(subset=['target'], inplace=True)


# Transform train+test together so features are consistent
all_df = pd.concat(
    [train.drop(columns=["target"]), test_df],
    axis=0,
    ignore_index=True
)

all_df = fe.transform(all_df)
all_df = reduce_mem_usage(all_df, verbose=True)


# Split back
train_fe = all_df.iloc[:len(train)].copy()
train_fe["target"] = train["target"].values
test_fe = all_df.iloc[len(train):].copy()


# Features to use
drop_cols = {"target", "row_id", "currently_scored", "date_id", "time_id"}
feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print(f"[features] using {len(feature_cols)} features")


def get_predictions(X_test, models):
    predictions = np.zeros((X_test.shape[0], len(models)))
    for i, (model_lgb, model_xgb, model_cbt) in enumerate(models):
        # Generate predictions and store them
        predictions[:, i] = (model_lgb.predict(X_test) +
                             model_xgb.predict(X_test) +
                             model_cbt.predict(X_test)) / 3
    
    # Return the average prediction across all models
    return predictions.mean(axis=1)


# Train CV
models = train_models_cv(
    train_df=train_fe,
    feature_cols=feature_cols,
    target_col="target",
    group_col="date_id",
    n_splits=N_SPLITS,
    seed=SEED,
)

# Predict test (fold-average)
X_test = test_fe[feature_cols]
test_pred = get_predictions(X_test, models)


# Feature importance (quick peek)
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": np.mean([model_xgb.feature_importances_ for (model_lgb, model_xgb, model_cbt) in models], axis=0)
}).sort_values("importance", ascending=False)

print("\nTop 30 features:")
print(fi.head(30).to_string(index=False))


#if __name__ == "__main__":
#    main()


df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df.head()


print(df['date_id'].nunique())
print("\nDate Describe")
print(df['date_id'].describe())



df = df[['date_id', 'seconds_in_bucket','stock_id', 'target']]
df.isna().sum()


df = df.dropna(subset=['target'])
df_pivot = df.pivot(index=['date_id', 'seconds_in_bucket'], columns = 'stock_id', values = 'target').reset_index()
df_pivot.head()


df_pivot.columns = ['id_' + c for c in df_pivot.columns.astype(str)]
df_pivot.head()


corr = df_pivot.iloc[2:, 2:].corr()
corr


#plt.figure(figsize=(300, 300))
#sns.heatmap(corr, cmap='coolwarm')
#plt.show()


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(corr)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(corr)


kmeans.predict(corr)


from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

distances = 1 - corr.abs().values 

distArray = ssd.squareform(distances) 
hier = hierarchy.linkage(distArray, method="ward")

dend = hierarchy.dendrogram(hier, truncate_mode="level", p=2, color_threshold=1.5)


cluster_labels = hierarchy.fcluster(hier, criterion = 'distance', t=1.3)
cluster_labels

