import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats.mstats import winsorize


train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
sample_submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")


SEED = 42
N_SPLITS = 5
TEST_SIZE = 20000
DEBUG = False


# x = test_df.index
# y = test_df['volume']

# plt.plot(x, y)
# plt.show()


# x_cols = [col for col in train_df.columns if col.startswith("X")]
# corrs = train_df[x_cols + ['label']].corr()['label'].sort_values(ascending=False)
# print(corrs.head(20))


x_cols = [col for col in train_df.columns if col.startswith("X")]
X_raw = train_df[x_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
x_var_series = X_raw.var().sort_values(ascending=False)
top_x_cols = x_var_series[x_var_series > 0].head(50).index.tolist()

del X_raw
gc.collect()


def add_new_features(df):
    df = df.copy()
    
    df["bid_ask_diff"] = df["bid_qty"] - df["ask_qty"]
    df["buy_sell_ratio"] = df["buy_qty"] / (df["sell_qty"] + 1e-6)
    df["bid_ask_ratio"] = df["bid_qty"] / (df["ask_qty"] + 1e-6)
    df["buy_volume_ratio"] = df["buy_qty"] / (df["volume"] + 1e-6)
    df["sell_volume_ratio"] = df["sell_qty"] / (df["volume"] + 1e-6)

    df["sell_buy_ratio"] = df["sell_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-9)
    df["buy_sell_diff"] = df["buy_qty"] - df["sell_qty"]
    df["buy_sell_sum"] = df["buy_qty"] + df["sell_qty"]
    df["ask_bid_ratio"] = df["ask_qty"] / (df["bid_qty"] + df["ask_qty"] + 1e-9)
    df["bid_ask_sum"] = df["bid_qty"] + df["ask_qty"]

    df["order_pressure"] = (df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + 1e-6)
    df["quoted_pressure"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)
    df["execution_ratio"] = (df["buy_qty"] + df["sell_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)
    df["volume_imbalance"] = (df["buy_qty"] - df["sell_qty"]) / (df["volume"] + 1e-6)
    df["order_book_total"] = df["bid_qty"] + df["ask_qty"]
    df["execution_total"] = df["buy_qty"] + df["sell_qty"]
    df["execution_share"] = df["execution_total"] / (df["order_book_total"] + 1e-6)
    
    # X_raw = df[top_x_cols]
    # df["X_mean"] = X_raw.mean(axis=1)
    # df["X_std"] = X_raw.std(axis=1)
    # df["X_min"] = X_raw.min(axis=1)
    # df["X_max"] = X_raw.max(axis=1)
    # df["X_q25"] = X_raw.quantile(0.25, axis=1)
    # df["X_q75"] = X_raw.quantile(0.75, axis=1)
    # df["X_skew"] = X_raw.skew(axis=1)
    # df["X_kurt"] = X_raw.kurtosis(axis=1)
    
    return df

train_df = add_new_features(train_df)
test_df = add_new_features(test_df)

# train_df = add_new_features(train_df, top_x_cols)
# test_df = add_new_features(test_df, top_x_cols)


# start_date = '2024-01-01'
# end_date = '2024-01-02'
# period_df = train_df.loc[start_date:end_date, ['label', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']]

# fig, ax1 = plt.subplots(figsize=(15, 7))

# for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
#     ax1.plot(period_df.index, period_df[col], label=col)
# ax1.set_ylabel('Quantity / Volume')
# ax1.grid(True)

# ax2 = ax1.twinx()
# ax2.plot(period_df.index, period_df['label'], color='black', linestyle='--', label='label')
# ax2.set_ylabel('Label')

# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# plt.title(f'Time Series Plot from {start_date} to {end_date}')
# plt.tight_layout()
# plt.show()


features = [col for col in train_df.drop(columns=["label"]).columns if 'X' not in col]


# Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

km_features = features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[km_features])

kmeans = KMeans(n_clusters=5, random_state=SEED)
train_df['cluster'] = kmeans.fit_predict(X_train_scaled)

X_test_scaled = scaler.transform(test_df[km_features])
test_df['cluster'] = kmeans.predict(X_test_scaled)


# Select features
features = [
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
    "bid_ask_diff", "buy_sell_ratio", "bid_ask_ratio",
    "buy_volume_ratio", "sell_volume_ratio",
    "sell_buy_ratio", "buy_sell_diff", "buy_sell_sum",
    "ask_bid_ratio", "bid_ask_sum", "order_pressure",
    "quoted_pressure", "execution_ratio", "volume_imbalance",
    "order_book_total", "execution_total", "execution_share",
    "cluster"
] + top_x_cols
# features = [col for col in train_df.drop(columns=["label"]).columns if 'X' not in col]
# features = train_df.drop(columns=["label"]).columns + top_x_cols
target = 'label'


X = train_df[features]

# clipping
y = train_df[target].clip(-3, 3)

# using winsorize
# lower_limit = 0.01
# upper_limit = 0.01
# y_raw = train_df[target].values
# y_winsor = winsorize(y_raw, limits=(lower_limit, upper_limit))
# y = pd.Series(y_winsor.data, index=train_df.index)

# transform the target variable using log1p 
# y_raw = train_df[target].values
# shift = abs(np.min(y_raw)) + 1e-3
# y_shifted = y_raw + shift

# y_log = np.log1p(y_shifted)
# y = pd.Series(y_log, index=train_df.index)


params = {
    "objective": "huber", 
    "metric": "rmse",
    "learning_rate": 0.01,
    "n_estimators": 3000,
    "num_leaves": 64, 
    "max_depth": -1, 
    "min_child_samples": 100,
    "min_split_gain": 0.02,
    "subsample": 0.8,        
    "colsample_bytree": 0.8,
    "reg_alpha": 3.0,        
    "reg_lambda": 3.0,       
    "feature_fraction": 0.8,
    "random_state": SEED,
    "verbosity": -1,
    "force_col_wise": True
}


# # TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

kf = KFold(n_splits=N_SPLITS, shuffle=False)

models = []
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n========= Fold {fold + 1} =========")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)
        ]
    )

    y_val_pred = model.predict(X_val)

    # Z-score normalization for Pearson correlation
    if y_val_pred.std() > 0:
        y_val_pred = (y_val_pred - y_val_pred.mean()) / y_val_pred.std()
        
    val_pearson = pearsonr(y_val, y_val_pred)[0]
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"✅ Fold {fold+1} Pearson: {val_pearson:.4f} / RMSE: {val_rmse:.4f}")

    val_scores.append(val_pearson)
    models.append(model)


print(f"\n✅ Average CV Pearson: {np.mean(val_scores):.4f}")


submit_score = []

for fold_, model in enumerate(models):
    pred_ = model.predict(test_df[features])
    submit_score.append(pred_)
    # pred_log = model.predict(test_df[features])
    # pred = np.expm1(pred_log) - shift
    # submit_score.append(pred)

# predict test data
pred = np.mean(submit_score, axis=0)

# Z-score normalization for Pearson correlation
if pred.std() > 0:
    pred = (pred - pred.mean()) / pred.std()


submission = pd.DataFrame({
    'ID': sample_submission.ID,
    'prediction': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission




