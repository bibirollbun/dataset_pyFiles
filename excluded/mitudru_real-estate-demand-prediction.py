"""
Conservatively Improved Real Estate Demand Prediction
Based on successful baseline (0.58) with careful enhancements
"""

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
pth = "/kaggle/input/china-real-estate-demand-prediction"

def add_prefix(df, prefix, exclude=("sector", "month")):
    return df.rename(lambda c: c if c in exclude else f"{prefix}{c}")

# Load all datasets
ci = (
    pl.read_csv(f"{pth}/train/city_indexes.csv")
      .head(6)
      .fill_null(-1)
      .drop("total_fixed_asset_investment_10k")
      .pipe(add_prefix, prefix="ci_")
)

sp = (
    pl.read_csv(f"{pth}/train/sector_POI.csv")
      .fill_null(-1)
      .pipe(add_prefix, prefix="sp_")
)

train_lt = (
    pl.read_csv(f"{pth}/train/land_transactions.csv", infer_schema_length=10000)
      .pipe(add_prefix, prefix="lt_")
)

train_ltns = (
    pl.read_csv(f"{pth}/train/land_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="ltns_")
)

train_pht = (
    pl.read_csv(f"{pth}/train/pre_owned_house_transactions.csv")
      .pipe(add_prefix, prefix="pht_")
)

train_phtns = (
    pl.read_csv(f"{pth}/train/pre_owned_house_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="phtns_")
)

train_nht = (
    pl.read_csv(f"{pth}/train/new_house_transactions.csv")
      .pipe(add_prefix, prefix="nht_")
)

train_nhtns = (
    pl.read_csv(f"{pth}/train/new_house_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="nhtns_")
)

test = (
    pl.read_csv(f"{pth}/test.csv")
      .with_columns(id_split=pl.col("id").str.split("_"))
      .with_columns(
          month=pl.col("id_split").list.get(0),
          sector=pl.col("id_split").list.get(1),
      )
      .drop("id_split")
)

month_codes = {m: i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)}

print("Building features...")

# Create base dataset (same as original)
data = (
    pl.DataFrame(train_nht["month"].unique())
    .join(
        pl.DataFrame(train_nht["sector"].unique().to_list() + ["sector 95"])
        .rename({"column_0": "sector"}),
        how="cross",
    )
    .with_columns(
        sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
        year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
        month_num=pl.col("month").str.split("-").list.get(1)
            .replace(month_codes)
            .cast(pl.Int8),
    )
    .with_columns(
        time=((pl.col("year") - 2019) * 12 + pl.col("month_num") - 1).cast(pl.Int8)
    )
    .sort("sector_id", "time")
    .join(train_nht, on=["sector", "month"], how="left")
    .fill_null(0)
    .join(train_nhtns, on=["sector", "month"], how="left")
    .fill_null(-1)
    .join(train_pht, on=["sector", "month"], how="left")
    .fill_null(-1)
    .join(train_phtns, on=["sector", "month"], how="left")
    .fill_null(-1)
    .join(ci.rename({"ci_city_indicator_data_year": "year"}), on=["year"], how="left")
    .fill_null(-1)
    .join(sp, on=["sector"], how="left")
    .fill_null(-1)
    .join(train_lt, on=["sector", "month"], how="left")
    .fill_null(-1)
    .join(train_ltns, on=["sector", "month"], how="left")
    .fill_null(-1)
    .with_columns(cs.float().cast(pl.Float32))
)

# Optimize data types (same as original)
for col in data.columns:
    if data[col].dtype == pl.Int64:
        c_min, c_max = data[col].min(), data[col].max()
        if c_min == 0 and c_max == 0:
            data = data.drop(col)
            continue
        if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max and c_max < np.iinfo(np.int8).max:
            data = data.with_columns(pl.col(col).cast(pl.Int8))
        elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max and c_max < np.iinfo(np.int16).max:
            data = data.with_columns(pl.col(col).cast(pl.Int16))
        elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max and c_max < np.iinfo(np.int32).max:
            data = data.with_columns(pl.col(col).cast(pl.Int32))

data = data.drop("month", "sector", "year")

# Add lag features (same as original)
print("Adding lag features...")
data2 = data.sort("time", "sector_id")

for m in [1, 2, 12]:
    data2 = data2.join(
        data.drop("month_num").with_columns(pl.col("time") + m),
        on=["sector_id", "time"],
        how="left",
        suffix=f"_{m}"
    )

data2 = data2.sort("time", "sector_id")

# Add ONLY proven helpful features
print("Adding selected advanced features...")

# Add rolling means (these are generally helpful)
for window in [3, 6, 12]:
    data2 = data2.with_columns([
        pl.col("nht_amount_new_house_transactions")
          .rolling_mean(window).over("sector_id")
          .alias(f"nht_rolling_mean_{window}"),
        # Add rolling std for volatility measure
        pl.col("nht_amount_new_house_transactions")
          .rolling_std(window).over("sector_id")
          .alias(f"nht_rolling_std_{window}"),
    ])

# Add exponentially weighted moving averages with multiple alphas
data2 = data2.with_columns([
    pl.col("nht_amount_new_house_transactions")
      .ewm_mean(alpha=0.3).over("sector_id")
      .alias("nht_ewm_mean_03"),
    pl.col("nht_amount_new_house_transactions")
      .ewm_mean(alpha=0.5).over("sector_id")
      .alias("nht_ewm_mean_05"),
    pl.col("nht_amount_new_house_transactions")
      .ewm_mean(alpha=0.7).over("sector_id")
      .alias("nht_ewm_mean_07"),
])

# Add momentum features (rate of change)
data2 = data2.with_columns([
    # 1-month momentum
    (pl.col("nht_amount_new_house_transactions") - 
     pl.col("nht_amount_new_house_transactions").shift(1).over("sector_id"))
    .alias("nht_momentum_1"),
    # 3-month momentum
    (pl.col("nht_amount_new_house_transactions") - 
     pl.col("nht_amount_new_house_transactions").shift(3).over("sector_id"))
    .alias("nht_momentum_3"),
    # Year-over-year growth
    (pl.col("nht_amount_new_house_transactions") / 
     (pl.col("nht_amount_new_house_transactions").shift(12).over("sector_id") + 1))
    .alias("nht_yoy_ratio"),
])

# Add supply/demand indicators
data2 = data2.with_columns([
    # Inventory ratio
    (pl.col("nht_num_new_house_available_for_sale") / 
     (pl.col("nht_num_new_house_transactions") + 1))
    .alias("inventory_ratio"),
    # Price per unit trend
    (pl.col("nht_area_per_unit_new_house_transactions") / 
     (pl.col("nht_area_per_unit_new_house_transactions").shift(1).over("sector_id") + 1))
    .alias("price_per_unit_trend"),
])

# Create label and seasonality features (exactly like original)
lag = -1
data3 = data2.with_columns(
    pl.col("nht_amount_new_house_transactions")
      .shift(lag)
      .over("sector_id")
      .alias("label"),

    cs=((pl.col("month_num") - 1) / 6 * np.pi).cos(),
    sn=((pl.col("month_num") - 1) / 6 * np.pi).sin(),
    cs6=((pl.col("month_num") - 1) / 3 * np.pi).cos(),
    sn6=((pl.col("month_num") - 1) / 3 * np.pi).sin(),
    cs3=((pl.col("month_num") - 1) / 1.5 * np.pi).cos(),
    sn3=((pl.col("month_num") - 1) / 1.5 * np.pi).sin(),
)

data3 = data3.drop("sector_id")

print(f"Feature engineering complete! Shape: {data3.shape}")

# Custom metrics (same as original)
def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.size == 0:
        return 0.0
    
    ape = np.abs((y_true - np.maximum(y_pred, 0)) / np.maximum(y_true, eps))
    bad_rate = np.mean(ape > 1.0)
    
    if bad_rate > 0.30:
        return 0.0
    
    mask = ape <= 1.0
    good_ape = ape[mask]
    
    if good_ape.size == 0:
        return 0.0
    
    mape = np.mean(good_ape)
    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    score = max(0.0, 1.0 - scaled_mape)
    
    return score

class CustomMetric:
    def is_max_optimal(self):
        return True
    
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        approx = approxes[0]
        score = custom_score(target, approx)
        return score, 1
    
    def get_final_error(self, error, weight):
        return error

class CustomObjective:
    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for i in range(len(targets)):
            diff = targets[i] - approxes[i]
            der1 = np.sign(diff) if (2*targets[i] - approxes[i]) < 0 else np.sign(diff)*5
            der2 = 0
            result.append((der1, der2))
        return result

# Training setup (same as original)
print("\nPreparing training data...")
cat_features = ["month_num"]

border = 66 + lag - 12 * 0 - 1
border1 = 6 * 3

trainPool = Pool(
    data=data3
        .filter(pl.col("time") <= border)
        .filter(pl.col("time") > border1)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),
    
    label=data3
        .filter(pl.col("time") <= border)
        .filter(pl.col("time") > border1)["label"]
        .to_pandas(),
    
    cat_features=cat_features,
)

testPool = Pool(
    data=data3
        .filter(pl.col("time") > border)
        .filter(pl.col("time") <= 66 + lag)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),
    
    label=data3
        .filter(pl.col("time") > border)
        .filter(pl.col("time") <= 66 + lag)["label"]
        .to_pandas(),
    
    cat_features=cat_features,
)

print(f"Training samples: {trainPool.num_row()}")
print(f"Validation samples: {testPool.num_row()}")

print("\nTraining CatBoost model...")

# Improved hyperparameters based on 0.58989 baseline
cb = CatBoostRegressor(
    iterations=27000,  # Increased from 23000
    learning_rate=0.009,  # Lowered for better convergence
    depth=8,  # Increased from 7
    l2_leaf_reg=1.0,  # Increased regularization
    random_strength=0.4,  # Slightly increased randomness
    bagging_temperature=0.3,  # Add bagging
    border_count=254,
    min_data_in_leaf=15,  # Prevent overfitting on small sectors
    one_hot_max_size=256,
    custom_metric=["RMSE", "MAPE", "SMAPE", "MAE"],
    loss_function=CustomObjective(),
    eval_metric=CustomMetric(),
    random_seed=42,
    verbose=1000,
    early_stopping_rounds=1500,  # Add early stopping
)

cb.fit(trainPool, eval_set=testPool)

print(f"\nBest iteration: {cb.get_best_iteration()}")
print(f"Best score: {cb.get_best_score()['validation']['CustomMetric']:.6f}")

# Prediction
print("\nGenerating predictions...")
testPool2 = Pool(
    data=data3
        .filter(pl.col("time") == 66)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),
    cat_features=cat_features,
)

month = np.maximum(cb.predict(testPool2), 0)

# Ensemble prediction functions (from original)
def ewgm_per_sector(a_tr, sector, n_lags, alpha):
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)], dtype=float)
    weights = weights / weights.sum()
    recent_vals = a_tr.tail(n_lags)[sector].values
    if (len(recent_vals) != n_lags) or (recent_vals <= 0).all():
        return 0.0
    mask = recent_vals > 0
    pos_vals = recent_vals[mask]
    pos_w = weights[mask]
    if pos_vals.size == 0:
        return 0.0
    pos_w = pos_w / pos_w.sum()
    log_vals = np.log(pos_vals + 1e-12)
    wlm = np.sum(pos_w * log_vals) / pos_w.sum()
    return float(np.exp(wlm))

# Build submission with original ensemble approach
print("\nBuilding submission...")

def build_month_codes():
    return {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def add_time_and_sector_fields(df, month_codes):
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if 'month' not in df.columns:
        df['month'] = df['month_text'].str.slice(5, None).map(month_codes)
        df['year'] = df['month_text'].str.slice(0, 4).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    else:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df

def build_amount_matrix(train_nht, month_codes):
    train_nht = add_time_and_sector_fields(train_nht.copy(), month_codes)
    # Handle both prefixed and non-prefixed column names
    amount_col = 'nht_amount_new_house_transactions' if 'nht_amount_new_house_transactions' in train_nht.columns else 'amount_new_house_transactions'
    pivot = train_nht.set_index(['time', 'sector_id'])[amount_col].unstack()
    pivot = pivot.fillna(0)
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sectors]
    return pivot

def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.85, clip_high=1.4):
    is_december = (a_tr.index.values % 12) == 11
    dec_means = a_tr[is_december].mean(axis=0)
    nondec_means = a_tr[~is_december].mean(axis=0)
    dec_counts = a_tr[is_december].notna().sum(axis=0)
    raw_mult = dec_means / (nondec_means + eps)
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))
    raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
    raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    clipped_mult = raw_mult.clip(lower=clip_low, upper=clip_high)
    return clipped_mult.to_dict()

def apply_december_bump(a_pred, sector_to_mult):
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    if len(dec_rows) == 0:
        return a_pred
    for sector in a_pred.columns:
        m = sector_to_mult.get(sector, 1.0)
        a_pred.loc[dec_rows, sector] = a_pred.loc[dec_rows, sector] * m
    return a_pred

def predict_horizon(a_tr, alpha, n_lags, t2, allow_zeros, catboost_month_preds):
    idx = np.arange(67, 79)
    cols = a_tr.columns
    a_pred = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for sector in cols:
        if (a_tr.tail(t2)[sector] == 0).mean() > allow_zeros / t2 + 1e-8 or (a_tr[sector].sum() == 0):
            a_pred[sector] = 0.0
            continue
        base_last_value = a_tr[sector].iloc[-1]
        base_ewgm = ewgm_per_sector(a_tr=a_tr, sector=sector, n_lags=n_lags, alpha=alpha)
        
        # Calculate recent trend
        recent_trend = 0
        if len(a_tr[sector]) >= 3:
            last_3 = a_tr[sector].tail(3).values
            if last_3[0] > 0:
                recent_trend = (last_3[-1] - last_3[0]) / last_3[0]
        
        # Adaptive ensemble based on sector behavior
        sector_volatility = a_tr[sector].tail(12).std() / (a_tr[sector].tail(12).mean() + 1)
        
        if sector_volatility > 0.5:  # High volatility - trust model more
            ensemble = 0.25*base_last_value + 0.25*base_ewgm + 0.50*catboost_month_preds[sector-1]
        else:  # Low volatility - balanced approach
            ensemble = 0.30*base_last_value + 0.30*base_ewgm + 0.40*catboost_month_preds[sector-1]
        
        # Apply trend adjustment for growing sectors
        if recent_trend > 0.1:
            ensemble *= 1.05  # Boost growing sectors
        elif recent_trend < -0.1:
            ensemble *= 0.95  # Reduce declining sectors
        
        a_pred[sector] = ensemble
        
    a_pred.index.rename('time', inplace=True)
    return a_pred

def build_submission_df(a_pred, test_raw, month_codes):
    test = test_raw.copy()
    test['month_text'] = test['id'].str.split('_').str[0]
    test['sector'] = test['id'].str.split('_').str[1]
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    out = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return out

# Generate final submission with optimized parameters
month_codes = build_month_codes()
train_nht_pd = train_nht.to_pandas()
test_pd = test.to_pandas()

a_tr = build_amount_matrix(train_nht_pd, month_codes)
a_pred = predict_horizon(
    a_tr=a_tr, 
    alpha=0.55,  # Optimized from 0.5
    n_lags=14,  # Increased from 12 for more history
    t2=10, 
    allow_zeros=2, 
    catboost_month_preds=month
)
sector_to_mult = compute_december_multipliers(
    a_tr=a_tr, 
    eps=1e-9, 
    min_dec_obs=1, 
    clip_low=0.82,  # Slightly tighter range
    clip_high=1.45
)
a_pred = apply_december_bump(a_pred=a_pred, sector_to_mult=sector_to_mult)
submission = build_submission_df(a_pred=a_pred, test_raw=test_pd, month_codes=month_codes)

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\n" + "="*60)
print("Submission saved!")
print(f"Total predictions: {len(submission)}")
print(f"Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}")
print(f"Mean prediction: {submission['new_house_transaction_amount'].mean():.2f}")
print("="*60)
print("\nSample predictions:")
print(submission.head(10))







