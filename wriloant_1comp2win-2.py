!pip install lightgbm optuna


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 3. Data Loading and Preprocessing
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    return df


train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')



train = reduce_memory_usage(train)
test = reduce_memory_usage(test)


# 4. Feature Engineering (Same as before)
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber'
]


def create_features(df):
    """
    Return a copy of df enriched with engineered features for flight-ranking models.
    """
    

    df = df.copy()

    def hms_to_minutes(s: pd.Series) -> np.ndarray:
        """Vectorised 'HH:MM:SS' → minutes (seconds ignored)."""
        mask = s.notna()
        out = np.zeros(len(s), dtype=float)
        if mask.any():
            parts = s[mask].astype(str).str.split(':', expand=True)
            out[mask] = (
                pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60 +
                pd.to_numeric(parts[1], errors="coerce").fillna(0)
            )
        return out

    # Duration columns
    dur_cols = (
        ["legs0_duration", "legs1_duration"] +
        [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    )
    for col in dur_cols:
        if col in df.columns:
            df[col] = hms_to_minutes(df[col])

    # Feature container
    feat = {}

    # Price-related features
    feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
    feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
    feat["log_price"] = np.log1p(df["totalPrice"])

    # Duration features
    df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
    feat["duration_ratio"] = np.where(
        df["legs1_duration"].fillna(0) > 0,
        df["legs0_duration"] / (df["legs1_duration"] + 1),
        1.0,
    )

    # Segment counts
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{i}_duration" for i in (0, 1)]
        feat[f"n_segments_leg{leg}"] = df[seg_cols].notna().sum(axis=1)
    feat["total_segments"] = feat["n_segments_leg0"] + feat["n_segments_leg1"]

    # Trip type
    feat["is_one_way"] = df["legs1_duration"].isna().astype(int)

    # Rank features
    grp = df.groupby("ranker_id")
    feat["price_rank"] = grp["totalPrice"].rank()
    feat["price_pct_rank"] = grp["totalPrice"].rank(pct=True)
    feat["duration_rank"] = grp["total_duration"].rank()
    feat["is_cheapest"] = (grp["totalPrice"].transform("min") == df["totalPrice"]).astype(int)
    feat["is_most_expensive"] = (grp["totalPrice"].transform("max") == df["totalPrice"]).astype(int)
    feat["price_from_median"] = grp["totalPrice"].transform(
        lambda x: (x - x.median()) / (x.std() + 1)
    )

    # Frequent-flyer features
    ff = df["frequentFlyer"].astype(str).fillna("")
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")
    airlines = ["SU", "S7", "U6", "TK", "DP", "UT", "EK", "N4", "5N", "LH"]
    for al in airlines:
        feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)
    feat["ff_matches_carrier"] = np.select(
        [
            (feat[f"ff_{al}"] == 1) &
            (df["legs0_segments0_marketingCarrier_code"] == al)
            for al in ["SU", "S7", "U6", "TK"]
        ],
        [1, 1, 1, 1],
        default=0,
    )

    # Binary flags
    feat.update(
        dict(
            is_vip_freq=((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int),
            has_return=(~df["legs1_duration"].isna()).astype(int),
            has_corporate_tariff=(~df["corporateTariffCode"].isna()).astype(int),
        )
    )

    # Baggage and fees
    feat["baggage_total"] = (
        df["legs0_segments0_baggageAllowance_quantity"].fillna(0) +
        df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
    )
    feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
    feat["total_fees"] = (
        df["miniRules0_monetaryAmount"].fillna(0) +
        df["miniRules1_monetaryAmount"].fillna(0)
    )
    feat["has_fees"] = (feat["total_fees"] > 0).astype(int)
    feat["fee_rate"] = feat["total_fees"] / (df["totalPrice"] + 1)

    # Time-of-day
    for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            feat[f"{col}_hour"] = dt.dt.hour.fillna(12)
            feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
            h = dt.dt.hour.fillna(12)
            feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)

    # Direct-flight flags
    feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
    feat["is_direct_leg1"] = (feat["n_segments_leg1"] == 1).astype(int)
    feat["both_direct"] = feat["is_direct_leg0"] & feat["is_direct_leg1"]

    # Cheapest direct
    df["_direct"] = feat["n_segments_leg0"] == 1
    direct_min_price = df.loc[df["_direct"]].groupby("ranker_id")["totalPrice"].min()
    feat["is_direct_cheapest"] = (
        df["_direct"] & (df["totalPrice"] == df["ranker_id"].map(direct_min_price))
    ).astype(int)
    df.drop(columns="_direct", inplace=True)

    # Misc flags
    feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)
    feat["group_size"] = df.groupby("ranker_id")["Id"].transform("count")
    feat["group_size_log"] = np.log1p(feat["group_size"])
    feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
    popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"}
    feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)
    feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
    feat["cabin_class_diff"] = (
        df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
    )

    # New Features
    df["price_per_minute"] = df["totalPrice"] / (df["total_duration"] + 1)

    # Time-based features
    for leg in [0, 1]:
        dep_col = f"legs{leg}_departureAt"
        arr_col = f"legs{leg}_arrivalAt"
        if dep_col in df.columns and arr_col in df.columns:
            dep_dt = pd.to_datetime(df[dep_col])
            arr_dt = pd.to_datetime(df[arr_col])
            df[f"leg{leg}_overnight"] = (dep_dt.dt.day != arr_dt.dt.day).astype(int)
            df[f"leg{leg}_departure_daypart"] = dep_dt.dt.hour // 6

    # Carrier dominance features
    carriers = ["SU", "S7", "U6", "TK"]
    for carrier in carriers:
        df[f"is_{carrier}_dominant"] = (
            (df["legs0_segments0_marketingCarrier_code"] == carrier) &
            (df["legs1_segments0_marketingCarrier_code"] == carrier)
        ).astype(int)

    # Advanced baggage features
    df["baggage_quantity_diff"] = (
        df["legs0_segments0_baggageAllowance_quantity"] -
        df["legs1_segments0_baggageAllowance_quantity"]
    )

    # Connection quality metrics
    for leg in [0, 1]:
        if f"legs{leg}_segments1_duration" in df.columns and f"legs{leg}_segments0_duration" in df.columns:
            df[f"leg{leg}_connection_ratio"] = (
                df[f"legs{leg}_segments1_duration"] /
                (df[f"legs{leg}_segments0_duration"] + 1e-6)
            )

    # Advanced group statistics
    grp = df.groupby("ranker_id")
    df["price_group_skewness"] = grp["totalPrice"].transform(lambda x: x.skew())
    df["duration_group_kurtosis"] = grp["total_duration"].transform(lambda x: x.kurt())

    # Merge new features
    df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)

    # Final NaN handling
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")

    return df



train = create_features(train)
test = create_features(test)



# 5. Feature Selection (Same as before)
exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
               'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
               'miniRules0_percentage', 'miniRules1_percentage',
               'frequentFlyer']

for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_qu00antity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')



for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')



feature_cols = [col for col in train.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]



# Convert categorical features for LightGBM
for col in cat_features_final:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# Convert categorical features for LightGBM
for col in cat_features_final:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 6. Train/Validation Split
X_train = train[feature_cols]
y_train = train['selected']
groups_train = train['ranker_id']

X_test = test[feature_cols]
groups_test = test['ranker_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]


# Prepare group counts for LightGBM
train_groups = groups_tr.value_counts().sort_index().values
val_groups = groups_val.value_counts().sort_index().values



# 7. Evaluation Metrics
def calculate_hitrate_at_k(df, k=3):
    hits = []
    for ranker_id, group in df.groupby('ranker_id'):
        if len(group) > 10:
            top_k = group.nlargest(k, 'pred')
            hit = (top_k['selected'] == 1).any()
            hits.append(hit)
    return np.mean(hits) if hits else 0.0



# 8. LightGBM Model Training
def train_lightgbm(params, X_tr, y_tr, X_val, y_val, train_groups, val_groups):
    train_data = lgb.Dataset(
        X_tr, 
        label=y_tr,
        group=train_groups,
        categorical_feature=cat_features_final
    )
    
    val_data = lgb.Dataset(
        X_val, 
        label=y_val,
        group=val_groups,
        categorical_feature=cat_features_final,
        reference=train_data
    )
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    return model





# 9. Hyperparameter Tuning with Optuna
def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3],
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2,log=True),
        'num_leaves': trial.suggest_int('num_leaves', 50, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'bagging_freq': 5,
        'num_threads':4,
        'verbosity': -1,
        'seed': RANDOM_STATE
    }
    
    model = train_lightgbm(params, X_tr, y_tr, X_val, y_val, train_groups, val_groups)
    
    val_preds = model.predict(X_val)
    val_df = pd.DataFrame({'ranker_id': groups_val, 'pred': val_preds, 'selected': y_val})
    return calculate_hitrate_at_k(val_df, k=3)

print("\nStarting hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20,timeout= 1800)


# 10. Train Final Model with Best Parameters
print("\nTraining final model with best parameters...")
best_params = study.best_params
best_params.update({
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [3],
    'verbosity': 20,
    'seed': RANDOM_STATE,
    'max_position': 10 
})

final_model = train_lightgbm(best_params, X_tr, y_tr, X_val, y_val, train_groups, val_groups)


# 11. Evaluation
val_preds = final_model.predict(X_val)
val_df = pd.DataFrame({
    'ranker_id': groups_val,
    'pred': val_preds,
    'selected': y_val
})

def sigmoid(x):
    return 1 / (1 + np.exp(-x / 10))

# Fix: groupby with observed=True
top_preds = val_df.loc[val_df.groupby('ranker_id', observed=True)['pred'].idxmax()]
top_preds['prob'] = sigmoid(top_preds['pred'])
val_logloss = log_loss(top_preds['selected'], top_preds['prob'])

hitrate_at_3 = calculate_hitrate_at_k(val_df, k=3)
val_accuracy = (top_preds['selected'] == 1).mean()
group_sizes = val_df.groupby('ranker_id', observed=True).size()

print("\nFinal Model Performance:")
print(f"HitRate@3: {hitrate_at_3:.4f}")
print(f"LogLoss: {val_logloss:.4f}")
print(f"Top-1 Accuracy: {val_accuracy:.4f}")
print(f"Avg Group Size: {group_sizes.mean():.1f}")



#12 Feature importance visualization
lgb.plot_importance(final_model, max_num_features=20, figsize=(10, 6))
plt.title('Feature Importance (rank_xendcg)')
plt.show()


# 13. Create Submission
test_preds = final_model.predict(X_test)

submission = test[['Id', 'ranker_id']].copy()
submission['pred_score'] = test_preds
submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
    ascending=False, method='first'
).astype(int)

submission[['Id', 'ranker_id', 'selected']].to_csv('submission.csv', index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")

