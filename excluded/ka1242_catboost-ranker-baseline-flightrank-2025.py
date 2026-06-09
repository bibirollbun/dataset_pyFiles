# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Set display options for better readability
# pd.set_option('display.max_columns', 50)


# Global parameters
TRAIN_SAMPLE_FRAC = 0.20  # Sample 30% of data for faster iteration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Initialize Kaggle API
# api = KaggleApi()
# api.authenticate()


# Load parquet files
train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')


print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Unique ranker_ids in train: {train['ranker_id'].nunique():,}")
print(f"Selected rate: {train['selected'].mean():.3f}")


# Sample by ranker_id to keep groups intact
if TRAIN_SAMPLE_FRAC < 1.0:
    unique_rankers = train['ranker_id'].unique()
    n_sample = int(len(unique_rankers) * TRAIN_SAMPLE_FRAC)
    sampled_rankers = np.random.RandomState(RANDOM_STATE).choice(
        unique_rankers, size=n_sample, replace=False
    )
    train = train[train['ranker_id'].isin(sampled_rankers)]
    print(f"Sampled train to {len(train):,} rows ({train['ranker_id'].nunique():,} groups)")


# Convert ranker_id to string for CatBoost
train['ranker_id'] = train['ranker_id'].astype(str)
test['ranker_id'] = test['ranker_id'].astype(str)


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
                pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
                + pd.to_numeric(parts[1], errors="coerce").fillna(0)
            )
        return out

    # Duration columns
    dur_cols = (
        ["legs0_duration", "legs1_duration"]
        + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    )
    for col in dur_cols:
        if col in df.columns:
            df[col] = hms_to_minutes(df[col])

    # Feature container
    feat = {}

    # Price
    feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
    feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
    feat["log_price"] = np.log1p(df["totalPrice"])

    # Durations
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

    # Frequent-flyer
    ff = df["frequentFlyer"].fillna("").astype(str)
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")
    airlines = ["SU", "S7", "U6", "TK", "DP", "UT", "EK", "N4", "5N", "LH"]
    for al in airlines:
        feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)
    feat["ff_matches_carrier"] = np.select(
        [
            (feat[f"ff_{al}"] == 1)
            & (df["legs0_segments0_marketingCarrier_code"] == al)
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
        df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
        + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
    )
    feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
    feat["total_fees"] = (
        df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
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

    # Merge new features
    df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)

    # Final NaN handling (loop avoids duplicate-column error)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")

    return df


# Apply feature engineering
train = create_features(train)
test = create_features(test)


# Exclude columns
exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
                'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
                'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
                'frequentFlyer']  # Already processed

# Exclude segment 2-3 columns (>98% missing)
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

feature_cols = [col for col in train.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")


# Prepare data
X_train = train[feature_cols]
y_train = train['selected']
groups_train = train['ranker_id']

X_test = test[feature_cols]
groups_test = test['ranker_id']

# Group-based split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]

print(f"Train: {len(X_tr):,} rows, Val: {len(X_val):,} rows, Test: {len(X_test):,} rows")


# # Quick data exploration
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Selection rate by price rank
# train_sample = train.sample(min(10000, len(train)))
# price_rank_selection = train_sample.groupby('price_rank')['selected'].mean()
# axes[0].plot(price_rank_selection.index[:20], price_rank_selection.values[:20], marker='o')
# axes[0].set_xlabel('Price Rank within Group')
# axes[0].set_ylabel('Selection Rate')
# axes[0].set_title('Selection Rate by Price Rank')

# # Direct vs connecting flights
# direct_selection = train.groupby('total_segments')['selected'].mean()
# axes[1].bar(direct_selection.index, direct_selection.values)
# axes[1].set_xlabel('Total Segments')
# axes[1].set_ylabel('Selection Rate')
# axes[1].set_title('Selection Rate by Number of Segments')

# plt.tight_layout()
# plt.show()


%%capture
!pip install -U catboost


from catboost import CatBoostRanker, Pool


# Create CatBoost pools
train_pool = Pool(X_tr, y_tr, group_id=groups_tr, cat_features=cat_features_final)
val_pool = Pool(X_val, y_val, group_id=groups_val, cat_features=cat_features_final)


# Initialize CatBoost Ranker
model = CatBoostRanker(
    loss_function='YetiRank',
    iterations=1000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=0.2,
    random_seed=RANDOM_STATE,
    eval_metric='PrecisionAt:top=3',
    early_stopping_rounds=100,
    verbose=20,
    task_type='CPU',  # Change to 'GPU' if available
    cat_features=cat_features_final,
#     grow_policy='Lossguide'
)


# Train model
model.fit(train_pool, eval_set=val_pool, use_best_model=True);


# Convert scores to probabilities using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x / 10))  # Scale factor for CatBoost scores

# HitRate@3 calculation
def calculate_hitrate_at_k(df, k=3):
    """Calculate HitRate@k for groups with >10 options"""
    hits = []
    for ranker_id, group in df.groupby('ranker_id'):
        # Only consider groups with >10 options
        if len(group) > 10:
            # Get top-k predictions
            top_k = group.nlargest(k, 'pred')
            # Check if selected item is in top-k
            hit = (top_k['selected'] == 1).any()
            hits.append(hit)
    return np.mean(hits) if hits else 0.0


# Evaluate on validation
val_preds = model.predict(X_val)
val_df = pd.DataFrame({
    'ranker_id': groups_val,
    'pred': val_preds,
    'selected': y_val
})

# Get top prediction per group
top_preds = val_df.loc[val_df.groupby('ranker_id')['pred'].idxmax()]
top_preds['prob'] = sigmoid(top_preds['pred'])
val_logloss = log_loss(top_preds['selected'], top_preds['prob'])

hitrate_at_3 = calculate_hitrate_at_k(val_df, k=3)

# Additional metrics
val_accuracy = (top_preds['selected'] == 1).mean()
group_sizes = val_df.groupby('ranker_id').size()
avg_group_size = group_sizes.mean()

print(f"HitRate@3 (groups >10):  {hitrate_at_3:.4f}")
print(f"\nLogLoss:                 {val_logloss:.4f}")
print(f"Top-1 Accuracy:          {val_accuracy:.4f}")
print(f"Groups with >10 options: {(group_sizes > 10).sum()} / {len(group_sizes)} ({(group_sizes > 10).mean():.1%})")


# # Feature importance
# feature_importance = pd.DataFrame({
#     'feature': feature_cols,
#     'importance': model.get_feature_importance(data=val_pool, type='LossFunctionChange')
# }).sort_values('importance', ascending=False)

# plt.figure(figsize=(10, 8))
# top_features = feature_importance.head(20)
# plt.barh(range(len(top_features)), top_features['importance'])
# plt.yticks(range(len(top_features)), top_features['feature'])
# plt.xlabel('Feature Importance')
# plt.title('Top 20 Most Important Features')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()


# Create test pool and predict
test_pool = Pool(X_test, group_id=groups_test, cat_features=cat_features_final)
test_preds = model.predict(test_pool)


# Create submission
submission = test[['Id', 'ranker_id']].copy()
submission['pred_score'] = test_preds

# Assign ranks (1 = best option)
submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
    ascending=False, method='first'
).astype(int)


# Verify ranking integrity
# assert submission.groupby('ranker_id')['selected'].apply(
#     lambda x: sorted(x.tolist()) == list(range(1, len(x)+1))
# ).all(), "Invalid ranking!"


# Save submission
# submission[['Id', 'ranker_id', 'selected']].to_parquet('submission.parquet', index=False)
submission[['Id', 'ranker_id', 'selected']].to_csv('submission.csv', index=False)
print(f"Submission saved. Shape: {submission.shape}")


# # Submit to competition
# api.competition_submit(
#     file_name="submission.parquet", 
#     competition="aeroclub-recsys-2025", 
#     message="CatBoost Ranking Baseline"
# )




