# %%capture
!pip install --upgrade pip
!pip install polars


import kagglehub
import os

path = kagglehub.competition_download("aeroclub-recsys-2025")
print("✅ Dataset downloaded to:", path)


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


import polars as pl
import numpy as np
import time

print("=" * 60)
print("Starting Simplified Feature Engineering Pipeline for DL")
print("=" * 60)

# Load data
print("\n[1/8] Loading data...")
start_time = time.time()
train = pl.read_parquet(f'{path}/train.parquet').drop('__index_level_0__')
test = pl.read_parquet(f'{path}/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

test_start_idx = len(train)

data_raw = pl.concat((train, test))

print(f"   ✓ Loaded {len(data_raw)} rows")
print(f"   ✓ Total dataset: {len(data_raw)} rows, {len(data_raw.columns)} columns")
print(f"   Time: {time.time() - start_time:.2f}s")

df = data_raw.clone()


def hitrate_at_3(y_true, y_pred, groups):
    df = pl.DataFrame({
        'group': groups,
        'pred': y_pred,
        'true': y_true
    })

    return (
        df.filter(pl.col("group").count().over("group") > 10)
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(3)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )


def dur_to_hour(col: pl.Expr) -> pl.Expr:
    # Always treat input as string during processing
    col_str = col.cast(pl.Utf8)

    # Extract days part (e.g., "1." from "1.10:30:00")
    days = (
        col_str
        .str.extract(r"^(\d+)\.", 1)
        .cast(pl.Float64)
        .fill_null(0) * 24
    )

    # Remove "X." prefix
    time_str = col_str.str.replace(r"^\d+\.", "")

    # Extract hours/minutes
    hours = (
        time_str.str.extract(r"(\d+):", 1)
        .cast(pl.Float64)
        .fill_null(0)
    )

    minutes = (
        time_str.str.extract(r":(\d+):", 1)
        .cast(pl.Float64)
        .fill_null(0) / 60
    )

    return (days + hours + minutes).fill_null(0)
# Process duration columns
print("\n[2/8] Processing duration columns...")
start_time = time.time()
dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1, 2, 3)]
dur_exprs = [dur_to_hour(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

if dur_exprs:
    df = df.with_columns(dur_exprs)
    print(f"   ✓ Converted {len(dur_exprs)} duration columns to minutes")
print(f"   Time: {time.time() - start_time:.2f}s")

# === CORE NUMERICAL FEATURES ===
print("\n[3/8] Creating core numerical features...")
start_time = time.time()

# Get all segment columns for aggregation
all_seg_cols = [c for c in df.columns if '_segments' in c]
cabin_cols = [c for c in all_seg_cols if c.endswith('_cabinClass')]
baggage_cols = [c for c in all_seg_cols if 'baggageAllowance_quantity' in c]
duration_seg_cols = [c for c in all_seg_cols if c.endswith('_duration')]



df = df.with_columns([
    # Price features
    (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
    (pl.col("taxes") * 100 / (pl.col("totalPrice") + 1)).alias("tax_ratex100"),
    pl.col("totalPrice").log1p().alias("log_price"),

    # Duration features
    (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
    pl.when(pl.col("legs1_duration").fill_null(0) > 0)
        .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 0.01))
        .otherwise(1.0).alias("duration_ratio"),

    # Baggage - aggregate ALL segments
    (pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in baggage_cols]) if baggage_cols else pl.lit(0)).alias("baggage_mean"),

    # Fees
    (pl.col("miniRules0_monetaryAmount").fill_null(0) +
     pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),

    # Cabin class - average across ALL segments (not just first)
    (pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in cabin_cols]) if cabin_cols else pl.lit(0)).alias("avg_cabin_class_all"),

    # Cabin class difference between legs (if round trip)
    pl.when(pl.col("legs1_duration").is_not_null())
        .then(
            pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in cabin_cols if 'legs0_' in c]) -
            pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in cabin_cols if 'legs1_' in c])
        )
        .otherwise(0.0).alias("cabin_class_diff_legs"),
])

# We use .over("ranker_id") to calculate stats per search query
df = df.with_columns([
    # 1. Calculate the minimum and mean duration for this specific search (ranker_id)
    pl.col("total_duration").min().over("ranker_id").alias("min_dur_group"),
    pl.col("total_duration").mean().over("ranker_id").alias("mean_dur_group"),
    pl.col("total_duration").std().over("ranker_id").fill_null(1).alias("std_dur_group"),
])

df = df.with_columns([
    # === APPROACH C: Z-Score (Standardized) ===
    # Positive values = Faster than average (Good)
    # Negative values = Slower than average (Bad)
    ((pl.col("mean_dur_group") - pl.col("total_duration")) / (pl.col("std_dur_group") + 1e-6)).alias("duration_z_score")
])

# Drop the temporary aggregate columns
df = df.drop(["min_dur_group", "mean_dur_group", "std_dur_group"])

# Segment counting
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

df = df.with_columns([
    (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists])
     if mc_exists else pl.lit(0)).alias("total_segments"),
])

# Derived fee features
df = df.with_columns([
    (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
])

print(f"   ✓ Created aggregated price, duration, baggage, and cabin features")
print(f"   Time: {time.time() - start_time:.2f}s")

# === BINARY FEATURES ===
print("\n[4/8] Creating binary features...")
start_time = time.time()

df = df.with_columns([
    # Trip type
    (pl.col("legs1_duration").is_null() |
     (pl.col("legs1_duration") == 0) |
     pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),

    # Direct flights
    pl.when(pl.col("legs1_duration").is_not_null())
        .then((pl.sum_horizontal([pl.col(c).is_not_null() for c in mc_exists if 'legs1_' in c]) == 1).cast(pl.Int32))
        .otherwise(0).alias("is_direct_leg1"),

    # Corporate & VIP
    pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
    (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
    ((pl.col("isVip") == 1) | (pl.col("frequentFlyer").fill_null("") != "")).cast(pl.Int32).alias("is_vip_freq"),

    # Baggage & fees
    (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),


    # Major carriers
    (pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"]) if "legs0_segments0_marketingCarrier_code" in df.columns
     else pl.lit(False)).cast(pl.Int32).alias("is_major_carrier"),
])


print(f"   ✓ Created binary trip type, VIP, baggage, and carrier features")
print(f"   Time: {time.time() - start_time:.2f}s")

# === DATETIME FEATURES WITH CYCLIC ENCODING ===
print("\n[5/8] Processing datetime features with cyclic encoding...")
start_time = time.time()

# Cyclic encoding for hour and weekday
time_exprs = []
for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
    if col in df.columns:
        dt = pl.col(col).str.to_datetime(strict=False)
        h = dt.dt.hour().fill_null(12)
        wd = dt.dt.weekday().fill_null(0)

        # Sin/cos for hour (24-hour cycle)
        time_exprs.extend([
            (np.sin(2 * np.pi * h / 24)).alias(f"{col}_hour_sin"),
            (np.cos(2 * np.pi * h / 24)).alias(f"{col}_hour_cos"),
            # Sin/cos for weekday (7-day cycle)
            (np.sin(2 * np.pi * wd / 7)).alias(f"{col}_weekday_sin"),
            (np.cos(2 * np.pi * wd / 7)).alias(f"{col}_weekday_cos"),
            # Business time flag
            (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
        ])

if time_exprs:
    df = df.with_columns(time_exprs)
    print(f"   ✓ Created cyclic datetime features (sin/cos encoding)")

print(f"   Time: {time.time() - start_time:.2f}s")

# === TRUST VALUE (FREQUENT FLYER ALIGNMENT) ===
print("\n[6/8] Computing trust/alignment features...")
start_time = time.time()

# Extract frequent flyer carrier codes
df = df.with_columns([
    pl.col("frequentFlyer").fill_null("").str.split("/").alias("ff_carriers_list"),
])

# Count matching carriers in segments
carrier_cols = [c for c in mc_exists]
df = df.with_columns([
    pl.sum_horizontal([
        pl.col(c).is_in(pl.col("ff_carriers_list")).cast(pl.Int32)
        for c in carrier_cols
    ]).alias("ff_matches")
])

# Trust value = matching carriers / total segments
df = df.with_columns([
    (pl.col("ff_matches") / (pl.col("total_segments") + 1)).alias("trust_value"),
])

df = df.drop(["ff_carriers_list", "ff_matches"])
print(f"   ✓ Created trust value based on frequent flyer alignment")
print(f"   Time: {time.time() - start_time:.2f}s")

# === CATEGORICAL FEATURES FOR EMBEDDING ===
print("\n[7/8] Extracting categorical features for embeddings...")
start_time = time.time()

# First and last airports
df = df.with_columns([
    pl.col("legs0_segments0_departureFrom_airport_iata").fill_null("UNK").alias("first_departure_airport"),

    # Last arrival - check all possible segments
    pl.coalesce([
        pl.col(f"legs1_segments{i}_arrivalTo_airport_iata")
        for i in range(3, -1, -1)
    ] + [
        pl.col(f"legs0_segments{i}_arrivalTo_airport_iata")
        for i in range(3, -1, -1)
    ]).fill_null("UNK").alias("last_arrival_airport"),
])

# Collect all carriers used (as comma-separated string for embedding)
carrier_list_expr = pl.concat_str([
    pl.col(c).fill_null("") for c in carrier_cols
], separator=",").str.replace_all(r",+", ",").str.strip_chars(",")

df = df.with_columns([
    carrier_list_expr.alias("carriers_used"),
])

print(f"   ✓ Created categorical features: airports and carriers for embedding")
print(f"   Time: {time.time() - start_time:.2f}s")

# === HISTORICAL EMBEDDING ====
# === CARRIER POPULARITY (FROM TRAINING DATA) ===
print("\n[8/8] Computing carrier popularity features...")
start_time = time.time()

# Aggregate carrier popularity across ALL segments (not just first)
all_carrier_pops = []
for l in (0, 1):
    for s in range(4):
        col_name = f'legs{l}_segments{s}_marketingCarrier_code'
        if col_name in df.columns:
            # carrier_pop = train.group_by(col_name).agg(
            carrier_pop = df.group_by(col_name).agg(
                pl.mean('selected').alias(f'carrier_pop_{l}_{s}')
            )
            df = df.join(carrier_pop, on=col_name, how='left')
            df = df.with_columns(pl.col(f'carrier_pop_{l}_{s}').fill_null(0.0))
            all_carrier_pops.append(f'carrier_pop_{l}_{s}')

# Average carrier popularity across all segments
if all_carrier_pops:
    df = df.with_columns([
        pl.mean_horizontal([pl.col(c) for c in all_carrier_pops]).alias("avg_carrier_popularity")
    ])
    # Clean up individual carrier pop columns
    df = df.drop(all_carrier_pops)
else:
    df = df.with_columns(pl.lit(0.0).alias("avg_carrier_popularity"))

print(f"   ✓ Created aggregated carrier popularity from training data")
print(f"   Time: {time.time() - start_time:.2f}s")

# === FINAL CLEANUP: DROP RAW SEGMENT COLUMNS ===
print("\nDropping raw segment-level columns...")
start_time = time.time()

# Identify columns to drop (all segment-level detail columns)
cols_to_drop = [c for c in df.columns if any([
    '_segments' in c and c.endswith('_code'),  # aircraft codes, carrier codes
    '_segments' in c and 'flightNumber' in c,
    '_segments' in c and 'seatsAvailable' in c,
    '_segments' in c and 'baggageAllowance_weightMeasurementType' in c,
    '_segments' in c and 'airport_city_iata' in c,
    '_segments' in c and '_duration' in c,  # Already aggregated
    '_segments' in c and 'baggageAllowance_quantity' in c,  # Already aggregated
    '_segments' in c and 'departureFrom_airport_iata' in c and c != 'legs0_segments0_departureFrom_airport_iata',
    '_segments' in c and 'arrivalTo_airport_iata' in c,
])]

# Also drop legs-level durations (already have total_duration)
cols_to_drop.extend(['legs0_duration', 'legs1_duration'])

df = df.drop([c for c in cols_to_drop if c in df.columns])
print(f"   ✓ Dropped {len([c for c in cols_to_drop if c in df.columns])} redundant segment-level columns")
print(f"   Time: {time.time() - start_time:.2f}s")


# === FINAL CLEANUP: DROP RAW SEGMENT COLUMNS ===
print("\nDropping raw segment-level columns...")
start_time = time.time()

# Identify columns to drop (all segment-level detail columns)
cols_to_drop = [c for c in df.columns if any([
    '_segments' in c and c.endswith('_code'),  # aircraft codes, carrier codes
    '_segments' in c and 'flightNumber' in c,
    '_segments' in c and 'seatsAvailable' in c,
    '_segments' in c and 'baggageAllowance_weightMeasurementType' in c,
    '_segments' in c and 'airport_city_iata' in c,
    '_segments' in c and '_duration' in c,  # Already aggregated
    '_segments' in c and 'baggageAllowance_quantity' in c,  # Already aggregated
    '_segments' in c and 'departureFrom_airport_iata' in c and c != 'legs0_segments0_departureFrom_airport_iata',
    '_segments' in c and 'arrivalTo_airport_iata' in c,
])]

# Also drop legs-level durations (already have total_duration)
cols_to_drop.extend(['legs0_duration', 'legs1_duration'])

df = df.drop([c for c in cols_to_drop if c in df.columns])
print(f"   ✓ Dropped {len([c for c in cols_to_drop if c in df.columns])} redundant segment-level columns")
print(f"   Time: {time.time() - start_time:.2f}s")


# === CREATE CLEAN FEATURE DATAFRAME ===
print("\nCreating clean feature-only DataFrame...")
start_time = time.time()

group_feature = [
    "Id",
    "ranker_id",
    "selected",
]

# Define the core input features (X)
feature_columns = [
    # === PERSONAL/USER DATA ===

    # === NUMERICAL FEATURES ===
    'log_price',
    'tax_ratex100',
    'fee_rate',
    'duration_z_score',
    'duration_ratio',
    'baggage_mean',
    'avg_cabin_class_all',
    'cabin_class_diff_legs',
    'total_segments',
    'trust_value',
    'avg_carrier_popularityx10',

    # === BINARY FEATURES ===
    # 'is_one_way',
    'is_vip_freq', # Or use 'is_vip_freq' if you prefer the engineered version created in step 4
    'has_corporate_tariff',
    'has_access_tp',

    # === CATEGORICAL FEATURES (Uncomment if your DL model uses Embeddings) ===
    # 'carriers_used',
    # 'searchRoute'
]

# === DATETIME FEATURES (Cyclic) ===
# Dynamically add the sin/cos/business_time columns created in step [5/8]
datetime_features = [c for c in df.columns if any(
    suffix in c for suffix in ['_hour_sin', '_hour_cos', '_weekday_sin', '_weekday_cos', '_business_time']
)]

# Combine to get the final model input list
input_features = feature_columns + datetime_features

# Select only the engineered features
df_clean = df.select([c for c in input_features if c in df.columns])
df_id = df.select([c for c in group_feature if c in df.columns])


print(f"   ✓ Created clean DataFrame with {len(df_clean.columns)} engineered features")
print(f"   Time: {time.time() - start_time:.2f}s")

print("\n" + "=" * 60)
print("Simplified Feature Engineering Complete!")
print("=" * 60)
print(f"Final dataset shape: {len(df_clean)} rows × {len(df_clean.columns)} columns")
print("\nFeature categories:")
print(f"  - IDs & Target: Id, ranker_id, selected")
print(f"  - Numerical ({15}): price, duration, baggage, fees, cabin, segments, trust")
print(f"  - Binary ({11}): trip type, direct flights, VIP, baggage, fees, carriers, routes")
print(f"  - Datetime ({len(datetime_features)}): cyclic hour/weekday encoding, business time, days_to_departure")
print(f"  - Categorical ({4}): airports (2), carriers_used, searchRoute")
print(f"  - Popularity ({1}): avg_carrier_popularity")
print(f"\nTotal engineered features: {len(df_clean.columns)}")


# Fill nulls
df_clean = df_clean.with_columns(
    [pl.col(c).fill_null(0) for c in df_clean.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df_clean.select(pl.selectors.string()).columns]
)


print(df_id)
print(df_clean)


# Write output
print("\nWriting features to CSV...")
start_time = time.time()
df_clean[:1000].write_csv("feature_dl_simplified.csv")
print(f"   ✓ Saved {len(df_clean)} rows, {len(df_clean.columns)} columns to 'feature_dl_simplified.csv'")
print(f"   Time: {time.time() - start_time:.2f}s")


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_values(X, max_samples=500, figsize=(16, 8)):
    """
    X: numpy array or torch tensor [num_samples, num_features]
    max_samples: limit how many samples to plot (prevents overload)
    """
    if hasattr(X, "cpu"):
        X = X.cpu().numpy()

    # sample limit for readability
    if X.shape[0] > max_samples:
        X = X[:max_samples]

    num_samples, num_features = X.shape

    plt.figure(figsize=figsize)
    for f in range(num_features):
        plt.plot(X[:, f], alpha=0.4)

    plt.title(f"Feature values for {num_features} features across {num_samples} samples")
    plt.xlabel("Sample index")
    plt.ylabel("Feature value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import torch

def find_nan_features(X):
    """
    Check for NaNs in a feature matrix.

    X: np.ndarray or torch.Tensor [num_samples, num_features]

    Returns:
        nan_rows: indices of rows with NaNs
        nan_cols: indices of columns with NaNs
    """
    if isinstance(X, torch.Tensor):
        nan_mask = torch.isnan(X)
        nan_rows = torch.any(nan_mask, dim=1).nonzero(as_tuple=True)[0].cpu().numpy()
        nan_cols = torch.any(nan_mask, dim=0).nonzero(as_tuple=True)[0].cpu().numpy()
    else:  # assume numpy
        nan_mask = np.isnan(X)
        nan_rows = np.where(np.any(nan_mask, axis=1))[0]
        nan_cols = np.where(np.any(nan_mask, axis=0))[0]

    return nan_rows, nan_cols


X = df_clean.to_numpy()
# X = scaler.fit_transform(df_clean.to_numpy())
# plot_feature_values(X)

nan_rows, nan_cols = find_nan_features(X)

print(f"Rows with NaN: {nan_rows}")
print(f"Columns with NaN: {nan_cols}")
print(f"Number of NaNs in total: {np.isnan(X).sum() if isinstance(X, np.ndarray) else torch.isnan(X).sum()}")

if len(nan_cols) > 0:
    print("Feature indices containing NaN:", nan_cols)
else:
    print("No NaNs in features!")


import torch
from torch.utils.data import Dataset, DataLoader

class RankingDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        features = torch.tensor(entry["features"], dtype=torch.float32)
        pos_idx = torch.tensor(entry["positive_idx"], dtype=torch.long)
        return features, pos_idx, entry["Id"], entry["ranker_id"]



from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler



# ============================================================
# Build samples from groups
# ============================================================

def build_samples(df_clean, df_id, test_sample = False):
    print(f"Data with {len(df_clean)} rows")
    
    groups = defaultdict(list)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean.to_numpy())
    
    # Process ALL rows, not just first 20
    for i in range(len(df_id)):
        id = df_id['Id'][i]
        rid = df_id['ranker_id'][i]
        label = df_id['selected'][i]
        feat = X[i]
        groups[rid].append((feat, label, id))

    print(f"\nNumber of unique rankers: {len(groups)}")

    samples = []
    
    for ranker_id, items in groups.items():
        features_list = []
        id_list = []
        positive_idx = None
        for i, (feat, label, ids) in enumerate(items):
            features_list.append(feat)
            id_list.append(ids)
            if label == 1:
                positive_idx = i

        if not test_sample and positive_idx is None:
            continue
        if test_sample: positive_idx = -1

        features_array = np.stack(features_list, axis=0)

        samples.append({
            "features": features_array,
            "positive_idx": positive_idx,
            "Id": id_list,
            "ranker_id": ranker_id              # <── added
        })    
    return samples

# ============================================================
# Train/Val/Test Split
# ============================================================

train_df_feat, test_df_feat = df_clean[:test_start_idx], df_clean[test_start_idx:]
train_df_id, test_df_id = df_id[:test_start_idx], df_id[test_start_idx:]



train_samples = build_samples(train_df_feat, train_df_id)
train_s, val_s = train_test_split(train_samples, test_size=0.1, shuffle=True)

test_s = build_samples(test_df_feat, test_df_id, test_sample = True)

train_ds = RankingDataset(train_s)
val_ds   = RankingDataset(val_s)
test_ds  = RankingDataset(test_s)
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")


import torch.nn as nn

class MLPScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, layers=3):
        super().__init__()
        blocks = []
        dim = input_dim
        for _ in range(layers):
            blocks.append(nn.Linear(dim, hidden_dim))
            blocks.append(nn.ReLU())
            dim = hidden_dim
        blocks.append(nn.Linear(dim, 1))  # output score per item
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [N, D]
        return self.net(x).squeeze(-1)  # [N]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from dataclasses import dataclass
import numpy as np


# ============================================================
# Training Configuration (Dataclass)
# ============================================================

@dataclass
class TrainingConfig:
    batch_size: int = 128
    hidden_dim: int = 512
    layers: int = 4
    lr: float = 1e-4
    num_epochs: int = 30

    # Dynamically set later
    input_dim: int = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = TrainingConfig()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TPU/XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# --- 1. Optimized Collate Function (Padding & Masking) ---
def bucketed_collate_fn(batch):
    """
    Factory function to create a bucket-based collate function.
    """
    bucket_boundaries = [5, 20, 50, 160, 400, 620, 8300]
    
    # Sort boundaries to ensure correct bucketing
    bucket_boundaries = sorted(bucket_boundaries)
    
    batch_size = len(batch)
    
    # Extract components
    raw_feats = [b[0] for b in batch]
    pos_idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    
    # Get dimensions
    lengths = torch.tensor([f.shape[0] for f in raw_feats], dtype=torch.long)
    max_len_in_batch = lengths.max().item()
    feature_dim = raw_feats[0].shape[1]
    
    # Determine the appropriate bucket size
    padded_len = max_len_in_batch
    for boundary in bucket_boundaries:
        if max_len_in_batch <= boundary:
            padded_len = boundary
            break
    else:
        # If exceeds all boundaries, use the max length in batch
        # or optionally round up to nearest multiple
        padded_len = max_len_in_batch
    
    # Pre-allocate tensors with bucket size
    padded_features = torch.zeros((batch_size, padded_len, feature_dim), dtype=torch.float32)
    attention_masks = torch.zeros((batch_size, padded_len), dtype=torch.bool)
    
    # Vectorized padding
    for i, f in enumerate(raw_feats):
        curr_len = f.shape[0]
        padded_features[i, :curr_len] = f
        attention_masks[i, :curr_len] = True
    
    return padded_features, pos_idx, attention_masks
    
# --- 2. Vectorized Loss Function ---
def batch_ranking_loss(scores, pos_idxs, mask):
    """
    scores:   [Batch, Max_Len]
    pos_idxs: [Batch]
    mask:     [Batch, Max_Len]
    """
    scores = scores.masked_fill(mask == 0, -1e9)
    return F.cross_entropy(scores, pos_idxs)

# --- 3. FIXED Single-Core TPU Training ---
def train_model(config, train_dataset, val_dataset=None):
    device = xm.xla_device()
    xm.master_print(f"Training on TPU device: {device}")
    
    xm.master_print(f"Batch size: {config.batch_size}")
    
    # DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=bucketed_collate_fn,
        num_workers=4,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,  # FIX: Reuse workers
    )
    
    val_device_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=bucketed_collate_fn,
            num_workers=4,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    
    config.input_dim = train_dataset[0][0].shape[-1]
    
    model = MLPScoreModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers
    ).to(device)
    
    # Enable bfloat16 for 2-3x speedup on TPU
    use_bfloat16 = getattr(config, 'use_bfloat16', True)
    if use_bfloat16:
        xm.master_print("Converting model to bfloat16 for faster training...")
        model = model.to(torch.bfloat16)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=0.01
    )
    
    xm.master_print("Setup complete. Start training...")
    
    # FIX: More frequent marking to prevent graph explosion
    log_steps = max(len(train_loader) // 5, 1)  # Log 20 times per epoch
    
    xm.master_print(f"Log every {log_steps} steps")
    
    try:
        for epoch in range(config.num_epochs):
            model.train()
            
            # FIX: Use scalar accumulation instead of tensor accumulation
            epoch_loss_sum = 0.0
            step_count = 0
            
            # FIX: Recreate ParallelLoader each epoch to avoid exhaustion
            train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
            
            for step, (batch_feats, batch_pos, batch_masks) in enumerate(train_device_loader):
                
                # Cast inputs to bfloat16 if enabled
                if use_bfloat16:
                    batch_feats = batch_feats.to(torch.bfloat16)
                
                out = model(batch_feats)
                scores = out.squeeze(-1)
                
                # Loss calculation stays in float32 for numerical stability
                if use_bfloat16:
                    scores = scores.to(torch.float32)
                
                loss = batch_ranking_loss(scores, batch_pos, batch_masks)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                
                # FIX: Convert to Python scalar immediately to avoid memory buildup
                epoch_loss_sum += loss.item()
                step_count += 1
                
                # FIX: Mark step regularly to prevent graph explosion
                xm.mark_step()
                
                # # Logging
                # if step_count % log_steps == 0:
                #     avg_loss = epoch_loss_sum / step_count
                #     xm.master_print(f"[Epoch {epoch+1}] Step {step}/{len(train_loader)} - Loss: {avg_loss:.4f}")
            
            # Mark step at epoch end
            xm.mark_step()
            
            final_avg_loss = epoch_loss_sum / step_count
            xm.master_print(f"[Epoch {epoch+1}] Train Loss: {final_avg_loss:.4f}")
            
            # ---- Validation ----
            if val_dataset and val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_steps = 0

                xm.master_print("Start Validating")
                
                # FIX: Recreate ParallelLoader for validation each epoch
                val_device_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                
                with torch.no_grad():
                    for step, (batch_feats, batch_pos, batch_masks) in enumerate(val_device_loader):
                        
                        # Cast to bfloat16 if enabled
                        if use_bfloat16:
                            batch_feats = batch_feats.to(torch.bfloat16)
                        
                        out = model(batch_feats)
                        scores = out.squeeze(-1)
                        
                        # Convert back to float32 for loss
                        if use_bfloat16:
                            scores = scores.to(torch.float32)
                        
                        loss = batch_ranking_loss(scores, batch_pos, batch_masks)
                        
                        # FIX: Immediate conversion to Python scalar
                        val_loss_sum += loss.item()
                        val_steps += 1
                        
                        # FIX: Mark step every iteration in validation
                        xm.mark_step()
                
                xm.mark_step()
                avg_val_loss = val_loss_sum / val_steps
                xm.master_print(f"            Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint less frequently
            if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
                xm.master_print(f"Saving checkpoint at epoch {epoch+1}...")
                xm.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
                xm.mark_step()  # FIX: Mark step after save

    except Exception as e:
        xm.master_print(f"Training interrupted: {e}")
        raise
    finally:
        # FIX: Cleanup to prevent issues on reruns
        xm.master_print("Cleaning up...")
        xm.mark_step()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    xm.master_print("Training complete!")
    
    # Convert model back to float32 for saving/inference
    if use_bfloat16:
        model = model.to(torch.float32)
    
    return model


model = train_model(config, train_ds, val_ds)


import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import polars
import numpy as np

# Get TPU device
device = xm.xla_device()

def bucketed_test_collate_fn(batch):
    """
    Bucket-based collate function for efficient padding.
    """
    bucket_boundaries = [5, 20, 50, 160, 400, 620, 8300]
    bucket_boundaries = sorted(bucket_boundaries)
    
    batch_size = len(batch)
    
    # Extract components
    raw_feats = [torch.tensor(b[0], dtype=torch.float32) for b in batch]
    pos_idx = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    rank_ids = [b[3] for b in batch]
    
    # Get dimensions
    lengths = torch.tensor([f.shape[0] for f in raw_feats], dtype=torch.long)
    max_len_in_batch = lengths.max().item()
    feature_dim = raw_feats[0].shape[1]
    
    # Determine the appropriate bucket size
    padded_len = max_len_in_batch
    for boundary in bucket_boundaries:
        if max_len_in_batch <= boundary:
            padded_len = boundary
            break
    
    # Pre-allocate tensors with bucket size
    padded_features = torch.zeros((batch_size, padded_len, feature_dim), dtype=torch.float32)
    attention_masks = torch.zeros((batch_size, padded_len), dtype=torch.bool)
    
    # Fill in the data
    for i, f in enumerate(raw_feats):
        curr_len = f.shape[0]
        padded_features[i, :curr_len] = f
        attention_masks[i, :curr_len] = True
    
    return padded_features, lengths, attention_masks, pos_idx, ids, rank_ids

# DataLoader with larger batch size
BATCH_SIZE = 32  # Adjust based on your TPU memory
val_loader = DataLoader(
    val_ds, 
    collate_fn=bucketed_test_collate_fn, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# Wrap with MpDeviceLoader for TPU
val_loader = pl.MpDeviceLoader(val_loader, device)

# ============================================================
# Inference on val_ds → build val_df
# ============================================================
model.eval()
model = model.to(device)  # Ensure model is on TPU

all_ids = []
all_ranker_ids = []
all_pred_scores = []
all_selected = []

with torch.no_grad():
    for batch_idx, (padded_features, lengths, attention_masks, pos_idx_list, ids_list, rank_ids_list) in enumerate(val_loader):
        # All tensors are already on TPU device via MpDeviceLoader
        # padded_features: (batch_size, padded_len, feature_dim)
        # lengths: (batch_size,)
        # attention_masks: (batch_size, padded_len)
        
        batch_size = padded_features.size(0)
        
        # Forward pass - pass attention mask if your model supports it
        preds_batch = model(padded_features)  # (batch_size, padded_len)
        
        # Mark step every few batches for optimal TPU performance
        if batch_idx % 5 == 0:
            xm.mark_step()
        
        # Move to CPU for processing
        preds_cpu = preds_batch.cpu()
        lengths_cpu = lengths.cpu()
        xm.mark_step()  # Ensure transfer completes
        
        preds_np = preds_cpu.numpy()
        lengths_np = lengths_cpu.numpy()
        
        # Process each item in the batch
        for i in range(batch_size):
            L = lengths_np[i]  # Actual length (before padding)
            preds_item = preds_np[i, :L]  # Only take non-padded predictions
            pos_idx = pos_idx_list[i]
            id_group = ids_list[i]
            rid = rank_ids_list[i]
            
            # Build labels for the whole group
            selected = np.zeros(L, dtype=int)
            selected[pos_idx] = 1
            
            # Store - handle if id_group is a list or single value
            if isinstance(id_group, list):
                all_ids.extend(id_group)
            else:
                all_ids.extend([id_group] * L)
            
            all_ranker_ids.extend([rid] * L)
            all_pred_scores.extend(preds_item.tolist())
            all_selected.extend(selected.tolist())

# Final mark_step to complete all pending operations
xm.mark_step()

# Convert to Polars DataFrame
val_df = polars.DataFrame({
    "Id": all_ids,
    "ranker_id": all_ranker_ids,
    "pred_score": all_pred_scores,
    "selected": all_selected
})

# Add group_size
val_df = val_df.join(
    val_df.group_by("ranker_id").agg(polars.len().alias("group_size")),
    on="ranker_id"
)

print(val_df)


# ============================================================
# Visualization on val_df
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

# Color palette
red = (0.86, 0.08, 0.24)
blue = (0.12, 0.56, 1.0)



# ============================================================
# val_df requirement
# Columns available:
#   - ranker_id
#   - pred_score
#   - selected
#   - group_size
# ============================================================

# Keep only groups with more than 10 items
va_df = val_df.filter(pl.col("group_size") > 10)

# Compute quantiles of group sizes (unique per ranker_id)
size_quantiles = (
    va_df.select("ranker_id", "group_size")
         .unique()
         .select(
             pl.col("group_size").quantile(0.25).alias("q25"),
             pl.col("group_size").quantile(0.50).alias("q50"),
             pl.col("group_size").quantile(0.75).alias("q75")
         )
         .to_dicts()[0]
)



# ============================================================
# HitRate@k Curve Function
# ============================================================
def calculate_hitrate_curve(df, k_values):
    sorted_df = df.sort(["ranker_id", "pred_score"], descending=[False, True])
    return [
        (
            sorted_df.group_by("ranker_id", maintain_order=True)
            .head(k)
            .group_by("ranker_id")
            .agg(pl.col("selected").max().alias("hit"))
            .select(pl.col("hit").mean())
            .item()
        )
        for k in k_values
    ]


k_values = list(range(1, 21))

curves = {
    'All groups (>10)': calculate_hitrate_curve(va_df, k_values),
    f"Small (11-{int(size_quantiles['q25'])})": calculate_hitrate_curve(
        va_df.filter(pl.col('group_size') <= size_quantiles['q25']), k_values
    ),
    f"Medium ({int(size_quantiles['q25']+1)}-{int(size_quantiles['q75'])})": calculate_hitrate_curve(
        va_df.filter(
            (pl.col('group_size') > size_quantiles['q25']) &
            (pl.col('group_size') <= size_quantiles['q75'])
        ),
        k_values
    ),
    f"Large (>{int(size_quantiles['q75'])})": calculate_hitrate_curve(
        va_df.filter(pl.col('group_size') > size_quantiles['q75']),
        k_values
    ),
}



# ============================================================
# HitRate@3 vs Group Size (log-binned)
# ============================================================

# Create log bins
min_size = va_df['group_size'].min()
max_size = va_df['group_size'].max()
bins = np.logspace(np.log10(min_size), np.log10(max_size), 51)

# One HitRate@3 per ranker
ranker_hr3 = (
    va_df.sort(["ranker_id", "pred_score"], descending=[False, True])
         .group_by("ranker_id", maintain_order=True)
         .agg([
             pl.col("selected").head(3).max().alias("hit_top3"),
             pl.col("group_size").first()
         ])
)

# Assign bins
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_indices = np.digitize(ranker_hr3["group_size"].to_numpy(), bins) - 1

size_analysis = (
    pl.DataFrame({
        "bin_idx": bin_indices,
        "bin_center": bin_centers[np.clip(bin_indices, 0, len(bin_centers)-1)],
        "hit_top3": ranker_hr3["hit_top3"]
    })
    .group_by(["bin_idx", "bin_center"])
    .agg([
        pl.col("hit_top3").mean().alias("hitrate3"),
        pl.len().alias("n_groups")
    ])
    .filter(pl.col("n_groups") >= 3)
    .sort("bin_center")
)



# ============================================================
# Plot Figures
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=400)

# Left: HitRate@k Curves
colors = ["black"]  # All groups
for i in range(3):
    t = i / 2
    color = tuple(blue[j] * (1 - t) + red[j] * t for j in range(3))
    colors.append(color)

for (label, hitrates), color in zip(curves.items(), colors):
    ax1.plot(k_values, hitrates, marker='o', label=label, color=color, markersize=3)

ax1.set_xlabel("k (top-k predictions)")
ax1.set_ylabel("HitRate@k")
ax1.set_title("HitRate@k by Group Size")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 21)
ax1.set_ylim(-0.025, 1.025)

# Right: HitRate@3 vs Group Size
ax2.scatter(
    size_analysis["bin_center"],
    size_analysis["hitrate3"],
    s=30,
    alpha=0.6,
    color=blue
)
ax2.set_xlabel("Group Size")
ax2.set_ylabel("HitRate@3")
ax2.set_title("HitRate@3 vs Group Size")
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Summary
print(f"HitRate@1: {curves['All groups (>10)'][0]:.3f}")
print(f"HitRate@3: {curves['All groups (>10)'][2]:.3f}")
print(f"HitRate@5: {curves['All groups (>10)'][4]:.3f}")
print(f"HitRate@10: {curves['All groups (>10)'][9]:.3f}")


import torch
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as xla_pl
import polars as pl
from torch.utils.data import DataLoader

# Get TPU device
device = xm.xla_device()

# DataLoader with bucketed collate function
BATCH_SIZE = 32  # Adjust based on TPU memory and your data
test_loader = DataLoader(
    test_ds, 
    collate_fn=bucketed_test_collate_fn,
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# Wrap the DataLoader with ParallelLoader for TPU optimization
test_loader = xla_pl.MpDeviceLoader(test_loader, device)

# ============================================================
# Inference on test_ds → build test_df
# ============================================================
model.eval()
model = model.to(device)  # Ensure model is on TPU

all_ids = []
all_ranker_ids = []
all_pred_scores = []
all_selected = []

with torch.no_grad():
    for batch_idx, (padded_features, lengths, attention_masks, pos_idx_list, ids_list, rank_ids_list) in enumerate(test_loader):
        # All tensors already on TPU device via MpDeviceLoader
        # padded_features: (batch_size, padded_len, feature_dim)
        # lengths: (batch_size,)
        # attention_masks: (batch_size, padded_len)
        
        batch_size = padded_features.size(0)
        
        # Forward pass with attention mask (if your model supports it)
        preds_batch = model(padded_features)
        
        # Mark step every few batches for optimal TPU performance
        if batch_idx % 5 == 0:
            xm.mark_step()
        
        # Move to CPU for numpy conversion
        preds_cpu = preds_batch.cpu()
        lengths_cpu = lengths.cpu()
        xm.mark_step()  # Ensure transfer completes
        
        preds_np = preds_cpu.numpy()
        lengths_np = lengths_cpu.numpy()
        
        # Process each item in the batch
        for i in range(batch_size):
            L = lengths_np[i]  # Actual length (before padding)
            preds_item = preds_np[i, :L]  # Only take non-padded predictions
            id_group = ids_list[i]
            rid = rank_ids_list[i]
            
            # Build labels based on MAX SCORE (Argmax)
            selected = np.zeros(L, dtype=int)
            
            # Find the index where the model output is highest
            best_idx = np.argmax(preds_item)
            
            # Set that index to 1
            selected[best_idx] = 1
            
            # Store - handle if id_group is a list or single value
            if isinstance(id_group, list):
                all_ids.extend(id_group)
            else:
                all_ids.extend([id_group] * L)
            
            all_ranker_ids.extend([rid] * L)
            all_pred_scores.extend(preds_item.tolist())
            all_selected.extend(selected.tolist())

# Final mark_step to complete all pending operations
xm.mark_step()

# Convert to Polars DataFrame
test_df = pl.DataFrame({
    "Id": all_ids,
    "ranker_id": all_ranker_ids,
    "pred_score": all_pred_scores,
    "selected": all_selected  # Now contains 1 for the highest score, 0 for others
})

# Add group_size
test_df = test_df.join(
    test_df.group_by("ranker_id").agg(pl.len().alias("group_size")),
    on="ranker_id"
)

# Filter to show only selected items
filtered_df = test_df.filter(pl.col("selected") == 1)
print(filtered_df)
print(f"Total groups: {test_df['ranker_id'].n_unique()}")
print(f"Selected items: {len(filtered_df)}")


# ============================================================
# Build submission from test_df + model predictions
# ============================================================

# test_df must contain:
#   - Id
#   - ranker_id
#   - pred_score  (already added when building test_df)

submission_df = (
    test_df
    .select(["Id", "ranker_id", "pred_score"])
    .with_columns(
        # Rank pred_score within each ranker_id group
        # Higher pred_score → Lower rank number (rank 1 = best)
        pl.col("pred_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("selected")
    )
    .select(["Id", "ranker_id", "selected"])
    # CRITICAL: Sort by Id to preserve original test.csv row order
    .sort("Id")
)

# Save to CSV
submission_df.write_csv("submission.csv")

print("Submission saved to submission.csv")
print(f"Total rows: {len(submission_df)}")

# Validation checks
print("\n=== Validation Checks ===")
validation = submission_df.group_by("ranker_id").agg([
    pl.col("selected").min().alias("min_rank"),
    pl.col("selected").max().alias("max_rank"),
    pl.col("selected").n_unique().alias("unique_ranks"),
    pl.col("selected").count().alias("n_flights")
])

# Check if ranks form valid permutations (1, 2, 3, ..., N)
invalid = validation.filter(
    (pl.col("min_rank") != 1) |
    (pl.col("max_rank") != pl.col("n_flights")) |
    (pl.col("unique_ranks") != pl.col("n_flights"))
)

if len(invalid) > 0:
    print(f"⚠️  WARNING: {len(invalid)} ranker_ids have invalid rank permutations!")
    print(invalid)
else:
    print("✓ All ranker_ids have valid rank permutations (1, 2, 3, ..., N)")

print(f"✓ Total unique ranker_ids: {submission_df['ranker_id'].n_unique()}")


!kaggle competitions submit -c aeroclub-recsys-2025 -f submission.csv -m "Deep Learning Ranking Model"

