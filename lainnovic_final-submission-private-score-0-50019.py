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


print("super starrrr")


%%capture
!pip install -U xgboost
!pip install -U polars

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
TRAIN_VAL_SPLIT = 16487352
PENALTY_FACTOR = 0.1

# Set random seed
np.random.seed(RANDOM_STATE)

class FlightDataProcessor:
    """Encapsulate data processing logic for flight recommendation system."""
    
    def __init__(self):
        self.categorical_features = []
        self.feature_columns = []
        
    def load_data(self, train_path: str, test_path: str) -> pl.DataFrame:
        """Load and combine train/test data."""
        logger.info("Loading data...")
        train = pl.read_parquet(train_path).drop('__index_level_0__')
        test = (pl.read_parquet(test_path)
                .drop('__index_level_0__')
                .with_columns(pl.lit(0, dtype=pl.Int64).alias("selected")))
        
        return pl.concat((train, test))
    
    @staticmethod
    def duration_to_minutes(col: pl.Expr) -> pl.Expr:
        """Convert duration string to minutes more efficiently."""
        # Extract days and time parts in one pass
        days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
        time_str = pl.when(col.str.contains(r"^\d+\.")).then(
            col.str.replace(r"^\d+\.", "")
        ).otherwise(col)
        hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
        minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
        
        return (days + hours + minutes).fill_null(0)
    
    def create_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create price-related features."""
        return df.with_columns([
            (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
            pl.col("totalPrice").log1p().alias("log_price"),
        ])
    
    def create_duration_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create duration-related features."""
        return df.with_columns([
            (pl.col("legs0_duration").fill_null(0) + 
             pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
            pl.when(pl.col("legs1_duration").fill_null(0) > 0)
                .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
                .otherwise(1.0).alias("duration_ratio"),
        ])
    
    def create_trip_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create trip-type and routing features."""
        # Check for marketing carrier columns
        mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' 
                  for l in (0, 1) for s in range(4)]
        mc_exists = [col for col in mc_cols if col in df.columns]
        
        return df.with_columns([
            # Trip type
            (pl.col("legs1_duration").is_null() | 
             (pl.col("legs1_duration") == 0) | 
             pl.col("legs1_segments0_departureFrom_airport_iata").is_null())
                .cast(pl.Int32).alias("is_one_way"),
            
            # Segment counts
            (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) 
                             for col in mc_exists) if mc_exists else pl.lit(0))
                .alias("l0_seg"),
            
            # Route popularity
            pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", 
                                       "MOWLED", "LEDMOW"])
                .cast(pl.Int32).alias("is_popular_route"),
        ])
    
    def create_passenger_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create passenger-related features."""
        return df.with_columns([
            # Frequent flyer
            (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + 
             (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32))
                .alias("n_ff_programs"),
            
            # Binary features
            pl.col("corporateTariffCode").is_not_null().cast(pl.Int32)
                .alias("has_corporate_tariff"),
            (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32)
                .alias("has_access_tp"),
            
            # Cancellation/exchange rules
            ((pl.col("miniRules0_monetaryAmount") == 0) & 
             (pl.col("miniRules0_statusInfos") == 1))
                .cast(pl.Int8).alias("free_cancel"),
            ((pl.col("miniRules1_monetaryAmount") == 0) & 
             (pl.col("miniRules1_statusInfos") == 1))
                .cast(pl.Int8).alias("free_exchange"),
        ])
    
    def create_cabin_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create cabin class features."""
        return df.with_columns([
            pl.mean_horizontal(["legs0_segments0_cabinClass", 
                              "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
            (pl.col("legs0_segments0_cabinClass").fill_null(0) - 
             pl.col("legs1_segments0_cabinClass").fill_null(0))
                .alias("cabin_class_diff"),
        ])
    
    def create_segment_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create segment-based features."""
        # Create segment counts
        seg_exprs = []
        for leg in (0, 1):
            seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) 
                       if f"legs{leg}_segments{s}_duration" in df.columns]
            if seg_cols:
                seg_exprs.append(
                    pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                        .cast(pl.Int32).alias(f"n_segments_leg{leg}")
                )
            else:
                seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
        
        df = df.with_columns(seg_exprs)
        
        # Derived features
        return df.with_columns([
            (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
            (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
            pl.when(pl.col("is_one_way") == 1).then(0)
                .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32))
                .alias("is_direct_leg1"),
        ]).with_columns([
            (pl.col("is_direct_leg0") & pl.col("is_direct_leg1"))
                .cast(pl.Int32).alias("both_direct"),
            ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0))
                .cast(pl.Int32).alias("is_vip_freq"),
            pl.col("Id").count().over("ranker_id").alias("group_size"),
        ])
    
    def create_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create time-based features."""
        time_exprs = []
        time_cols = ["legs0_departureAt", "legs0_arrivalAt", 
                    "legs1_departureAt", "legs1_arrivalAt"]
        
        for col in time_cols:
            if col in df.columns:
                dt = pl.col(col).str.to_datetime(strict=False)
                h = dt.dt.hour().fill_null(12)
                time_exprs.extend([
                    h.alias(f"{col}_hour"),
                    dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
                    (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20)))
                        .cast(pl.Int32).alias(f"{col}_business_time")
                ])
        
        return df.with_columns(time_exprs) if time_exprs else df
    
    def create_ranking_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create ranking and competitive features."""
        df = df.with_columns([
            pl.col("group_size").log1p().alias("group_size_log"),
        ])
        
        # Basic ranks
        rank_exprs = [
            pl.col("totalPrice").rank().over("ranker_id").alias("price_rank"),
            pl.col("total_duration").rank().over("ranker_id").alias("duration_rank"),
        ]
        
        # Price-specific features
        price_exprs = [
            (pl.col("totalPrice").rank("average").over("ranker_id") / 
             pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
            (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id"))
                .cast(pl.Int32).alias("is_cheapest"),
            ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / 
             (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
            (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id"))
                .cast(pl.Int32).alias("is_min_segments"),
        ]
        
        return df.with_columns(rank_exprs + price_exprs)
    
    def add_carrier_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add carrier-related features."""
        # Major carrier flag
        if "legs0_segments0_marketingCarrier_code" in df.columns:
            df = df.with_columns(
                pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"])
                    .cast(pl.Int32).alias("is_major_carrier")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("is_major_carrier"))
        
        return df
    
    def add_direct_cheapest_feature(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add direct cheapest flight feature."""
        direct_cheapest = (
            df.filter(pl.col("is_direct_leg0") == 1)
            .group_by("ranker_id")
            .agg(pl.col("totalPrice").min().alias("min_direct"))
        )
        
        return (df.join(direct_cheapest, on="ranker_id", how="left")
                .with_columns(
                    ((pl.col("is_direct_leg0") == 1) & 
                     (pl.col("totalPrice") == pl.col("min_direct")))
                        .cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
                ).drop("min_direct"))
    
    def add_popularity_features(self, df: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
        """Add carrier popularity features based on training data."""
        carrier0_pop = train.group_by('legs0_segments0_marketingCarrier_code').agg(
            pl.mean('selected').alias('carrier0_pop')
        )
        carrier1_pop = train.group_by('legs1_segments0_marketingCarrier_code').agg(
            pl.mean('selected').alias('carrier1_pop')
        )
        
        return (df.join(carrier0_pop, on='legs0_segments0_marketingCarrier_code', how='left')
                .join(carrier1_pop, on='legs1_segments0_marketingCarrier_code', how='left')
                .with_columns([
                    pl.col('carrier0_pop').fill_null(0.0),
                    pl.col('carrier1_pop').fill_null(0.0),
                ])
                .with_columns([
                    (pl.col('carrier0_pop') * pl.col('carrier1_pop'))
                        .alias('carrier_pop_product'),
                ]))
    
    def process_durations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process duration columns."""
        dur_cols = ["legs0_duration", "legs1_duration"] + [
            f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)
        ]
        dur_exprs = [self.duration_to_minutes(pl.col(c)).alias(c) 
                    for c in dur_cols if c in df.columns]
        
        return df.with_columns(dur_exprs) if dur_exprs else df
    
    def get_categorical_features(self) -> List[str]:
        """Define categorical features."""
        return [
            'nationality', 'searchRoute', 'corporateTariffCode',
            'bySelf', 'sex', 'companyID',
            # Leg 0 segments
            'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
            'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
            'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
            'legs0_segments0_flightNumber',
            'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
            'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
            'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
            'legs0_segments1_flightNumber',
            # Leg 1 segments
            'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
            'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
            'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
            'legs1_segments0_flightNumber',
            'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
            'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
            'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
            'legs1_segments1_flightNumber',
        ]
    
    def get_exclude_columns(self) -> List[str]:
        """Define columns to exclude from features."""
        exclude_cols = [
            'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
            'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
            'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
            'frequentFlyer',  # Already processed
            'pricingInfo_passengerCount'  # Constant
        ]
        
        # Add high-missing segments
        for leg in [0, 1]:
            for seg in [0, 1]:
                if seg == 0:
                    suffixes = ["seatsAvailable"]
                else:
                    suffixes = [
                        "cabinClass", "seatsAvailable", "baggageAllowance_quantity",
                        "baggageAllowance_weightMeasurementType", "aircraft_code",
                        "arrivalTo_airport_city_iata", "arrivalTo_airport_iata",
                        "departureFrom_airport_iata", "flightNumber",
                        "marketingCarrier_code", "operatingCarrier_code",
                    ]
                for suffix in suffixes:
                    exclude_cols.append(f"legs{leg}_segments{seg}_{suffix}")
        
        # Exclude segment 2-3 columns (>98% missing)
        for leg in [0, 1]:
            for seg in [2, 3]:
                for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 
                             'arrivalTo_airport_iata', 'baggageAllowance_quantity',
                             'baggageAllowance_weightMeasurementType', 'cabinClass',
                             'departureFrom_airport_iata', 'duration', 'flightNumber',
                             'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
                    exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')
        
        return exclude_cols
    
    def process_features(self, df: pl.DataFrame, train: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """Main feature processing pipeline."""
        logger.info("Processing durations...")
        df = self.process_durations(df)
        
        logger.info("Creating features...")
        df = self.create_price_features(df)
        df = self.create_duration_features(df)
        df = self.create_trip_features(df)
        df = self.create_passenger_features(df)
        df = self.create_cabin_features(df)
        df = self.create_segment_features(df)
        df = self.create_time_features(df)
        df = self.create_ranking_features(df)
        df = self.add_carrier_features(df)
        df = self.add_direct_cheapest_feature(df)
        df = self.add_popularity_features(df, train)
        
        # Fill nulls
        logger.info("Filling nulls...")
        df = df.with_columns(
            [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
            [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
        )
        
        # Get feature columns
        cat_features = self.get_categorical_features()
        exclude_cols = self.get_exclude_columns()
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        cat_features_final = [col for col in cat_features if col in feature_cols]
        
        logger.info(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")
        
        self.categorical_features = cat_features_final
        self.feature_columns = feature_cols
        
        return df, feature_cols

def hitrate_at_3(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Calculate hit rate at 3."""
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

def re_rank(test: pl.DataFrame, submission_xgb: pl.DataFrame, 
           penalty_factor: float = PENALTY_FACTOR) -> pl.DataFrame:
    """Re-rank submissions to avoid duplicate flights."""
    COLS_TO_COMPARE = [
        "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt",
        "legs0_segments0_flightNumber", "legs1_segments0_flightNumber",
        "legs0_segments0_aircraft_code", "legs1_segments0_aircraft_code",
        "legs0_segments0_departureFrom_airport_iata", "legs1_segments0_departureFrom_airport_iata",
    ]

    test_processed = test.with_columns(
        [pl.col(c).cast(str).fill_null("NULL") for c in COLS_TO_COMPARE]
    )

    df = submission_xgb.join(test_processed, on=["Id", "ranker_id"], how="left")

    # Create flight hash
    df = df.with_columns(
        pl.concat_str([pl.col(c) for c in COLS_TO_COMPARE], separator="_")
            .alias("flight_hash")
    )

    # Apply penalty for duplicate flights
    df = df.with_columns(
        pl.max("pred_score").over(["ranker_id", "flight_hash"])
            .alias("max_score_same_flight")
    ).with_columns(
        (pl.col("pred_score") - 
         penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score")))
            .alias("reorder_score")
    ).with_columns(
        pl.col("reorder_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("new_selected")
    )

    return df.select(["Id", "ranker_id", "new_selected", "pred_score", "reorder_score"])

def main():
    """Main execution function."""
    # Initialize processor
    processor = FlightDataProcessor()
    
    # Load data
    data_raw = processor.load_data(
        '/kaggle/input/aeroclub-recsys-2025/train.parquet',
        '/kaggle/input/aeroclub-recsys-2025/test.parquet'
    )
    
    # Load train separately for popularity features
    train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
    test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__')
    
    # Process features
    data, feature_cols = processor.process_features(data_raw, train)
    
    # Prepare data for XGBoost
    X = data.select(feature_cols)
    y = data.select('selected')
    groups = data.select('ranker_id')
    
    # Encode categorical features
    data_xgb = X.with_columns([
        (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) 
        for c in processor.categorical_features
    ])
    
    # Split data
    n2 = train.height
    data_xgb_tr = data_xgb[:TRAIN_VAL_SPLIT]
    data_xgb_va = data_xgb[TRAIN_VAL_SPLIT:n2]
    data_xgb_te = data_xgb[n2:]
    
    y_tr, y_va, y_te = y[:TRAIN_VAL_SPLIT], y[TRAIN_VAL_SPLIT:n2], y[n2:]
    groups_tr, groups_va, groups_te = groups[:TRAIN_VAL_SPLIT], groups[TRAIN_VAL_SPLIT:n2], groups[n2:]
    
    # Prepare group sizes for XGBoost
    group_sizes_tr = groups_tr.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
    group_sizes_va = groups_va.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
    group_sizes_te = groups_te.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=data_xgb.columns)
    dval = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=data_xgb.columns)
    dtest = xgb.DMatrix(data_xgb_te, label=y_te, group=group_sizes_te, feature_names=data_xgb.columns)

    print("nodel training is finally starting")
    # XGBoost parameters
    xgb_params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@3',
        "learning_rate": 0.022641389657079056,
        "max_depth": 14,
        "min_child_weight": 2,
        "subsample": 0.8842234913702768,
        "colsample_bytree": 0.45840689146263086,
        "gamma": 3.3084297630544888,
        "lambda": 6.952586917313028,
        "alpha": 0.6395254133055179,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # Train model
    logger.info("Training XGBoost model...")
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=50
    )
    
    # Generate predictions
    logger.info("Generating predictions...")
    submission_xgb = (
        test.select(['Id', 'ranker_id'])
        .with_columns(pl.Series('pred_score', xgb_model.predict(dtest)))
        .with_columns(
            pl.col('pred_score')
            .rank(method='ordinal', descending=True)
            .over('ranker_id')
            .cast(pl.Int32)
            .alias('selected')
        )
        .select(['Id', 'ranker_id', 'selected', 'pred_score'])
    )
    
    # Apply re-ranking
    logger.info("Applying re-ranking...")
    top = re_rank(test, submission_xgb)
    submission_final = (
        submission_xgb.join(top, on=["Id", "ranker_id"], how="left")
        .with_columns([
            pl.when(pl.col("new_selected").is_not_null())
            .then(pl.col("new_selected"))
            .otherwise(pl.col("selected"))
            .alias("selected")
        ])
        .select(["Id", "ranker_id", "selected"])
    )
    
    # Save submission
    logger.info("Saving submission...")
    submission_final.write_csv('submission.csv')
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

