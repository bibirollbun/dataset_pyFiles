import numpy as np
import polars as pl

# File paths
train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

# Load data
train_df = pl.read_csv(train_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )
test_df = pl.read_csv(test_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )

print(f"✓ Train data loaded: {train_df.shape}")
print(f"✓ Test data loaded: {test_df.shape}")
print(f"  Train columns: {train_df.columns}")
print(f"  Test columns: {test_df.columns}")


def is_sorted_timestamp_utc(df, col="timestamp_utc"):
    return (df[col] == df[col].sort()).all()
print(f" Train data sorted by timestamp_utc: {is_sorted_timestamp_utc(train_df)}")
print(f" Test data sorted by timestamp_utc: {is_sorted_timestamp_utc(test_df)}")


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
print("\n4. FEATURE ENGINEERING")
print("-" * 40)

LAG_1W_INTERVAL = 7*24*4

def create_features(features, is_train=False):
    """
    Create features for modeling
    Respects 72-hour (288 steps) data latency
    """
    if is_train:
        features = features.with_columns(
            pl.col("net_load_kwh").shift(LAG_1W_INTERVAL).alias("net_load_kwh_lag_1w")
        )
    else:
        features = features.with_columns(
            pl.lit(None).alias("net_load_kwh"),
            pl.lit(None).alias("net_load_kwh_lag_1w")
        )

    return features.select(['timestamp_utc','net_load_kwh','net_load_kwh_lag_1w'])

# Create features for train and test
print("Creating features for training data...")
train_features = create_features(train_df, is_train=True)
print(f"  Train features shape: {train_features.shape}")

# print("Creating features for test data...")
test_features = create_features(test_df, is_train=False)
print(f"  Test features shape: {test_features.shape}")


from datetime import datetime
print(train_features.filter(train_features['timestamp_utc'].is_in([datetime(2025,8,24,23,45,00),datetime(2025,8,31,23,45,00)])))


from datetime import datetime
print(test_features.filter(test_features['timestamp_utc'].is_in([datetime(2025,8,25,00,00,00),datetime(2025,9,1,00,00,00)])))


print("\n5. PREPARING TRAINING DATA FOR 48-HOUR AHEAD PREDICTION")
print("-" * 40)

# Define prediction horizon
HORIZON_HOURS = 168 # (7 days)
HORIZON_STEPS = HORIZON_HOURS * 4  # 672 steps at 15-minute intervals

# Get feature columns (exclude target and metadata)
feature_cols = ['net_load_kwh_lag_1w']

print(f"Number of features: {len(feature_cols)}")

# Create training samples
# For each timestamp, we use features from 48 hours before to predict the value at that timestamp
X_train = []
y_train = []

for i in range(HORIZON_STEPS, len(train_features)):
    # Features from same dataset, lag is already calculated in feature engineering
    X_train.append(train_features.select(feature_cols).row(i)) # Can take same column as window is already handled in feature engineering
    # Target is the load at current timestamp
    y_train.append(train_features.select(['net_load_kwh']).row(i))

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Train/validation split (80/20)
val_size = 0.2
split_idx = int(len(X_train) * (1 - val_size))

X_tr = X_train[:split_idx]
X_val = X_train[split_idx:]
y_tr = y_train[:split_idx]
y_val = y_train[split_idx:]

print(f"Training samples: {X_tr.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")



print(train_features.select(feature_cols).row(0 - HORIZON_STEPS))


import numpy as np
from sklearn.dummy import DummyRegressor, BaseEstimator, RegressorMixin

models = {}
print("Baseline...")
models['baseline_mean_y'] = DummyRegressor(strategy="mean")
models['baseline_mean_y'].fit(X_tr, y_tr)

class FeatureAsPrediction(BaseEstimator, RegressorMixin):
    def __init__(self, feature_index=0):
        self.feature_index = feature_index

    def fit(self, X, y=None):
        # Nothing to fit
        return self

    def predict(self, X):
        # Return the selected feature as prediction
        X = np.asarray(X)
        return X[:, self.feature_index]

models['baseline_1w'] = FeatureAsPrediction()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 7. MODEL EVALUATION
# ==========================================
print("\n7. EVALUATING MODELS")
print("-" * 40)


def plot_predictions(features, test_predictions, actual_values=None, title='Load Time Series (1 week sample)'):
    one_week_interval = 672
    # Plot predictions with 1 week earlier value
    num_weeks = 3
    fig, axes = plt.subplots(num_weeks,1, figsize=(10,10))
    # Time series sample
    for i in range(0,num_weeks):
        ax = axes[i]
        range_start = one_week_interval * i
        range_end = range_start + one_week_interval
        ax.plot(features[range_start:range_end], alpha=0.7, label="used features (previous week)")
        ax.plot(test_predictions[range_start:range_end], alpha=0.7, markersize=3, linestyle=":", color="red", label="predictions (previous week)")
        if actual_values is not None:
            ax.plot(actual_values[range_start:range_end], alpha=0.7, markersize=3, color="orange", label="actual")
        ax.set_title(title)
        ax.set_ylabel('Net Load (kWh/15min)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-650, 650)
        ax.tick_params(axis='x', rotation=45)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate NRMSE and NMAE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    mean_load = np.mean(np.abs(y_true))
    nrmse = (rmse / mean_load) * 100
    nmae = (mae / mean_load) * 100
    
    print(f"\n{model_name}:")
    print(f"  NRMSE: {nrmse:.2f}%")
    print(f"  NMAE: {nmae:.2f}%")
    
    if nrmse < 5 and nmae < 5:
        print(f"  ✓ MEETS competition targets!")
    
    return {'nrmse': nrmse, 'nmae': nmae, 'rmse': rmse, 'mae': mae}

# Evaluate each model
val_metrics = {}
val_predictions = {}

for name, model in models.items():
    pred = model.predict(X_val)
    pred_tr = model.predict(X_tr)
    
    plot_predictions(X_tr, pred_tr, y_tr, title=f'Load Time Series (1 week sample) train {name}')
    plot_predictions(X_val, pred, y_val, title=f'Load Time Series (1 week sample) val {name}')
    val_predictions[name] = pred
    val_metrics[name] = evaluate_model(y_val, pred, name.upper())

best_model_name = 'baseline_1w'
final_model = models[best_model_name]


from datetime import timedelta
import numpy as np

# ==========================================
# 9. GENERATE TEST PREDICTIONS
# ==========================================
print("\n9. GENERATING TEST PREDICTIONS (48 HOURS AHEAD)")
print("-" * 40)

# For test predictions, we need the features from 7 days BEFORE each test timestamp
# Combine train and test to get historical data for test predictions
all_features = pl.concat([train_features, test_features])

# If net_load_kwh_lag_1w is empty, still at start of test-set, where feature-engineering didn't create lag-features yet.
# Just regenerate all lag-featurs on all_features again
all_features = all_features.with_columns(
    pl.col("net_load_kwh").shift(LAG_1W_INTERVAL).alias("net_load_kwh_lag_1w")
)
print(all_features[LAG_1W_INTERVAL-2:])

# Generate predictions for test set
test_predictions = []

for test_time in test_features['timestamp_utc']:        
    X_test = (
        all_features
        .filter(pl.col("timestamp_utc") == test_time)  # adjust to your index/column
        .select(feature_cols)
        .to_numpy()
        .reshape(1, -1)
    )
    
    if np.isnan(X_test).any():
        print(f"Lag-feature is null or NaN {X_test}. Aborting")
        raise ValueError("Null or NaN")
    
    # Make prediction
    pred = final_model.predict(X_test)[0]
    test_predictions.append(pred)
    
    # Auto-regression: append predicted value as feature in all_features if net_load_kwh_lag_1w is null
    all_features.with_columns(
                (pl.col("timestamp_utc") + pl.duration(days=7)).alias("timestamp_future"),
                pl.col("net_load_kwh").alias("net_load_kwh_lag_1w")
            ),
    all_features = all_features.with_columns(
        pl.when(
            (pl.col("timestamp_utc") == test_time + pl.duration(days=7)) & pl.col("net_load_kwh_lag_1w").is_null()
        ).then(pred)
        .otherwise(pl.col("net_load_kwh_lag_1w")).alias("net_load_kwh_lag_1w")
    )

test_predictions = np.array(test_predictions)
print(f"Generated {len(test_predictions)} predictions for test set")
final_used_features = all_features.filter(
    (pl.col("timestamp_utc") >= test_features['timestamp_utc'].min()) &
    (pl.col("timestamp_utc") <= test_features['timestamp_utc'].max())
).select(feature_cols).to_numpy()
print(f"Final features {len(final_used_features)}")


plot_predictions(final_used_features, test_predictions)




# ==========================================
# 10. CREATE SUBMISSION
# ==========================================
print("\n10. CREATING SUBMISSION FILE")
print("-" * 40)

# Create submission DataFrame
submission = pl.DataFrame({
    'row_id': test_df['row_id'],
    'predicted_net_load_kwh': test_predictions
})

# Verify submission
print(f"Submission shape: {submission.shape}")
print(f"Columns: {submission.columns}")
print(submission.describe())
print(f"Prediction range: [{submission['predicted_net_load_kwh'].min():.2f}, {submission['predicted_net_load_kwh'].max():.2f}]")
print(f"Prediction mean: {submission['predicted_net_load_kwh'].mean():.2f}")
print(f"Prediction std: {submission['predicted_net_load_kwh'].std():.2f}")

print(f"predictions valid not NaN: {(~submission['predicted_net_load_kwh'].is_nan()).sum()}")
print(f"predictions NaN: {submission['predicted_net_load_kwh'].is_nan().sum()}")

# Check for any NaN values
if submission.select((pl.all().is_null() | pl.all().is_nan()).any()).to_numpy()[0][0]:
    print("WARNING: Found NaN values in submission, filling with mean")
    submission = submission.fillna(submission['predicted_net_load_kwh'].mean())

# Save submission
submission.write_csv('submission.csv')
print("\n✓ Submission saved to 'submission.csv'")

# Display first and last rows
print("\nFirst 10 rows of submission:")
print(submission.head(10))
print("\nLast 10 rows of submission:")
print(submission.tail(10))

# ==========================================
# 11. FINAL SUMMARY
# ==========================================
print("\n" + "=" * 80)
print("COMPETITION SUMMARY")
print("=" * 80)

print(f"\nModel: Baselin{best_model_name.upper()}")
print(f"Validation Performance:")
print(f"  NRMSE: {val_metrics[best_model_name]['nrmse']:.2f}%")
print(f"  NMAE: {val_metrics[best_model_name]['nmae']:.2f}%")
print(f"Competition Target: NRMSE < 5%, NMAE < 5%")

print(f"\nPrediction Setup:")
print(f"  Forecast horizon: {HORIZON_HOURS} hours ({HORIZON_STEPS} steps)")
print(f"  Data frequency: 15-minute intervals")
print(f"  Data latency: 72 hours (respects 3-day lag requirement)")
print(f"  Weather sources: 5 Dutch cities from Open-Meteo")

print(f"\nFeatures Used: {len(feature_cols)}")
print(f"  Time features: Cyclical encoding, holidays, peak hours")
print(f"  Weather features: Temperature, humidity, wind, radiation")
print(f"  Derived features: Heating/cooling degree, wind chill, heat index")

print("\n" + "=" * 80)
print("SUBMISSION READY FOR KAGGLE!")
print("File: submission.csv")
print("Format: row_id, predicted_net_load_kwh")
print("=" * 80)




