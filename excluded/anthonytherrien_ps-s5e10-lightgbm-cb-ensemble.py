# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================
def create_features(df):
    # Copy dataframe
    df = df.copy()

    # Polynomial features
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2

    # Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])

    # Interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']

    # Risk combinations
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = (
        ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) &
        ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    ).astype(int)

    # Derived categorical indicators
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)

    # Time-based and holiday proxies
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)

    # Safety and danger scores
    df['safety_score'] = (
        df['road_signs_present'].astype(int) * 2 +
        (df['lighting'] == 'daylight').astype(int) +
        (df['weather'] == 'clear').astype(int)
    )

    df['danger_score'] = (
        (df['curvature'] > 0.6).astype(int) +
        (df['speed_limit'] >= 60).astype(int) +
        df['is_bad_weather'] +
        df['is_night'] +
        (df['num_reported_accidents'] >= 2).astype(int)
    )

    # Ratio and intensity features
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50

    return df


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================
def train_models(X, y, X_test, n_folds=5):
    # Bin target for stratification
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

    # Initialize Stratified K-Fold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Define models
    models = {
        'LightGBM': LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=7, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=7,
            l2_leaf_reg=3, random_state=42, verbose=0
        )
    }

    # Store results
    results = {}
    oof_predictions = {}
    test_predictions = {}

    # Train each model
    for name, model in models.items():
        print(f"\n{'='*60}\nTraining {name}\n{'='*60}")

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_binned), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train
            model.fit(X_train, y_train)

            # Predict
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / n_folds

            # Compute RMSE
            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            fold_scores.append(fold_rmse)
            print(f"Fold {fold}: RMSE = {fold_rmse:.6f}")

        # Overall RMSE
        oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        results[name] = {
            'oof_score': oof_rmse,
            'fold_scores': fold_scores,
            'std': np.std(fold_scores)
        }
        oof_predictions[name] = oof_preds
        test_predictions[name] = test_preds

        print(f"Overall OOF RMSE: {oof_rmse:.6f} (+/- {np.std(fold_scores):.6f})")

    return results, oof_predictions, test_predictions


# ============================================================
# ENSEMBLE FUNCTION
# ============================================================
def create_ensemble(results, oof_predictions, test_predictions, y, X, X_test):
    # Compute weights
    results_df = pd.DataFrame(results).T.sort_values('oof_score')
    weights = 1 / results_df['oof_score'].values
    weights = weights / weights.sum()

    # Initialize predictions
    ensemble_oof = np.zeros(len(X))
    ensemble_test = np.zeros(len(X_test))

    # Weighted average
    for model, weight in zip(results_df.index, weights):
        ensemble_oof += oof_predictions[model] * weight
        ensemble_test += test_predictions[model] * weight

    # Compute ensemble RMSE
    ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
    improvement = (
        (results_df['oof_score'].iloc[0] - ensemble_rmse) /
        results_df['oof_score'].iloc[0] * 100
    )

    print(f"\nEnsemble OOF RMSE: {ensemble_rmse:.6f}")
    print(f"Improvement over best single model: {improvement:.2f}%")

    return ensemble_test, ensemble_rmse, results_df


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    # Load data
    train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

    # Apply feature engineering
    train_fe = create_features(train)
    test_fe = create_features(test)

    # Prepare train/test data
    X = train_fe.drop(['id', 'accident_risk'], axis=1)
    y = train_fe['accident_risk']
    X_test = test_fe.drop(['id'], axis=1)

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Convert boolean columns
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    # Identify numeric columns for scaling
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Apply StandardScaler
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"\n✅ Applied StandardScaler to {len(numeric_cols)} numeric columns.")

    # Train models
    results, oof_predictions, test_predictions = train_models(X, y, X_test)

    # Ensemble
    ensemble_test, ensemble_rmse, results_df = create_ensemble(
        results, oof_predictions, test_predictions, y, X, X_test
    )

    # Final submission
    submission = sample_submission.copy()
    submission['accident_risk'] = ensemble_test

    # Blend with another submission (optional)
    other_submission = pd.read_csv('/kaggle/input/predicting-road-accident-risk-vault/submission.csv')
    submission['accident_risk'] = (
        submission['accident_risk'] * 0.001 +
        other_submission['accident_risk'] * 0.999
    )

    # Clip values
    submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

    # Save final output
    submission.to_csv('submission.csv', index=False)
    print("\n✅ Submission saved to 'submission.csv'")
    print(f"Best Model: {results_df.index[0]} (RMSE: {results_df['oof_score'].iloc[0]:.6f})")
    print(f"Ensemble RMSE: {ensemble_rmse:.6f}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

