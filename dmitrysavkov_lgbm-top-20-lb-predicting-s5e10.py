from pathlib import Path
import sys
import warnings
import pickle

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path.cwd().resolve().parent 
print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

RAW_DATA_DIR = PROJECT_ROOT / 'input' / 'playground-series-s5e10'
SUBMISSIONS_DIR = PROJECT_ROOT / 'working'

TARGET_COL = "accident_risk"
GLOBAL_SEED = 42
N_JOBS = 8
N_SEEDS = 10
warnings.filterwarnings("ignore", category=UserWarning)





train_raw = pd.read_csv(RAW_DATA_DIR / "train.csv", index_col="id")
test_raw = pd.read_csv(RAW_DATA_DIR / "test.csv", index_col="id")
orig = pd.read_csv(PROJECT_ROOT / 'input' / "simulated-roads-accident-data" / "synthetic_road_accidents_100k.csv")

y_full = train_raw[TARGET_COL].copy()
X_raw = train_raw.drop(columns=[TARGET_COL])


class SelectiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """Lightweight version with only the most important features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Core risk score from generation strategy
        X['base_risk_score'] = (
            0.3 * X['curvature'] + 
            0.2 * (X['lighting'] == 'night').astype(int) + 
            0.1 * (X['weather'] != 'clear').astype(int) + 
            0.2 * (X['speed_limit'] >= 60).astype(int) + 
            0.1 * (X['num_reported_accidents'] > 2).astype(int)
        )
        
        # Top interactions
        X['curvature_x_speed'] = X['curvature'] * X['speed_limit']
        X['speed_per_lane'] = X['speed_limit'] / (X['num_lanes'] + 1)
        X['traffic_density'] = X['num_reported_accidents'] / (X['num_lanes'] + 1)
        
        # Road type flags
        X['is_highway'] = (X['road_type'] == 'highway').astype(int)
        X['highway_high_speed'] = X['is_highway'] * (X['speed_limit'] >= 60).astype(int)
        
        # Infrastructure safety
        X['safety_score'] = X['road_signs_present'] * X['num_lanes'] - X['curvature']
        X['public_road_density'] = X['public_road'] * X['traffic_density']
        
        # Visibility
        visibility_map = {'daylight': 1.0, 'dusk': 0.6, 'dawn': 0.6, 'night': 0.2}
        X['visibility_score'] = X['lighting'].map(visibility_map).fillna(0.5)
        
        weather_map = {'clear': 1.0, 'rain': 0.6, 'fog': 0.3, 'snow': 0.4}
        X['weather_visibility'] = X['weather'].map(weather_map).fillna(0.7)
        X['total_visibility'] = X['visibility_score'] * X['weather_visibility']
        
        # Temporal
        X['is_rush_hour'] = X['time_of_day'].isin(['morning', 'evening']).astype(int)
        X['holiday_rush'] = X['holiday'] * X['is_rush_hour']
        X['school_traffic'] = X['school_season'] * X['num_reported_accidents']
        
        return X
    
def change_obj_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

class DigitExtractorTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts digits from specified columns by converting to string and slicing.
    
    Parameters:
    -----------
    columns : list of str
        List of column names to extract digits from.
    num_digits : int, default=4
        Number of digits to extract.
    start_pos : int, default=2
        Starting position in the string (0-based index).
    fill_value : int, default=0
        Value to fill NaNs with.
    """
    
    def __init__(self, columns, num_digits=4, start_pos=2, fill_value=0):
        self.columns = columns
        self.num_digits = num_digits
        self.start_pos = start_pos
        self.fill_value = fill_value
        self.new_feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit method - computes new feature names."""
        self.new_feature_names_ = []
        for col in self.columns:
            for position in range(self.num_digits):
                self.new_feature_names_.append(f'{col}_digit{position}')
        return self
    
    def transform(self, X):
        """Transform method - extracts digits and adds as new columns."""
        X_transformed = X.copy()
        for col in self.columns:
            temp_col = f'{col}_temp'
            X_transformed[temp_col] = X_transformed[col].astype('string')
            for position in range(self.num_digits):
                new_col = f'{col}_digit{position}'
                X_transformed[new_col] = X_transformed[temp_col].str[self.start_pos + position].astype('Int8')
                X_transformed[new_col].fillna(self.fill_value, inplace=True)
            X_transformed.drop(columns=[temp_col], inplace=True)
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Returns feature names including new digit columns."""
        if input_features is None:
            raise ValueError("input_features must be provided")
        return np.array(list(input_features) + self.new_feature_names_)


generator = SelectiveFeatureEngineer()

X_eng = generator.fit_transform(X_raw)
X_test_eng = generator.transform(test_raw)

assert X_eng.index.equals(y_full.index)
assert X_test_eng.index.equals(test_raw.index)


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    digit_extractor = DigitExtractorTransformer(columns=["curvature"], num_digits=4, start_pos=2, fill_value=0)
    X_digits = digit_extractor.fit_transform(X_eng)
    X_test_digits = digit_extractor.transform(X_test_eng)

    assert X_digits.index.equals(y_full.index)
    assert X_test_digits.index.equals(test_raw.index)

    X_digits.head()


ext_columns = ["curvature", "speed_limit", "weather", "road_type", "lighting", "num_reported_accidents"]
stats = ['mean', 'std']

X_ext = X_digits.copy()
X_test_ext = X_test_digits.copy()

for group_col in ext_columns:
    grouped_values = orig.groupby(group_col)[TARGET_COL].agg(stats)
    for stat in stats:

        if group_col == 'curvature':
            X_array = X_ext[group_col].round(2)
            X_test_array = X_test_ext[group_col].round(2)
        else:
            X_array = X_ext[group_col]
            X_test_array = X_test_ext[group_col]

        new_col_name = f'{group_col}_ext_{stat}'
        values = grouped_values[stat]
        X_ext[new_col_name] = X_array.map(values).fillna(values.mean()).astype('float32')
        X_test_ext[new_col_name] = X_test_array.map(values).fillna(values.mean()).astype('float32')

assert X_ext.shape[1] == X_test_ext.shape[1]


X_ext = change_obj_to_cat(X_ext)
X_test_ext = change_obj_to_cat(X_test_ext)


def scale_target_on_value(y, *, value=1.0, factor=1.5):
    y_adj = y.copy()
    mask = y_adj.eq(value)
    y_adj.loc[mask] = y_adj.loc[mask] / factor
    return y_adj, mask.sum()

y_adjusted, adjusted_count = scale_target_on_value(y_full, value=1.0, factor=1.5)
adjusted_count


best_params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.9590314837620746, 'importance_type': 'split', 'learning_rate': 0.05714030628285888, 'max_depth': 9, 'min_child_samples': 12, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 504, 'n_jobs': 8, 'num_leaves': 31, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.9799757873178931, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'metric': 'rmse', 'verbosity': -1}
model = LGBMRegressor(**best_params)


def run_cross_validation(model, X, y, *, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    records = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = clone(model)
        fold_model.set_params(random_state=GLOBAL_SEED * fold)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = fold_model.predict(X.iloc[val_idx])
        records.append({
            "fold": fold,
            "rmse": np.sqrt(mean_squared_error(y.iloc[val_idx], preds)),
            "r2": r2_score(y.iloc[val_idx], preds)
        })
    metrics = pd.DataFrame(records)
    metrics.loc["mean"] = {"fold": "mean", "rmse": metrics["rmse"].mean(), "r2": metrics["r2"].mean()}
    metrics.loc["std"] = {"fold": "std", "rmse": metrics["rmse"].std(), "r2": metrics["r2"].std()}
    return metrics

cv_metrics = run_cross_validation(model, X_ext, y_adjusted)
cv_metrics


def multi_seed_predict(model, X_train, y_train, X_test, *, n_seeds=N_SEEDS):
    all_preds = np.zeros((X_test.shape[0], n_seeds))
    seeds = []
    for idx in range(n_seeds):
        seed = GLOBAL_SEED * (idx + 1)
        seeds.append(seed)
        seeded_model = clone(model)
        seeded_model.set_params(random_state=seed)
        seeded_model.fit(X_train, y_train)
        all_preds[:, idx] = seeded_model.predict(X_test)
    return all_preds, seeds

seed_preds, seed_values = multi_seed_predict(
    model, X_ext, y_adjusted, X_test_ext, n_seeds=N_SEEDS
)
ensemble_predictions = seed_preds.mean(axis=1)
ensemble_summary = pd.DataFrame({
    "seed": seed_values,
    "prediction_mean": seed_preds.mean(axis=0),
    "prediction_std": seed_preds.std(axis=0)
})
ensemble_summary


submission_name = "submission.csv"
sample_submission = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
sample_submission[TARGET_COL] = ensemble_predictions
output_path = SUBMISSIONS_DIR / submission_name
sample_submission.to_csv(output_path, index=False)
sample_submission.head()

