# Step 1: Installations
!pip install catboost lightgbm xgboost prophet -q

# Step 2: Imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import traceback
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from prophet import Prophet


# Step 3: Class Definition
class CryptoMarketPredictor:
    """
    An optimized pipeline using an ensemble of tree-based models with rich,
    two-pass feature engineering to predict crypto market movements.
    """
    def __init__(self, top_features=100, top_X_features_to_preselect=30, use_future_lags=False):
        self.top_features = top_features
        self.top_X_features_to_preselect = top_X_features_to_preselect
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.models = {}
        self.preselected_X_n_names = None
        self.use_future_lags = use_future_lags

    def optimize_memory(self, df):
        """Optimize DataFrame memory usage."""
        print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != 'object' and col not in ['timestamp', 'ID', 'label']:
                c_min, c_max = df[col].min(), df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
        print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df

    def clean_data(self, df):
        """Clean DataFrame by handling inf, NaN values."""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].dtype.kind in 'fc': # Check for float or complex types
                df[col] = df[col].ffill().bfill()
        df.fillna(0, inplace=True)
        return df

    def create_advanced_features(self, df, top_base_features):
        """
        Engineer a rich set of time-based and interaction features.
        This is the **ENHANCED** feature creation method.
        """
        print("ğŸ› ï¸� Engineering rich time-based and interaction features...")

        # Basic price and imbalance features
        if 'ask_qty' in df.columns and 'bid_qty' in df.columns:
            df['mid_price'] = (df['ask_qty'] + df['bid_qty']) / 2
            df['spread'] = df['ask_qty'] - df['bid_qty']
            df['imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
            df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-10)
        
        if 'buy_qty' in df.columns and 'sell_qty' in df.columns:
            df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)

        # --- NEW: Rolling Statistics & Interactions based on Top Features ---
        windows = [5, 10, 20, 30]
        # Use only the top N features passed to this function for these complex calculations
        features_for_adv_calcs = [f for f in top_base_features if f in df.columns]
        
        print(f"  Creating rolling stats for top features: {features_for_adv_calcs[:5]}...")
        for feature in features_for_adv_calcs[:10]: # Limit to top 10 to manage feature explosion
            for window in windows:
                df[f'{feature}_ma_{window}'] = df[feature].rolling(window, min_periods=1).mean()
                df[f'{feature}_vol_{window}'] = df[feature].rolling(window, min_periods=1).std()
        
        print(f"  Creating interactions for top features: {features_for_adv_calcs[:5]}...")
        top_5_for_interactions = features_for_adv_calcs[:5]
        for i in range(len(top_5_for_interactions)):
            for j in range(i + 1, len(top_5_for_interactions)):
                f1, f2 = top_5_for_interactions[i], top_5_for_interactions[j]
                df[f'{f1}_{f2}_ratio'] = (df[f1] / (df[f2].abs() + 1e-10))
                df[f'{f1}_{f2}_diff'] = (df[f1] - df[f2])

        # Lag features
        lag_periods = [1, 2, 5, 10]
        cols_for_lags = [f for f in ['mid_price', 'imbalance'] if f in df.columns]
        for col in cols_for_lags:
            for lag in lag_periods:
                if self.use_future_lags: df[f'{col}_lag_{lag}'] = df[col].shift(-lag)
                else: df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        df = self.clean_data(df)
        print(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def select_features(self, X_df, y_df):
        """Select top k features using f_regression."""
        print(f"Selecting top {self.top_features} features from {X_df.shape[1]}...")
        n_features = min(self.top_features, X_df.shape[1])
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_df_clean = X_df.replace([np.inf, -np.inf], 0).fillna(0)
        y_df_clean = y_df.replace([np.inf, -np.inf], 0).fillna(0)
        if y_df_clean.nunique() <= 1:
            print("Warning: Target variable is constant. Skipping feature selection.")
            self.selected_features = X_df_clean.columns.tolist()
            return X_df_clean.values
        selector.fit(X_df_clean, y_df_clean)
        self.feature_selector = selector
        self.selected_features = X_df_clean.columns[selector.get_support()].tolist()
        print(f"Selected {len(self.selected_features)} features.")
        return X_df_clean[self.selected_features].values

    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        correlation, _ = pearsonr(y_true, y_pred) if np.std(y_true) > 0 and np.std(y_pred) > 0 else (np.nan, np.nan)
        print(f"ğŸ“Š {model_name} - MAE: {mae:.4f}, Pearson Correlation: {correlation:.4f}")
        return correlation

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        params = {'objective':'regression_l1','metric':'mae','n_estimators':1500,'learning_rate':0.03,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':1,'lambda_l1':0.1,'lambda_l2':0.1,'num_leaves':31,'verbose':-1,'n_jobs':-1,'seed':42}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        params = {'objective':'reg:squarederror','eval_metric':'mae','n_estimators':1500,'learning_rate':0.03,'tree_method':'hist','subsample':0.8,'colsample_bytree':0.8,'seed':42,'n_jobs':-1}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        return model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        params = {'objective':'MAE','eval_metric':'MAE','iterations':1500,'learning_rate':0.03,'random_seed':42,'logging_level':'Silent','l2_leaf_reg':3,'bagging_temperature':1}
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        return model

    def fit(self, train_data_raw):
        """Main training pipeline."""
        print("ğŸš€ Starting training pipeline...")
        train_data_raw['timestamp'] = pd.to_datetime(train_data_raw['timestamp'])

        print("Filtering data to the last 3 months for relevance...")
        three_months_prior = train_data_raw['timestamp'].max() - pd.DateOffset(months=3)
        train_df = train_data_raw[train_data_raw['timestamp'] >= three_months_prior].copy()
        train_df.sort_values(by='timestamp', inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        del train_data_raw; gc.collect()

        # --- Pass 1: Pre-select top 'X' features ---
        print("\n--- Pass 1: Pre-selecting top anonymous 'X' features ---")
        sample_df = train_df.sample(n=min(50000, len(train_df)), random_state=42)
        X_n_cols_raw = [c for c in sample_df.columns if c.startswith('X')]
        pre_selector = SelectKBest(score_func=f_regression, k=self.top_X_features_to_preselect)
        pre_selector.fit(sample_df[X_n_cols_raw].fillna(0), sample_df['label'].fillna(0))
        self.preselected_X_n_names = [col for col, support in zip(X_n_cols_raw, pre_selector.get_support()) if support]
        print(f"Pre-selected {len(self.preselected_X_n_names)} 'X' features.")
        del sample_df; gc.collect()

        # --- Pass 2: Create Advanced Features ---
        print("\n--- Pass 2: Engineering advanced features on pre-selected data ---")
        base_cols = ['timestamp', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']
        cols_to_use = list(dict.fromkeys(base_cols + self.preselected_X_n_names))
        cols_to_use = [c for c in cols_to_use if c in train_df.columns]
        train_df = train_df[cols_to_use]
        train_df = self.optimize_memory(train_df)
        # Pass the pre-selected 'X' names to the advanced feature creator
        train_df = self.create_advanced_features(train_df, top_base_features=self.preselected_X_n_names)
        
        feature_cols = [c for c in train_df.columns if c not in ['timestamp', 'ID', 'label']]
        X_df = train_df[feature_cols]
        y_df = train_df['label']

        # --- Final Feature Selection & Scaling ---
        X_selected_array = self.select_features(X_df, y_df)
        X_scaled = self.scaler.fit_transform(X_selected_array)
        
        # --- Time-Based Validation Split ---
        val_size = int(len(X_scaled) * 0.2)
        train_size = len(X_scaled) - val_size
        X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_val = y_df.iloc[:train_size], y_df.iloc[train_size:]
        print(f"\nTrain set size: {len(X_train)}, Validation set size: {len(X_val)}")
        del X_df, y_df, X_selected_array, X_scaled, train_df; gc.collect()

        # --- Train Models ---
        self.models['lgb'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.models['xgb'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        self.models['cat'] = self.train_catboost(X_train, y_train, X_val, y_val)

        # --- Evaluate Ensemble ---
        print("\n--- Validation Set Evaluation ---")
        val_preds = {name: model.predict(X_val) for name, model in self.models.items()}
        for name, pred in val_preds.items(): self.evaluate_model(y_val, pred, name.upper())
        self.evaluate_model(y_val, np.mean(list(val_preds.values()), axis=0), "ENSEMBLE")
        print("\nâœ… Training pipeline complete.")
        return self

    def predict(self, test_data_raw):
        """Generate predictions for test data in memory-managed chunks."""
        print("\nGenerating predictions...")
        if not self.models: raise ValueError("Models must be trained first. Call fit().")
        
        # This will be returned for final mapping
        test_ids_in_prediction_order = test_data_raw['ID'].copy()

        # Feature Engineering on the whole (sorted) test set
        base_cols = ['timestamp', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        cols_to_use = list(dict.fromkeys(base_cols + self.preselected_X_n_names))
        cols_to_use = [c for c in cols_to_use if c in test_data_raw.columns]
        X_test = test_data_raw[cols_to_use]
        X_test = self.create_advanced_features(X_test, top_base_features=self.preselected_X_n_names)
        
        # Ensure all selected features are present
        for f in self.selected_features:
            if f not in X_test.columns: X_test[f] = 0
        X_test = X_test[self.selected_features]

        # Scaling
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled)
        
        # Prediction
        preds = {name: model.predict(X_test_scaled) for name, model in self.models.items()}
        final_predictions = np.mean(list(preds.values()), axis=0)

        print(f"Generated {len(final_predictions)} predictions.")
        return final_predictions, test_ids_in_prediction_order

# --- Prophet Enhancement Function ---
def train_prophet_enhancement(df_ensemble, prophet_weight=0.085, sample_size=100000):
    """
    Train a Prophet model on a SAMPLE and then PREDICT IN CHUNKS to save memory.
    """
    print("\n--- ğŸ”® Starting Prophet Enhancement ---")
    
    # --- 1. FIT ON A SAMPLE (No change here) ---
    if len(df_ensemble) > sample_size:
        print(f"Training Prophet on a sample of {sample_size} rows...")
        df_sample = df_ensemble.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_ensemble

    prophet_df_train = pd.DataFrame()
    base_date = pd.to_datetime('2024-01-01')
    prophet_df_train['ds'] = base_date + pd.to_timedelta(df_sample['ID'] - df_sample['ID'].min(), unit='H')
    prophet_df_train['y'] = df_sample['Prediction'].values

    model = Prophet(
        growth='linear',
        changepoint_prior_scale=0.06,
        yearly_seasonality=False,  # Disabling this saves a lot of memory
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
    )
    
    print("Fitting Prophet model on the sample...")
    model.fit(prophet_df_train)
    
    # --- 2. PREDICT IN CHUNKS (This is the fix) ---
    print("Predicting for the full dataset in chunks to save memory...")
    future_df = pd.DataFrame()
    future_df['ds'] = base_date + pd.to_timedelta(df_ensemble['ID'] - df_ensemble['ID'].min(), unit='H')
    
    chunk_size = 100000  # Process 100k rows at a time
    all_forecasts = []
    
    for i in range(0, len(future_df), chunk_size):
        chunk = future_df.iloc[i:i + chunk_size]
        forecast_chunk = model.predict(chunk)
        all_forecasts.append(forecast_chunk)
        
    forecast = pd.concat(all_forecasts, ignore_index=True)
    prophet_predictions = forecast['yhat'].values
    # --- END OF FIX ---
    
    # --- 3. BLEND RESULTS (No change here) ---
    ensemble_predictions = df_ensemble['Prediction'].values
    final_predictions = (1 - prophet_weight) * ensemble_predictions + prophet_weight * prophet_predictions
    
    df_final_prophet = pd.DataFrame({'ID': df_ensemble['ID'], 'Prediction': final_predictions})
    
    print("--- Prophet Enhancement Complete ---")
    return df_final_prophet


# --- Main Execution Function ---
def main():
    """A wrapper function to run the entire pipeline and manage memory."""
    try:
        # 1. LOAD DATA
        print("Loading data...")
        train_full_raw = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
        test_full_raw = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

        def setup_dataframe(df, name):
            print(f"  Setting up {name} DataFrame...")
            # The index is the timestamp, so reset it and rename it.
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp'}, inplace=True)
        
            # --- FIX: Only create an 'ID' column if one doesn't already exist ---
            if 'ID' not in df.columns:
                print("  'ID' column not found. Creating one from the index.")
                df['ID'] = df.index
            else:
                print("  'ID' column already exists. Using original IDs.")
            # --- END FIX ---
            
            return df

        train_full_raw = setup_dataframe(train_full_raw, "train")
        test_full_raw = setup_dataframe(test_full_raw, "test")

        original_shuffled_ids = test_full_raw['ID'].copy()

        # 2. TIMESTAMP RECONSTRUCTION (The Critical Step ğŸ¤«)
        print("\nApplying timestamp reconstruction...")
        timestamp_recon_path = '/kaggle/input/the-order-of-the-test-rows-2/closest_rows.csv'
        
        # --- FIX PART 1: REMOVED THE TRY...EXCEPT BLOCK TO ENFORCE FAILURE ---
        if os.path.exists(timestamp_recon_path):
            print("  Loading reconstruction file...")
            t_recon = pd.read_csv(timestamp_recon_path, header=None).iloc[:, 0]

            reorder_map = pd.DataFrame({
                'original_pos': np.arange(len(t_recon)),
                'chrono_pos': t_recon.to_numpy()
            })

            valid_matches = reorder_map[reorder_map['chrono_pos'] != -1].copy()
            print(f"  Found {len(valid_matches)} valid matches.")

            valid_matches.sort_values('chrono_pos', inplace=True)
            sorted_indices = valid_matches['original_pos'].to_numpy()

            # --- FIX PART 2: DEFENSIVE CHECK FOR OUT-OF-BOUNDS INDICES ---
            max_index = len(test_full_raw) - 1
            initial_count = len(sorted_indices)
            sorted_indices = sorted_indices[sorted_indices <= max_index]
            final_count = len(sorted_indices)

            if initial_count != final_count:
                print(f"  Warning: Removed {initial_count - final_count} out-of-bounds indices.")
            # --- END OF FIX PART 2 ---

            test_full_raw = test_full_raw.iloc[sorted_indices].copy()
            test_full_raw.reset_index(drop=True, inplace=True)
            print(f"  Test data successfully sorted. New shape: {test_full_raw.shape}")
        else:
            # If the file doesn't exist, raise an error to stop the script.
            raise FileNotFoundError(f"CRITICAL: Timestamp reconstruction file not found at {timestamp_recon_path}. Halting execution.")

        # 3. INITIALIZE AND TRAIN MODEL
        predictor = CryptoMarketPredictor(top_features=120, top_X_features_to_preselect=30)
        predictor.fit(train_full_raw)

        # 4. PREDICT
        predictions, sorted_test_ids = predictor.predict(test_full_raw)

        # 5. ENHANCE WITH PROPHET
        initial_submission_df = pd.DataFrame({'ID': sorted_test_ids, 'Prediction': predictions})
        final_submission_df = train_prophet_enhancement(initial_submission_df, prophet_weight=0.085, sample_size=100000)

        # 6. FINALIZE SUBMISSION
        print("\n" + "="*50)
        print("PREPARING SUBMISSION")
        print("="*50)

        results_df = pd.DataFrame({'ID': final_submission_df['ID'], 'Prediction': final_submission_df['Prediction']})
        submission = pd.DataFrame({'ID': original_shuffled_ids})
        submission = submission.merge(results_df, on='ID', how='left')
        submission['Prediction'].fillna(0, inplace=True)
        
        submission.to_csv('submission.csv', index=False)
        print("Submission saved to 'submission.csv'")
        print("\nSubmission preview:")
        print(submission.head())

    except Exception as e:
        print(f"\nAn error occurred in the main pipeline: {e}")
        traceback.print_exc()

    finally:
        print("\nğŸ§¹ Cleaning up to free memory...")
        gc.collect()
        print("Cleanup complete.")

# --- Run the pipeline ---
if __name__ == "__main__":
    main()




