import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import holidays
import warnings
warnings.filterwarnings(action="ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def custom_mape(y_true, y_pred):
    """Calculate MAPE with handling for edge cases"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class SalesPredictor:
    def __init__(self):
        self.country_means = None
        self.store_means = None
        self.model = None
        
    def create_features(self, df):
        df = df.copy()
        
        # Basic date features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Advanced date features
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Holiday features for each country
        holiday_dicts = {
            'Canada': holidays.CA(),
            'Norway': holidays.NO(),
            'Finland': holidays.FI(),
            'Italy': holidays.IT(),
            'Singapore': holidays.SG(),
            'Kenya': holidays.KE()
        }
        
        df['is_holiday'] = df.apply(
            lambda x: x['date'] in holiday_dicts.get(x['country'], []), axis=1
        ).astype(int)
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        return df
        
    def add_lag_features(self, df, target_col='num_sold'):
        df = df.copy()
        
        # Fill any missing values in categorical columns
        categorical_cols = ['country', 'store', 'product']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
            
        if target_col in df.columns:
            # Calculate country-level statistics
            if self.country_means is None:
                self.country_means = df.groupby('country')[target_col].agg(['mean', 'std']).fillna(0).to_dict('index')
                self.store_means = df.groupby(['country', 'store'])[target_col].mean().fillna(0).to_dict()
            
            # Add country and store level features
            df['country_mean'] = df['country'].map(lambda x: self.country_means.get(x, {'mean': 0})['mean'])
            df['country_std'] = df['country'].map(lambda x: self.country_means.get(x, {'std': 0})['std'])
            df['store_mean'] = df.apply(lambda x: self.store_means.get((x['country'], x['store']), 0), axis=1)
            
            # Calculate rolling statistics by group
            for group in ['country', 'store', 'product']:
                grouped = df.groupby(group)[target_col]
                for window in [7, 14, 30]:
                    df[f'{group}_rolling_mean_{window}d'] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).mean())
                    df[f'{group}_rolling_std_{window}d'] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).std())
        
        # Fill any remaining NaN values with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
        
    def train_and_predict(self, train_df, test_df):
        # Create features
        print("Creating features...")
        train_df = self.create_features(train_df)
        test_df = self.create_features(test_df)
        
        # Add lag features
        print("Adding lag features...")
        train_df = self.add_lag_features(train_df)
        test_df = self.add_lag_features(test_df)
        
        # Encode categorical variables
        categorical_cols = ['country', 'store', 'product']
        train_df = pd.get_dummies(train_df, columns=categorical_cols)
        test_df = pd.get_dummies(test_df, columns=categorical_cols)
        
        # Align features
        test_df = test_df.reindex(columns=train_df.columns.drop(['num_sold']), fill_value=0)
        
        # Define features
        feature_cols = [col for col in train_df.columns 
                       if col not in ['date', 'num_sold', 'id']]
        
        # Initialize model with optimized parameters
        self.model = lgb.LGBMRegressor(
            objective='regression',
            metric='mape',
            boosting_type='dart',
            n_estimators=3000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df), 1):
            print(f"\nFold {fold}")
            
            X_train = train_df.iloc[train_idx][feature_cols]
            y_train = train_df.iloc[train_idx]['num_sold']
            X_val = train_df.iloc[val_idx][feature_cols]
            y_val = train_df.iloc[val_idx]['num_sold']
            
            # Fill any remaining NaN values
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            y_train = y_train.fillna(y_train.mean())
            y_val = y_val.fillna(y_val.mean())
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            
            val_pred = self.model.predict(X_val)
            val_pred = np.maximum(val_pred, 0)  # Ensure non-negative predictions
            
            mape = custom_mape(y_val, val_pred)
            mape_scores.append(mape)
            print(f"MAPE: {mape:.4f}")
        
        print(f"\nAverage MAPE: {np.mean(mape_scores):.4f}")
        
        # Retrain on full dataset
        print("\nTraining final model...")
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['num_sold'].fillna(train_df['num_sold'].mean())
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        X_test = test_df[feature_cols].fillna(0)
        predictions = self.model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative predictions
        
        # Create submission file
        submission = pd.DataFrame({
            'id': test_df['id'],
            'num_sold': predictions.round(2)  # Round to 2 decimal places
        })
        
        return submission


def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
    
    # Initialize predictor
    predictor = SalesPredictor()
    
    # Train model and generate predictions
    submission = predictor.train_and_predict(train_df, test_df)
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created!")
    
    return predictor, submission


if __name__ == "__main__":
    predictor, submission = main()




