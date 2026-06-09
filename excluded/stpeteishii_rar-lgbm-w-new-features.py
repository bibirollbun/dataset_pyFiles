import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
train0 = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test0 = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

class FeatureEngineer:
    """Feature engineering pipeline that prevents data leakage"""
    
    def __init__(self):
        self.categorical_columns = None
        self.fitted = False
        
    def encode_categorical_features(self, df, is_train=True):
        """Encode categorical variables consistently"""
        df_encoded = df.copy()
        
        # Binary variables - Label Encoding
        binary_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        for col in binary_columns:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(int)
        
        # Multi-class categorical - One-Hot Encoding with drop_first
        multiclass_columns = ['road_type', 'lighting', 'weather', 'time_of_day']
        
        if is_train:
            self.categorical_columns = multiclass_columns
        
        for col in multiclass_columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    def create_engineered_features(self, df):
        """Create domain-specific features"""
        df_new = df.copy()
        
        # 1. Visibility Risk Score
        lighting_risk = (
            df_new.get('lighting_night', 0) * 0.9 +
            df_new.get('lighting_dim', 0) * 0.6 +
            df_new.get('lighting_daylight', 0) * 0.1
        )
        
        weather_risk = (
            df_new.get('weather_foggy', 0) * 0.8 +
            df_new.get('weather_rainy', 0) * 0.7 +
            df_new.get('weather_clear', 0) * 0.2
        )
        
        df_new['visibility_risk'] = (lighting_risk + weather_risk) / 2
        
        # 2. Road Complexity Index
        road_complexity = (
            df_new.get('road_type_urban', 0) * 0.8 +
            df_new.get('road_type_rural', 0) * 0.5 +
            df_new.get('road_type_highway', 0) * 0.7
        )
        
        if 'curvature' in df_new.columns:
            curvature_max = df_new['curvature'].max()
            if curvature_max > 0:
                curvature_normalized = df_new['curvature'] / curvature_max
            else:
                curvature_normalized = 0
            df_new['road_complexity_index'] = road_complexity * 0.6 + curvature_normalized * 0.4
        else:
            df_new['road_complexity_index'] = road_complexity
        
        # 3. Traffic Pattern Features (vectorized)
        traffic_density = (
            df_new.get('time_of_day_morning', 0) * 0.8 +
            df_new.get('time_of_day_evening', 0) * 0.9 +
            df_new.get('time_of_day_afternoon', 0) * 0.5
        )
        
        # Vectorized holiday and school adjustments
        if 'holiday' in df_new.columns:
            traffic_density = traffic_density * np.where(df_new['holiday'] == 1, 1.3, 1.0)
        
        if 'school_season' in df_new.columns:
            traffic_density = traffic_density * np.where(df_new['school_season'] == 1, 1.2, 0.9)
        
        df_new['traffic_density_score'] = traffic_density
        
        # 4. Safety Infrastructure Score
        safety_score = (
            df_new.get('road_signs_present', 0) * 0.4 +
            df_new.get('public_road', 0) * 0.3 +
            df_new.get('lighting_daylight', 0) * 0.3
        )
        df_new['safety_infrastructure_score'] = safety_score
        
        # 5. Environmental Hazard Index
        df_new['environmental_hazard_index'] = (
            df_new['visibility_risk'] * 0.6 + 
            (1 - df_new['safety_infrastructure_score']) * 0.4
        )
        
        # 6. Risk Interaction Terms
        df_new['complexity_traffic_interaction'] = (
            df_new['road_complexity_index'] * df_new['traffic_density_score']
        )
        
        df_new['visibility_traffic_interaction'] = (
            df_new['visibility_risk'] * df_new['traffic_density_score']
        )
        
        # 7. Speed Appropriateness (vectorized)
        if 'speed_limit' in df_new.columns and 'curvature' in df_new.columns:
            conditions = [
                (df_new['curvature'] > 0.5) & (df_new['speed_limit'] > 50),
                (df_new['curvature'] < 0.2) & (df_new['speed_limit'] < 30)
            ]
            choices = [0.2, 0.8]
            df_new['speed_appropriateness'] = np.select(conditions, choices, default=0.5)
        
        return df_new
    
    def fit_transform(self, df):
        """Fit on training data and transform"""
        self.fitted = True
        df_encoded = self.encode_categorical_features(df, is_train=True)
        df_final = self.create_engineered_features(df_encoded)
        return df_final
    
    def transform(self, df):
        """Transform test data using fitted parameters"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        df_encoded = self.encode_categorical_features(df, is_train=False)
        df_final = self.create_engineered_features(df_encoded)
        return df_final


def train_lightgbm(X_train, y_train, params=None, n_boost_round=2000, early_stopping_rounds=100):
    """Train LightGBM model with validation"""
    
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.03,            # Lower to improve accuracy (0.01-0.1)
            'num_leaves': 50,                 # Increase for more complexity (31-255)
            'max_depth': 8,                   # Limit depth to prevent overfitting (5-15, -1=unlimited)
            'min_data_in_leaf': 15,           # Decrease to learn more detail (10-50)
            'feature_fraction': 0.85,         # Feature sampling (0.6-0.95)
            'bagging_fraction': 0.85,         # Data sampling (0.6-0.95)
            'bagging_freq': 5,                # Bagging frequency (1-10)
            'lambda_l1': 0.05,                # L1 regularization (0-1)
            'lambda_l2': 0.05,                # L2 regularization (0-1)
            'min_gain_to_split': 0.01,        # Minimum gain for a split (0-1)
            'verbose': -1,
            'random_state': 42,
            'extra_trees': False,             # Set to True for more randomness
            'path_smooth': 0.0,               # Tree smoothing (0-10)
        }
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print("Training LightGBM model...")
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n=== {dataset_name} METRICS ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred, {'rmse': rmse, 'mae': mae, 'r2': r2}


def get_feature_importance(model, feature_names, top_n=20):
    """Get and display feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCES ===")
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df


# ============= MAIN PIPELINE =============

# Initialize feature engineer
fe = FeatureEngineer()

# Fit on training data only
print("\n1. Engineering features on TRAINING data...")
train_processed = fe.fit_transform(train0)
print(f"Training shape: {train_processed.shape}")

# Transform test data using training parameters
print("\n2. Engineering features on TEST data...")
test_processed = fe.transform(test0)
print(f"Test shape: {test_processed.shape}")

# Prepare data for modeling
target_col = 'accident_risk'
exclude_cols = ['id']

X_train = train_processed.drop(columns=[target_col] + exclude_cols, errors='ignore')
y_train = train_processed[target_col]
X_test = test_processed.drop(columns=[target_col] + exclude_cols, errors='ignore')

print(f"\n3. Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Test data: {X_test.shape[0]} samples")

# Handle any column mismatches
missing_cols = set(X_train.columns) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(X_train.columns)

if missing_cols:
    print(f"\n   Warning: Columns in train but not test: {missing_cols}")
    for col in missing_cols:
        X_test[col] = 0

if extra_cols:
    print(f"   Warning: Columns in test but not train: {extra_cols}")
    X_test = X_test.drop(columns=list(extra_cols))

# Align column order
X_test = X_test[X_train.columns]


model = train_lightgbm(X_train, y_train)

train_pred, train_metrics = evaluate_model(model, X_train, y_train, "Training")

importance_df = get_feature_importance(model, X_train.columns)

test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_processed['id'] if 'id' in test_processed.columns else test0['id'],
    'accident_risk': test_pred
})

print(submission.head(10))

submission.to_csv('submission.csv', index=False)





