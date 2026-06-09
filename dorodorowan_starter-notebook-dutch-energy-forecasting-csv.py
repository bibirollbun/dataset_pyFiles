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


#Dutch Energy Forecasting - Complete Solution with Correct Submission Format


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model imports - 仅保留LightGBM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error  
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

print("=" * 80)
print("DUTCH ENERGY FORECASTING - FINAL VERSION (LightGBM Only)")
print("=" * 80)

# ==========================================
# 1. DATA LOADING
# ==========================================
print("\n1. LOADING DATA")
print("-" * 40)

train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test: {test_df.shape}")
print(f"Test columns: {test_df.columns.tolist()}")

# Check if test has row_id
has_row_id = 'row_id' in test_df.columns
print(f"Test has row_id: {has_row_id}")

# Standardize columns
if 'Datetime' in train_df.columns:
    train_df = train_df.rename(columns={'Datetime': 'timestamp_utc', 'Actual Net': 'net_load_kwh'})
    test_df = test_df.rename(columns={'Datetime': 'timestamp_utc', 'Actual Net': 'net_load_kwh'})

train_df['timestamp_utc'] = pd.to_datetime(train_df['timestamp_utc'])
test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'])

train_df = train_df.sort_values('timestamp_utc').reset_index(drop=True)
test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)

print(f"Train period: {train_df['timestamp_utc'].min()} to {train_df['timestamp_utc'].max()}")
print(f"Test period: {test_df['timestamp_utc'].min()} to {test_df['timestamp_utc'].max()}")

# ==========================================
# 2. WEATHER DATA
# ==========================================
print("\n2. FETCHING WEATHER DATA")
print("-" * 40)

def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Open-Meteo"""
    locations = [
        (52.3676, 4.9041, 'Amsterdam'),
        (51.9244, 4.4777, 'Rotterdam'),
        (52.0907, 5.1214, 'Utrecht')
    ]
    
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'pressure_msl', 'cloud_cover', 'wind_speed_10m',
        'direct_radiation', 'diffuse_radiation'
    ]
    
    all_weather = []
    
    for lat, lon, city in locations[:1]:  # Use just Amsterdam to speed up
        print(f"  Fetching {city}...")
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(weather_features),
            'timezone': 'UTC'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            weather_df = pd.DataFrame(data['hourly'])
            weather_df['timestamp_utc'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.drop('time', axis=1)
            weather_df = weather_df.set_index('timestamp_utc')
            
            # Resample to 15-min
            weather_df = weather_df.resample('15min').interpolate(method='linear')
            
            all_weather.append(weather_df)
            
        except:
            print(f"    Using synthetic weather")
            dates = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
            n = len(dates)
            np.random.seed(42)
            
            weather_df = pd.DataFrame(index=dates)
            weather_df.index.name = 'timestamp_utc'
            
            # Simple synthetic patterns
            hour_of_year = (np.arange(n) % (365 * 24 * 4)) / 4
            weather_df['temperature_2m'] = 10 + 10*np.cos(2*np.pi*hour_of_year/(365 * 24)) + np.random.normal(0, 2, n)
            weather_df['relative_humidity_2m'] = 70 + np.random.normal(0, 10, n)
            weather_df['wind_speed_10m'] = np.abs(5 + np.random.normal(0, 2, n))
            weather_df['pressure_msl'] = 1013 + np.random.normal(0, 5, n)
            weather_df['cloud_cover'] = np.clip(50 + np.random.normal(0, 20, n), 0, 100)
            
            hour = (np.arange(n) % (24 * 4)) / 4
            weather_df['direct_radiation'] = np.maximum(0,
                500 * np.sin(np.maximum(0, (hour-6)*np.pi/12)) * (hour >= 6) * (hour <= 18)
                + np.random.normal(0, 30, n))
            
            all_weather.append(weather_df)
    
    return pd.concat(all_weather, axis=1) if all_weather else pd.DataFrame()

# Fetch weather
start = train_df['timestamp_utc'].min().strftime('%Y-%m-%d')
end = test_df['timestamp_utc'].max().strftime('%Y-%m-%d')
weather_df = fetch_weather_data(start, end)
print(f"  Weather shape: {weather_df.shape}")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("\n3. FEATURE ENGINEERING")
print("-" * 40)

def create_features(df, weather_df, include_lags=False):
    """Create features"""
    features = df.copy()
    features = features.set_index('timestamp_utc')
    
    # Merge weather
    if not weather_df.empty:
        weather_aligned = weather_df.reindex(features.index, method='nearest')
        features = pd.concat([features, weather_aligned], axis=1)
    
    # Time features
    features['hour'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month
    features['day_of_year'] = features.index.dayofyear
    features['week_of_year'] = features.index.isocalendar().week.astype(int)
    
    # Binary features
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
    features['is_peak'] = ((features['hour'] >= 17) & (features['hour'] <= 20)).astype(int)
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Weather features
    if 'temperature_2m' in features.columns:
        features['temp_squared'] = features['temperature_2m'] ** 2
        features['heating_degree'] = np.maximum(0, 18 - features['temperature_2m'])
        features['cooling_degree'] = np.maximum(0, features['temperature_2m'] - 22)
    
    # Lag features ONLY for training
    if include_lags and 'net_load_kwh' in features.columns:
        # Only lags > 72 hours (288 steps)
        for lag_hours in [73, 96, 168]:
            lag_steps = lag_hours * 4
            if lag_steps < len(features):
                features[f'lag_{lag_hours}h'] = features['net_load_kwh'].shift(lag_steps)
    
    # Fourier features
    for period in [24, 24 * 7]:
        for k in range(1, 3):
            features[f'fourier_{period}_{k}_sin'] = np.sin(2*np.pi*k*features.index.hour/period)
            features[f'fourier_{period}_{k}_cos'] = np.cos(2*np.pi*k*features.index.hour/period)
    
    return features

# Create features
train_features = create_features(train_df, weather_df, include_lags=True)
test_features = create_features(test_df, weather_df, include_lags=False)

# Fill NaN
train_features = train_features.ffill().bfill().fillna(0)
test_features = test_features.ffill().bfill().fillna(0)

# Get common features (exclude target and lag features)
common_features = [col for col in train_features.columns 
                  if col != 'net_load_kwh' and col != 'row_id' and col in test_features.columns]

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")
print(f"Common features: {len(common_features)}")

# ==========================================
# 4. PREPARE DATA
# ==========================================
print("\n4. PREPARING DATA")
print("-" * 40)

HORIZON_HOURS = 48
HORIZON_STEPS = HORIZON_HOURS * 4  # 192 steps

# Prepare training data
X_train_list = []
y_train_list = []

for i in range(len(train_features) - HORIZON_STEPS):
    X_train_list.append(train_features[common_features].iloc[i].values)
    y_train_list.append(train_features['net_load_kwh'].iloc[i:i+HORIZON_STEPS].values)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

# Validation split
val_size = 0.2
split_idx = int(len(X_train) * (1 - val_size))
X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

# ==========================================
# 5. TRAIN LightGBM MODEL
# ==========================================
print("\n5. TRAINING LightGBM MODEL")
print("-" * 40)

# Scale
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_tr_scaled = scaler_X.fit_transform(X_tr)
X_val_scaled = scaler_X.transform(X_val)
y_tr_scaled = scaler_y.fit_transform(y_tr)

# Train LightGBM with MultiOutputRegressor
print("Training LightGBM MultiOutputRegressor...")
model = MultiOutputRegressor(
    lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=50,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    n_jobs=1
)
model.fit(X_tr_scaled, y_tr_scaled)

# Validate
pred_val = scaler_y.inverse_transform(model.predict(X_val_scaled))
val_rmse = np.sqrt(mean_squared_error(y_val.flatten(), pred_val.flatten()))
val_mae = mean_absolute_error(y_val.flatten(), pred_val.flatten())
val_nrmse = (val_rmse / np.mean(np.abs(y_val.flatten()))) * 100
val_nmae = (val_mae / np.mean(np.abs(y_val.flatten()))) * 100

print(f"Validation - NRMSE: {val_nrmse:.2f}%, NMAE: {val_nmae:.2f}%")

# Retrain on full data
print("Retraining on full data...")
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
model.fit(X_train_scaled, y_train_scaled)

# # ==========================================
# # 6. GENERATE PREDICTIONS
# # ==========================================
# print("\n6. GENERATING PREDICTIONS")
# print("-" * 40)

# # Method 1: If we predict for every timestamp in test
# if len(test_features) <= HORIZON_STEPS:
#     # Single prediction window
#     X_test = test_features[common_features].iloc[0].values.reshape(1, -1)
#     X_test_scaled = scaler_X.transform(X_test)
#     test_pred_scaled = model.predict(X_test_scaled)
#     test_predictions = scaler_y.inverse_transform(test_pred_scaled)
#     all_predictions = test_predictions[0][:len(test_features)]
# else:
#     # Multiple prediction windows
#     all_predictions = []
    
#     # Sliding window predictions
#     for i in range(len(test_features)):
#         if i % HORIZON_STEPS == 0 and i + HORIZON_STEPS <= len(test_features):
#             # Make a new prediction
#             X_test = test_features[common_features].iloc[i].values.reshape(1, -1)
#             X_test_scaled = scaler_X.transform(X_test)
#             test_pred_scaled = model.predict(X_test_scaled)
#             test_pred = scaler_y.inverse_transform(test_pred_scaled)
#             all_predictions.extend(test_pred[0])
#         elif len(all_predictions) < len(test_features):
#             # Use last prediction or average
#             if all_predictions:
#                 all_predictions.append(all_predictions[-1])
#             else:
#                 all_predictions.append(-176.0)  # Default baseline
    
#     # Ensure we have predictions for all test samples
#     all_predictions = all_predictions[:len(test_features)]
    
#     # Fill any remaining with baseline
#     while len(all_predictions) < len(test_features):
#         all_predictions.append(-176.0)

# print(f"Generated {len(all_predictions)} predictions")

# # ==========================================
# # 7. CREATE SUBMISSION WITH ROW_ID
# # ==========================================
# print("\n7. CREATING SUBMISSION")
# print("-" * 40)

# # Check if test has row_id column
# if 'row_id' in test_df.columns:
#     print("Using existing row_id from test data")
#     submission = pd.DataFrame({
#         'row_id': test_df['row_id'].values,
#         'predicted_net_load_kwh': all_predictions[:len(test_df)]
#     })
# else:
#     print("Creating sequential row_id")
#     submission = pd.DataFrame({
#         'row_id': range(len(test_df)),
#         'predicted_net_load_kwh': all_predictions[:len(test_df)]
#     })

# # Ensure no NaN values
# submission['predicted_net_load_kwh'] = submission['predicted_net_load_kwh'].fillna(-176.0)

# # Save submission
# submission.to_csv('submission.csv', index=False)

# print(f"✓ Submission saved")
# print(f"  Shape: {submission.shape}")
# print(f"  Columns: {submission.columns.tolist()}")
# print(f"  Predictions range: [{submission['predicted_net_load_kwh'].min():.2f}, {submission['predicted_net_load_kwh'].max():.2f}]")

# print("\nFirst 10 rows:")
# print(submission.head(10))

# print("\nSubmission statistics:")
# print(submission['predicted_net_load_kwh'].describe())

# # ==========================================
# # 8. VERIFY SUBMISSION FORMAT
# # ==========================================
# print("\n8. VERIFYING SUBMISSION")
# print("-" * 40)

# # Check required columns
# required_cols = ['row_id', 'predicted_net_load_kwh']
# has_required = all(col in submission.columns for col in required_cols)
# print(f"Has required columns: {has_required}")

# # Check row_id is unique
# is_unique = submission['row_id'].nunique() == len(submission)
# print(f"row_id is unique: {is_unique}")

# # Check no missing values
# has_nulls = submission.isnull().any().any()
# print(f"Has null values: {has_nulls}")

# # Check data types
# print(f"row_id dtype: {submission['row_id'].dtype}")
# print(f"predicted_net_load_kwh dtype: {submission['predicted_net_load_kwh'].dtype}")

# if has_required and is_unique and not has_nulls:
#     print("\n✅ SUBMISSION FORMAT IS VALID!")
# else:
#     print("\n⚠️ SUBMISSION NEEDS FIXING")

# print("\n" + "=" * 80)
# print("COMPLETE! 'submission.csv' is ready for Kaggle upload")
# print("=" * 80)



# ==========================================
# 5.5 FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n5.5 ANALYZING WEATHER FEATURE IMPORTANCE")
print("-" * 40)

def analyze_feature_importance(model, X_val_scaled, y_val, feature_names, scaler_y):
    """Comprehensive feature importance analysis using multiple methods"""
    
    print("Performing comprehensive feature importance analysis...")
    
    # Method 1: Permutation Importance (模型无关，最可靠)
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
    
    # 使用原始量纲的y_val进行计算
    y_val_original = scaler_y.inverse_transform(y_val) if hasattr(scaler_y, 'inverse_transform') else y_val
    
    # 修复：定义自定义评分函数来处理多输出问题
    def custom_scorer(estimator, X, y):
        """自定义评分函数处理多输出预测"""
        y_pred = estimator.predict(X)
        # 展平所有维度为单一向量
        return -mean_squared_error(y.ravel(), y_pred.ravel())
    
    # 计算排列重要性 - 使用自定义评分函数
    perm_result = permutation_importance(
        model, 
        X_val_scaled, 
        y_val_original,  # 使用完整的目标值
        scoring=custom_scorer,  # 使用自定义评分函数
        n_repeats=5,  # 减少重复次数以加快计算
        random_state=42,
        n_jobs=-1
    )
    perm_importances = perm_result.importances_mean
    
    # Method 2: LightGBM内置特征重要性
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        lgb_importances = np.zeros(len(feature_names))
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                lgb_importances += estimator.feature_importances_
        lgb_importances /= len(model.estimators_)
    else:
        lgb_importances = np.zeros(len(feature_names))
    
    # Method 3: SHAP值分析（更精确但计算成本高）
    shap_importances = np.zeros(len(feature_names))
    try:
        import shap
        print("  Computing SHAP values (this may take a while)...")
        
        # 使用小样本计算SHAP值
        X_val_sample = X_val_scaled[:100]  # 使用前100个样本加快计算
        
        # 创建SHAP解释器
        if hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model.estimators_[0])
            shap_values = explainer.shap_values(X_val_sample)
            
            if isinstance(shap_values, list):
                # 多分类情况
                shap_importances = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                shap_importances = np.mean(np.abs(shap_values), axis=0)
    except ImportError:
        print("  SHAP not available, skipping SHAP analysis")
    except Exception as e:
        print(f"  SHAP computation failed: {e}")
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': perm_importances,
        'lgb_importance': lgb_importances,
        'shap_importance': shap_importances
    })
    
    # 标准化重要性分数 (0-1)
    for col in ['permutation_importance', 'lgb_importance', 'shap_importance']:
        if importance_df[col].max() > 0:
            importance_df[f'{col}_normalized'] = importance_df[col] / importance_df[col].max()
        else:
            importance_df[f'{col}_normalized'] = 0
    
    # 计算综合重要性得分
    importance_df['combined_importance'] = (
        importance_df['permutation_importance_normalized'] * 0.5 +
        importance_df['lgb_importance_normalized'] * 0.3 +
        importance_df['shap_importance_normalized'] * 0.2
    )
    
    return importance_df

def plot_feature_importance(importance_df, top_n=15):
    """可视化特征重要性结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 排列重要性（最可靠的方法）
    top_perm = importance_df.nlargest(top_n, 'permutation_importance')
    axes[0, 0].barh(range(len(top_perm)), top_perm['permutation_importance'])
    axes[0, 0].set_yticks(range(len(top_perm)))
    axes[0, 0].set_yticklabels(top_perm['feature'])
    axes[0, 0].set_title('Permutation Importance (Most Reliable)')
    axes[0, 0].set_xlabel('Importance Score')
    
    # 2. LightGBM内置重要性
    top_lgb = importance_df.nlargest(top_n, 'lgb_importance')
    axes[0, 1].barh(range(len(top_lgb)), top_lgb['lgb_importance'])
    axes[0, 1].set_yticks(range(len(top_lgb)))
    axes[0, 1].set_yticklabels(top_lgb['feature'])
    axes[0, 1].set_title('LightGBM Built-in Importance')
    axes[0, 1].set_xlabel('Importance Score')
    
    # 3. 综合重要性
    top_combined = importance_df.nlargest(top_n, 'combined_importance')
    axes[1, 0].barh(range(len(top_combined)), top_combined['combined_importance'])
    axes[1, 0].set_yticks(range(len(top_combined)))
    axes[1, 0].set_yticklabels(top_combined['feature'])
    axes[1, 0].set_title('Combined Importance Score')
    axes[1, 0].set_xlabel('Combined Importance')
    
    # 4. 气象要素筛选（重点关注天气相关特征）
    weather_features = [f for f in importance_df['feature'] if any(keyword in f for keyword in 
                                                                  ['temp', 'degree', 'humidity', 'precipitation', 
                                                                   'pressure', 'cloud', 'wind', 'radiation'])]
    weather_importance = importance_df[importance_df['feature'].isin(weather_features)]
    top_weather = weather_importance.nlargest(min(top_n, len(weather_importance)), 'combined_importance')
    
    if len(top_weather) > 0:
        axes[1, 1].barh(range(len(top_weather)), top_weather['combined_importance'])
        axes[1, 1].set_yticks(range(len(top_weather)))
        axes[1, 1].set_yticklabels(top_weather['feature'])
        axes[1, 1].set_title('Weather Features Only')
        axes[1, 1].set_xlabel('Combined Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_weather

def perform_ablation_study(model, X_val_scaled, y_val, feature_names, importance_df, scaler_y, top_k=5):
    """通过消融实验验证重要特征的实际影响"""
    print("\nPerforming feature ablation study...")
    
    # 获取最重要的k个特征
    top_features = importance_df.nlargest(top_k, 'combined_importance')['feature'].tolist()
    
    # 基准性能（使用所有特征）
    y_val_original = scaler_y.inverse_transform(y_val) if hasattr(scaler_y, 'inverse_transform') else y_val
    baseline_pred = scaler_y.inverse_transform(model.predict(X_val_scaled))
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original.flatten(), baseline_pred.flatten()))
    
    ablation_results = []
    
    for feature in top_features:
        # 创建不包含该特征的验证集
        feature_idx = feature_names.index(feature)
        X_val_ablated = np.delete(X_val_scaled, feature_idx, axis=1)
        
        # 重新训练一个不包含该特征的模型（简化版本）
        from sklearn.base import clone
        try:
            # 使用更少的树来加快计算
            ablated_model = clone(model)
            if hasattr(ablated_model, 'estimators_'):
                for est in ablated_model.estimators_:
                    if hasattr(est, 'set_params'):
                        est.set_params(n_estimators=50)  # 减少树的数量
            
            ablated_model.fit(X_val_ablated, y_val)
            
            # 预测并计算性能
            pred_ablated = scaler_y.inverse_transform(ablated_model.predict(X_val_ablated))
            ablated_rmse = np.sqrt(mean_squared_error(y_val_original.flatten(), pred_ablated.flatten()))
            
            performance_drop = ((ablated_rmse - baseline_rmse) / baseline_rmse) * 100
            
            ablation_results.append({
                'feature': feature,
                'baseline_rmse': baseline_rmse,
                'ablated_rmse': ablated_rmse,
                'performance_drop_pct': performance_drop
            })
            
            print(f"  Without {feature}: RMSE = {ablated_rmse:.4f} (+{performance_drop:.2f}%)")
            
        except Exception as e:
            print(f"  Ablation study failed for {feature}: {e}")
            continue
    
    return pd.DataFrame(ablation_results)

# 执行特征重要性分析
feature_names = common_features

print("Starting feature importance analysis...")
importance_df = analyze_feature_importance(model, X_val_scaled, y_val, feature_names, scaler_y)

# 显示最重要的特征
print("\nTop 10 Most Important Features:")
print("=" * 50)
top_10 = importance_df.nlargest(10, 'combined_importance')[['feature', 'combined_importance', 'permutation_importance']]
for i, row in top_10.iterrows():
    print(f"{i+1:2d}. {row['feature']:30s} | Combined: {row['combined_importance']:.4f} | Permutation: {row['permutation_importance']:.4f}")

# 可视化结果
top_weather_features = plot_feature_importance(importance_df)

# 执行消融实验
ablation_results = perform_ablation_study(model, X_val_scaled, y_val, feature_names, importance_df, scaler_y)

# 保存重要性结果
importance_df.to_csv('feature_importance_results.csv', index=False)
ablation_results.to_csv('ablation_study_results.csv', index=False)

print("\n✓ Feature importance analysis completed!")
print(f"  Top weather feature: {top_weather_features.iloc[0]['feature'] if len(top_weather_features) > 0 else 'N/A'}")
print(f"  Results saved to feature_importance_results.csv and ablation_study_results.csv")

# ==========================================
# 5.6 OPTIONAL: FEATURE SELECTION BASED ON IMPORTANCE
# ==========================================
print("\n5.6 OPTIONAL FEATURE SELECTION")
print("-" * 40)

def select_features_by_importance(importance_df, threshold=0.01):
    """基于重要性分数选择特征"""
    # 选择重要性超过阈值的特征
    selected_features = importance_df[importance_df['combined_importance'] >= threshold]['feature'].tolist()
    
    print(f"Selected {len(selected_features)} features with importance >= {threshold}:")
    for i, feature in enumerate(selected_features[:10]):  # 只显示前10个
        importance = importance_df[importance_df['feature'] == feature]['combined_importance'].iloc[0]
        print(f"  {i+1:2d}. {feature} ({importance:.4f})")
    
    return selected_features

# 可以选择性地使用特征选择结果重新训练模型
use_feature_selection = False  # 设为True来启用特征选择

if use_feature_selection:
    selected_features = select_features_by_importance(importance_df, threshold=0.02)
    
    # 更新common_features只包含选中的特征
    common_features = [f for f in common_features if f in selected_features]
    print(f"Updated common_features: {len(common_features)} features")
    
    # 重新准备数据（可选）
    # 注意：这需要重新运行数据准备和训练步骤
    # 在实际应用中，您可能想要比较特征选择前后的性能


# ==========================================
# 6. GENERATE PREDICTIONS
# ==========================================
print("\n6. GENERATING PREDICTIONS")
print("-" * 40)

# Method 1: If we predict for every timestamp in test
if len(test_features) <= HORIZON_STEPS:
    # Single prediction window
    X_test = test_features[common_features].iloc[0].values.reshape(1, -1)
    X_test_scaled = scaler_X.transform(X_test)
    test_pred_scaled = model.predict(X_test_scaled)
    test_predictions = scaler_y.inverse_transform(test_pred_scaled)
    all_predictions = test_predictions[0][:len(test_features)]
else:
    # Multiple prediction windows
    all_predictions = []
    
    # Sliding window predictions
    for i in range(len(test_features)):
        if i % HORIZON_STEPS == 0 and i + HORIZON_STEPS <= len(test_features):
            # Make a new prediction
            X_test = test_features[common_features].iloc[i].values.reshape(1, -1)
            X_test_scaled = scaler_X.transform(X_test)
            test_pred_scaled = model.predict(X_test_scaled)
            test_pred = scaler_y.inverse_transform(test_pred_scaled)
            all_predictions.extend(test_pred[0])
        elif len(all_predictions) < len(test_features):
            # Use last prediction or average
            if all_predictions:
                all_predictions.append(all_predictions[-1])
            else:
                all_predictions.append(-176.0)  # Default baseline
    
    # Ensure we have predictions for all test samples
    all_predictions = all_predictions[:len(test_features)]
    
    # Fill any remaining with baseline
    while len(all_predictions) < len(test_features):
        all_predictions.append(-176.0)

print(f"Generated {len(all_predictions)} predictions")

# ==========================================
# 7. CREATE SUBMISSION WITH ROW_ID
# ==========================================
print("\n7. CREATING SUBMISSION")
print("-" * 40)

# Check if test has row_id column
if 'row_id' in test_df.columns:
    print("Using existing row_id from test data")
    submission = pd.DataFrame({
        'row_id': test_df['row_id'].values,
        'predicted_net_load_kwh': all_predictions[:len(test_df)]
    })
else:
    print("Creating sequential row_id")
    submission = pd.DataFrame({
        'row_id': range(len(test_df)),
        'predicted_net_load_kwh': all_predictions[:len(test_df)]
    })

# Ensure no NaN values
submission['predicted_net_load_kwh'] = submission['predicted_net_load_kwh'].fillna(-176.0)

# Save submission
submission.to_csv('submission.csv', index=False)

print(f"✓ Submission saved")
print(f"  Shape: {submission.shape}")
print(f"  Columns: {submission.columns.tolist()}")
print(f"  Predictions range: [{submission['predicted_net_load_kwh'].min():.2f}, {submission['predicted_net_load_kwh'].max():.2f}]")

print("\nFirst 10 rows:")
print(submission.head(10))

print("\nSubmission statistics:")
print(submission['predicted_net_load_kwh'].describe())

# ==========================================
# 8. VERIFY SUBMISSION FORMAT
# ==========================================
print("\n8. VERIFYING SUBMISSION")
print("-" * 40)

# Check required columns
required_cols = ['row_id', 'predicted_net_load_kwh']
has_required = all(col in submission.columns for col in required_cols)
print(f"Has required columns: {has_required}")

# Check row_id is unique
is_unique = submission['row_id'].nunique() == len(submission)
print(f"row_id is unique: {is_unique}")

# Check no missing values
has_nulls = submission.isnull().any().any()
print(f"Has null values: {has_nulls}")

# Check data types
print(f"row_id dtype: {submission['row_id'].dtype}")
print(f"predicted_net_load_kwh dtype: {submission['predicted_net_load_kwh'].dtype}")

if has_required and is_unique and not has_nulls:
    print("\n✅ SUBMISSION FORMAT IS VALID!")
else:
    print("\n⚠️ SUBMISSION NEEDS FIXING")

print("\n" + "=" * 80)
print("COMPLETE! 'submission.csv' is ready for Kaggle upload")
print("=" * 80)
# 




