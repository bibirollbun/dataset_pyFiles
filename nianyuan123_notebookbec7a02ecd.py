#å¯¼å…¥åº“
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
except AttributeError:
    import sys
    sys.modules.pop('optuna.logging', None)
    sys.modules.pop('optuna', None)
    import lightgbm as lgb


# å®šä¹‰è¯„ä¼°å‡½æ•°
def rmsle(y_true, y_pred):
    """è®¡ç®—å�‡æ–¹æ ¹å¯¹æ•°è¯¯å·®ï¼ˆKaggleæŒ‡æ ‡ï¼‰"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
def calculate_all_metrics(y_true, y_pred, dataset_name=""):
    """å®Œæ•´çš„è¯„ä¼°æŒ‡æ ‡"""
    metrics = {
        'RMSLE': rmsle(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    mask = y_true > 0
    if mask.any():
        metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    if dataset_name:
        print(f"\nğŸ“Š {dataset_name} è¯„ä¼°ç»“æ�œ:")
        print("-" * 40)
        for key, value in metrics.items():
            if key in ['RMSLE', 'RMSE', 'MAE']:
                print(f"{key}: {value:.4f}")
            elif key == 'R2':
                print(f"{key}: {value:.4f}")
            elif key == 'MAPE':
                print(f"{key}: {value:.2f}%")
    return metrics


# æ•°æ�®åŠ è½½
train_path = '/kaggle/input/bike-sharing-demand/train.csv' 
test_path = '/kaggle/input/bike-sharing-demand/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# ç‰¹å¾�å·¥ç¨‹
def create_advanced_features(df):
    """
    ç‰¹å¾�å·¥ç¨‹
    å¢�åŠ ç‰¹å¾�å¤šæ ·æ€§ï¼Œæ��å�–æ›´å¤šæ—¶é—´ã€�å¤©æ°”ã€�äº¤äº’ç‰¹å¾�
    """
    df = df.copy()
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['dayofyear'] = df['datetime'].dt.dayofyear
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        df['time_of_day'] = pd.cut(df['hour'], 
                                   bins=[-1, 5, 11, 17, 21, 24],
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late_Night'])
    
    if 'temp' in df.columns:
        df['temp_category'] = pd.cut(df['temp'], 
                                     bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                     labels=['Very_Cold', 'Cold', 'Cool', 'Warm', 'Hot'])
        
        if 'atemp' in df.columns:
            df['temp_feel_diff'] = df['atemp'] - df['temp']
        
        if 'humidity' in df.columns:
            df['heat_index'] = df['temp'] * (df['humidity'] / 100) * 0.1
            
        if 'windspeed' in df.columns:
            df['windchill'] = 13.12 + 0.6215*df['temp'] - 11.37*(df['windspeed']**0.16) + 0.3965*df['temp']*(df['windspeed']**0.16)
            df['windchill'] = np.maximum(df['windchill'], df['temp'])
    
    if all(col in df.columns for col in ['temp', 'humidity', 'windspeed', 'weather']):
        df['ideal_conditions'] = ((df['temp'] > 15) & (df['temp'] < 25) & 
                                 (df['humidity'] < 70) & 
                                 (df['windspeed'] < 15) & 
                                 (df['weather'] == 1)).astype(int)
        
        df['bad_conditions'] = ((df['weather'] >= 3) | 
                               (df['windspeed'] > 25) | 
                               (df['temp'] < 0) | 
                               (df['temp'] > 35)).astype(int)
    
    return df
train = create_advanced_features(train)
test = create_advanced_features(test)


# æ•°æ�®é¢„å¤„ç�†å’Œç‰¹å¾�é€‰æ‹©ï¼ˆæ™ºèƒ½é¢„å¤„ç�†ï¼‰
def prepare_data(train_df, test_df):
    """æ•°æ�®é¢„å¤„ç�†å’Œç‰¹å¾�é€‰æ‹©"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    if 'count' in train_df.columns:
        y = train_df['count'].copy()
    elif 'registered' in train_df.columns and 'casual' in train_df.columns:
        y = train_df['registered'] + train_df['casual']
    else:
        raise ValueError("æœªæ‰¾åˆ°ç›®æ ‡å�˜é‡�")
    
    y_log = np.log1p(y)
    
    if 'count' in train_df.columns:
        Q1 = y.quantile(0.01)
        Q3 = y.quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        train_df = train_df[mask].copy()
        y = y[mask]
        y_log = y_log[mask]
    
    drop_cols = ['datetime', 'casual', 'registered', 'count']
    drop_cols = [col for col in drop_cols if col in train_df.columns]
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    
    if 'datetime' in test_df.columns:
        X_test = test_df.drop(columns=['datetime'], errors='ignore')
    else:
        X_test = test_df.copy()
    
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dummy_na=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dummy_na=True)
        
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        for col in train_cols - test_cols:
            X_test[col] = 0
        for col in test_cols - train_cols:
            X_train[col] = 0
        
        X_test = X_test[X_train.columns]
    
    if len(X_train.columns) > 50:
        xgb_simple = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        sample_size = min(2000, len(X_train))
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        
        xgb_simple.fit(X_train.iloc[sample_idx], y_log.iloc[sample_idx])
        
        importance = pd.Series(xgb_simple.feature_importances_, index=X_train.columns)
        selected_features = importance[importance > 0.001].index.tolist()
        
        if len(selected_features) > 20:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
    
    return X_train, X_test, y, y_log, train_df

X_train, X_test, y, y_log, train_clean = prepare_data(train, test)


# æ•°æ�®å�¯è§†åŒ–
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Bike Sharing Demand Analysis', fontsize=16, fontweight='bold')

axes[0, 0].hist(y, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Target Variable Distribution')
axes[0, 0].set_xlabel('Bike Count')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(y_log, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Log-transformed Target Distribution')
axes[0, 1].set_xlabel('log(Count + 1)')
axes[0, 1].set_ylabel('Frequency')

if 'hour' in train.columns:
    hourly_avg = train_clean.groupby('hour')['count'].mean() if 'count' in train_clean.columns else train_clean.groupby('hour').size()
    axes[0, 2].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
    axes[0, 2].set_title('Hourly Rental Pattern')
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Average Rentals')
    axes[0, 2].grid(True, alpha=0.3)

if 'dayofweek' in train.columns:
    weekday_avg = train_clean.groupby('dayofweek')['count'].mean() if 'count' in train_clean.columns else train_clean.groupby('dayofweek').size()
    axes[1, 0].bar(weekday_avg.index, weekday_avg.values, color=['blue' if i<5 else 'red' for i in weekday_avg.index])
    axes[1, 0].set_title('Weekly Rental Pattern')
    axes[1, 0].set_xlabel('Day of Week (0=Monday)')
    axes[1, 0].set_ylabel('Average Rentals')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

if 'temp' in train.columns and 'count' in train_clean.columns:
    axes[1, 1].scatter(train_clean['temp'], train_clean['count'], alpha=0.3, s=10, color='green')
    axes[1, 1].set_title('Temperature vs Rentals')
    axes[1, 1].set_xlabel('Temperature (Â°C)')
    axes[1, 1].set_ylabel('Rentals')
    
    z = np.polyfit(train_clean['temp'], train_clean['count'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(train_clean['temp'], p(train_clean['temp']), "r--", alpha=0.8, linewidth=2)

if len(X_train.columns) > 5:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 10:
        top_cols = numeric_cols[:10]
        corr_matrix = X_train[top_cols].corr()
        
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_title('Feature Correlation Heatmap')
        axes[1, 2].set_xticks(range(len(top_cols)))
        axes[1, 2].set_yticks(range(len(top_cols)))
        axes[1, 2].set_xticklabels([col[:10] for col in top_cols], rotation=45, ha='right')
        axes[1, 2].set_yticklabels([col[:10] for col in top_cols])
        plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.show()


# æ•°æ�®åˆ’åˆ†å’Œäº¤å�‰éªŒè¯�è®¾ç½®
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_log, test_size=0.2, random_state=42, shuffle=True
)

y_tr_orig = np.expm1(y_tr)
y_val_orig = np.expm1(y_val)


# è®­ç»ƒXGBoostæ¨¡å�‹ï¼ˆä¼˜åŒ–è¶…å�‚æ•°ï¼‰
optimized_xgb = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.015,
    max_depth=9,
    min_child_weight=4,
    subsample=0.75,
    colsample_bytree=0.75,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse'
)
xgb_model = optimized_xgb.fit(X_tr, y_tr)


# è®­ç»ƒLightGBMæ¨¡å�‹
if lgb is not None:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)


# Stackingé›†æˆ�ï¼ˆæ¨¡å�‹é›†æˆ�ï¼‰
base_models = [
    ('xgb', xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )),
    ('rf', RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )),
    ('ridge', Ridge(alpha=0.5, random_state=42))
]

if lgb is not None:
    base_models.insert(1, ('lgb', lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )))

meta_model = Ridge(alpha=1.0, random_state=42)

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    n_jobs=-1,
    passthrough=False
)

stacking_model.fit(X_tr, y_tr)



# æ¨¡å�‹è¯„ä¼°å’Œæ¯”è¾ƒ
def evaluate_model_performance(model, X_train, y_train_orig, X_val, y_val_orig, model_name):
    """è¯„ä¼°å�•ä¸ªæ¨¡å�‹æ€§èƒ½"""
    y_pred_train_log = model.predict(X_train)
    y_pred_val_log = model.predict(X_val)
    
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_val = np.expm1(y_pred_val_log)
    
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_val = np.maximum(y_pred_val, 0)
    
    train_metrics = calculate_all_metrics(y_train_orig, y_pred_train)
    val_metrics = calculate_all_metrics(y_val_orig, y_pred_val)
    
    print(f"\nğŸ“Š {model_name} æ€§èƒ½:")
    print("-" * 40)
    print(f"è®­ç»ƒé›† RMSLE: {train_metrics['RMSLE']:.4f}")
    print(f"éªŒè¯�é›† RMSLE: {val_metrics['RMSLE']:.4f}")
    print(f"éªŒè¯�é›† RÂ²: {val_metrics['R2']:.4f}")
    
    return train_metrics, val_metrics, y_pred_val

xgb_train_metrics, xgb_val_metrics, xgb_pred_val = evaluate_model_performance(
    xgb_model, X_tr, y_tr_orig, X_val, y_val_orig, "XGBoost"
)

if lgb is not None:
    lgb_train_metrics, lgb_val_metrics, lgb_pred_val = evaluate_model_performance(
        lgb_model, X_tr, y_tr_orig, X_val, y_val_orig, "LightGBM"
    )


# è¯„ä¼°Stackingæ¨¡å�‹
stacking_pred_val_log = stacking_model.predict(X_val)
stacking_pred_val = np.expm1(stacking_pred_val_log)
stacking_pred_val = np.maximum(stacking_pred_val, 0)

stacking_val_metrics = calculate_all_metrics(y_val_orig, stacking_pred_val, "Stackingé›†æˆ�æ¨¡å�‹")

# %%
# ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��
xgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = xgb_importance.head(15)
plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('Top 15 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# é¢„æµ‹å�¯è§†åŒ–
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(y_val_orig, xgb_pred_val, alpha=0.3, s=10, color='blue')
axes[0, 0].plot([y_val_orig.min(), y_val_orig.max()], 
                [y_val_orig.min(), y_val_orig.max()], 
                'r--', lw=2)
axes[0, 0].set_xlabel('Actual Rentals')
axes[0, 0].set_ylabel('Predicted Rentals (XGBoost)')
axes[0, 0].set_title(f'XGBoost: Actual vs Predicted (RÂ² = {xgb_val_metrics.get("R2", 0):.4f})')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_aspect('equal', 'box')

axes[0, 1].scatter(y_val_orig, stacking_pred_val, alpha=0.3, s=10, color='green')
axes[0, 1].plot([y_val_orig.min(), y_val_orig.max()], 
                [y_val_orig.min(), y_val_orig.max()], 
                'r--', lw=2)
axes[0, 1].set_xlabel('Actual Rentals')
axes[0, 1].set_ylabel('Predicted Rentals (Stacking)')
axes[0, 1].set_title(f'Stacking: Actual vs Predicted (RÂ² = {stacking_val_metrics.get("R2", 0):.4f})')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_aspect('equal', 'box')

residuals = y_val_orig - xgb_pred_val
axes[1, 0].scatter(xgb_pred_val, residuals, alpha=0.3, s=10, color='purple')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Rentals')
axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[1, 0].set_title('Residual Plot (XGBoost)')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Error Distribution')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()



# ç”Ÿæˆ�æœ€ç»ˆé¢„æµ‹
print("ç”Ÿæˆ�æœ€ç»ˆé¢„æµ‹...")
# ä½¿ç”¨Stackingæ¨¡å�‹è¿›è¡Œé¢„æµ‹
test_predictions_log = stacking_model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)
test_predictions = np.maximum(test_predictions, 0)
xgb_test_pred_log = xgb_model.predict(X_test)
xgb_test_pred = np.expm1(xgb_test_pred_log)
xgb_test_pred = np.maximum(xgb_test_pred, 0)
if lgb is not None:
    lgb_test_pred_log = lgb_model.predict(X_test)
    lgb_test_pred = np.expm1(lgb_test_pred_log)
    lgb_test_pred = np.maximum(lgb_test_pred, 0)
    # ä½¿ç”¨åŠ æ�ƒå¹³å�‡é›†æˆ�
    weights = {'xgb': 0.4, 'lgb': 0.3, 'stacking': 0.3}
    ensemble_predictions = (weights['xgb'] * xgb_test_pred + 
                           weights['lgb'] * lgb_test_pred + 
                           weights['stacking'] * test_predictions)
else:
    weights = {'xgb': 0.5, 'stacking': 0.5}
    ensemble_predictions = (weights['xgb'] * xgb_test_pred + 
                           weights['stacking'] * test_predictions)

# ç¡®ä¿�é��è´Ÿ
ensemble_predictions = np.maximum(ensemble_predictions, 0)

# ä¿�å­˜æœ€ç»ˆé¢„æµ‹
final_predictions = ensemble_predictions

print(f"âœ… æœ€ç»ˆé¢„æµ‹å®Œæˆ�")
print(f"é¢„æµ‹æ ·æœ¬æ•°: {len(final_predictions)}")
print(f"å¹³å�‡é¢„æµ‹å€¼: {final_predictions.mean():.2f}")
print(f"é¢„æµ‹èŒƒå›´: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
# æ˜¾ç¤ºå‰�5ä¸ªé¢„æµ‹ç»“æ�œ
print("\nå‰�5ä¸ªé¢„æµ‹ç»“æ�œ:")
for i in range(min(5, len(final_predictions))):
    print(f"æ ·æœ¬ {i+1}: {final_predictions[i]:.2f}")


# ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
if 'datetime' in test.columns:
    submission = pd.DataFrame({
        'datetime': test['datetime'],
        'count': ensemble_predictions
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
# %%
# ä¿�å­˜æ¨¡å�‹å’Œç»“æ�œ
import pickle
import json

models_to_save = {
    'xgb_model': xgb_model,
    'stacking_model': stacking_model
}

if lgb is not None:
    models_to_save['lgb_model'] = lgb_model

for name, model in models_to_save.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

results = {
    'xgb_metrics': xgb_val_metrics,
    'stacking_metrics': stacking_val_metrics,
    'ensemble_weights': weights
}

if lgb is not None:
    results['lgb_metrics'] = lgb_val_metrics

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

