import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')



from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"\nğŸ“‹ Dataset Shape:")
print(f"   Train: {train.shape}")
print(f"   Test:  {test.shape}")

print(f"\nğŸ“Š Target Variable Statistics:")
print(train['accident_risk'].describe())

print(f"\nğŸ”� Missing Values:")
print(train.isnull().sum())

print(f"\nğŸ“ˆ Feature Types:")
print(train.dtypes.value_counts())


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(train['accident_risk'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Accident Risk Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')
axes[0].axvline(train['accident_risk'].mean(), color='red', linestyle='--', label=f'Mean: {train["accident_risk"].mean():.3f}')
axes[0].legend()

axes[1].boxplot(train['accident_risk'], vert=True)
axes[1].set_title('Accident Risk Boxplot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accident Risk')

from scipy import stats
stats.probplot(train['accident_risk'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nğŸ“Š Target Statistics:")
print(f"   Mean: {train['accident_risk'].mean():.4f}")
print(f"   Median: {train['accident_risk'].median():.4f}")
print(f"   Std: {train['accident_risk'].std():.4f}")
print(f"   Skewness: {train['accident_risk'].skew():.4f}")
print(f"   Kurtosis: {train['accident_risk'].kurtosis():.4f}")



categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                        'public_road', 'time_of_day', 'holiday', 'school_season']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    if train[col].dtype == 'bool':
        data = train.groupby(col)['accident_risk'].mean().sort_values()
    else:
        data = train.groupby(col)['accident_risk'].mean().sort_values()
    
    data.plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black', alpha=0.8)
    axes[idx].set_title(f'Avg Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Avg Accident Risk')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('categorical_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical significance testing
print(f"\nğŸ”¬ Statistical Significance Tests (ANOVA):")
from scipy.stats import f_oneway

for col in categorical_features:
    if train[col].dtype != 'bool':
        groups = [train[train[col] == val]['accident_risk'].values 
                 for val in train[col].unique()]
        f_stat, p_value = f_oneway(*groups)
        print(f"   {col:25s} - F-stat: {f_stat:8.2f}, p-value: {p_value:.2e}")



numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, col in enumerate(numerical_features):
    # Distribution
    axes[0, idx].hist(train[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[0, idx].set_xlabel(col)
    axes[0, idx].set_ylabel('Frequency')
    
    # Relationship with target
    if col in ['num_lanes', 'speed_limit']:
        grouped = train.groupby(col)['accident_risk'].agg(['mean', 'std', 'count'])
        axes[1, idx].bar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                        color='lightcoral', edgecolor='black', alpha=0.8, capsize=5)
        axes[1, idx].set_xlabel(col)
    else:
        axes[1, idx].scatter(train[col], train['accident_risk'], alpha=0.3, s=1, color='green')
        z = np.polyfit(train[col], train['accident_risk'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(train[col].min(), train[col].max(), 100)
        axes[1, idx].plot(x_line, p(x_line), "r-", linewidth=2, label='Trend')
        axes[1, idx].legend()
        axes[1, idx].set_xlabel(col)
    
    axes[1, idx].set_ylabel('Accident Risk')
    axes[1, idx].set_title(f'{col} vs Accident Risk', fontsize=12, fontweight='bold')
    axes[1, idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('numerical_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation with target
print(f"\nğŸ“ˆ Correlation with Target:")
correlations = train[numerical_features + ['accident_risk']].corr()['accident_risk'].sort_values(ascending=False)
print(correlations)



plt.figure(figsize=(10, 8))
correlation_matrix = train[numerical_features + ['accident_risk']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


print(f"\nğŸ”— Key Feature Interactions:")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Road type + Weather
interaction_data = train.groupby(['road_type', 'weather'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[0, 0], width=0.8)
axes[0, 0].set_title('Road Type Ã— Weather', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Avg Accident Risk')
axes[0, 0].legend(title='Weather')
axes[0, 0].tick_params(axis='x', rotation=45)

# Lighting + Time of Day
interaction_data = train.groupby(['lighting', 'time_of_day'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[0, 1], width=0.8)
axes[0, 1].set_title('Lighting Ã— Time of Day', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Avg Accident Risk')
axes[0, 1].legend(title='Time of Day')
axes[0, 1].tick_params(axis='x', rotation=45)

# Speed Limit + Curvature (binned)
train['curvature_bin'] = pd.cut(train['curvature'], bins=[0, 0.3, 0.6, 1.0], 
                                 labels=['Low', 'Medium', 'High'])
interaction_data = train.groupby(['speed_limit', 'curvature_bin'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[0, 2], width=0.8)
axes[0, 2].set_title('Speed Limit Ã— Curvature', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Avg Accident Risk')
axes[0, 2].legend(title='Curvature')
axes[0, 2].tick_params(axis='x', rotation=45)

# Holiday + School Season
interaction_data = train.groupby(['holiday', 'school_season'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[1, 0], width=0.8)
axes[1, 0].set_title('Holiday Ã— School Season', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Avg Accident Risk')
axes[1, 0].legend(title='School Season')
axes[1, 0].tick_params(axis='x', rotation=0)

# Num Accidents by Road Type
interaction_data = train.groupby(['num_reported_accidents', 'road_type'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='line', ax=axes[1, 1], marker='o', linewidth=2)
axes[1, 1].set_title('Reported Accidents Ã— Road Type', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Avg Accident Risk')
axes[1, 1].set_xlabel('Number of Reported Accidents')
axes[1, 1].legend(title='Road Type')
axes[1, 1].grid(alpha=0.3)

# Public Road + Road Signs
interaction_data = train.groupby(['public_road', 'road_signs_present'])['accident_risk'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[1, 2], width=0.8)
axes[1, 2].set_title('Public Road Ã— Road Signs', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Avg Accident Risk')
axes[1, 2].legend(title='Road Signs')
axes[1, 2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('feature_interactions.png', dpi=300, bbox_inches='tight')
plt.show()


def create_features(df):
    """Advanced feature engineering"""
    df = df.copy()
    
    # 1. Polynomial features for key numerical variables
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    
    # 2. Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
    
    # 3. Interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    
    # 4. Risk score combinations
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) & \
                                   ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    df['weather_lighting_risk'] = df['weather_lighting_risk'].astype(int)
    
    # 5. Categorical aggregations (target encoding will be done in CV)
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    
    # 6. Time-based features
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)  # Using holiday as proxy
    
    # 7. Safety features
    df['safety_score'] = df['road_signs_present'].astype(int) * 2 + \
                         (df['lighting'] == 'daylight').astype(int) + \
                         (df['weather'] == 'clear').astype(int)
    
    df['danger_score'] = (df['curvature'] > 0.6).astype(int) + \
                         (df['speed_limit'] >= 60).astype(int) + \
                         df['is_bad_weather'] + df['is_night'] + \
                         (df['num_reported_accidents'] >= 2).astype(int)
    
    # 8. Ratio features
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50
    
    return df

# Apply feature engineering
print("Creating features...")
train_fe = create_features(train)
test_fe = create_features(test)

print(f" Original features: {train.shape[1]}")
print(f" After feature engineering: {train_fe.shape[1]}")
print(f" New features created: {train_fe.shape[1] - train.shape[1]}")


# Prepare data
X = train_fe.drop(['id', 'accident_risk'], axis=1)
y = train_fe['accident_risk']
X_test = test_fe.drop(['id'], axis=1)

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Handle boolean columns
bool_cols = X.select_dtypes(include=['bool']).columns
X[bool_cols] = X[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

print(f"\nğŸ“Š Final feature count: {X.shape[1]}")
print(f"ğŸ“Š Feature types: {X.dtypes.value_counts().to_dict()}")

# Cross-validation setup
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Stratify based on binned target
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Model configurations
models = {
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    )
}

# Train models and collect predictions
results = {}
oof_predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
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
        
        # Score
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
        print(f"   Fold {fold}: RMSE = {fold_rmse:.6f}")
    
    # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    results[name] = {
        'oof_score': oof_rmse,
        'fold_scores': fold_scores,
        'std': np.std(fold_scores)
    }
    oof_predictions[name] = oof_preds
    test_predictions[name] = test_preds
    
    print(f"   {'â”€'*50}")
    print(f"   OOF RMSE: {oof_rmse:.6f} (+/- {np.std(fold_scores):.6f})")


# Results summary
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('oof_score')
print("\n" + results_df.to_string())

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
axes[0].barh(results_df.index, results_df['oof_score'], color='steelblue', edgecolor='black')
axes[0].set_xlabel('OOF RMSE', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(results_df['oof_score']):
    axes[0].text(v + 0.0001, i, f'{v:.6f}', va='center')

# Box plot of fold scores
fold_data = [results[model]['fold_scores'] for model in results_df.index]
axes[1].boxplot(fold_data, labels=results_df.index, vert=True)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('Cross-Validation Stability', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create ensemble
print("\n" + "="*80)
print("ğŸ�¯ CREATING ENSEMBLE")
print("="*80)

# Weighted average based on performance
weights = 1 / results_df['oof_score'].values
weights = weights / weights.sum()

print("\nğŸ“Š Ensemble Weights:")
for model, weight in zip(results_df.index, weights):
    print(f"   {model:15s}: {weight:.4f}")

# Ensemble predictions
ensemble_oof = np.zeros(len(X))
ensemble_test = np.zeros(len(X_test))

for model, weight in zip(results_df.index, weights):
    ensemble_oof += oof_predictions[model] * weight
    ensemble_test += test_predictions[model] * weight

ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"\nâœ¨ Ensemble OOF RMSE: {ensemble_rmse:.6f}")

# Comparison
improvement = (results_df['oof_score'].iloc[0] - ensemble_rmse) / results_df['oof_score'].iloc[0] * 100
print(f"ğŸ“ˆ Improvement over best single model: {improvement:.2f}%")



import pandas as pd

# Your current predictions
submission = sample_submission.copy()
submission['accident_risk'] = ensemble_test

# Load another submission to ensemble with
other_submission = pd.read_csv('/kaggle/input/s5e10-nn-stacking-baseline/test_nn_ensemble.csv')




submission['accident_risk'] = (submission['accident_risk'] * 0.01 + other_submission['accident_risk']*0.99)

# Or weighted averaging, e.g., 70% your model, 30% other model
# submission['accident_risk'] = 0.7 * submission['accident_risk'] + 0.3 * other_submission['accident_risk']

# Clip to valid range
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

# Stats
print(f"\nğŸ“Š Submission Statistics:")
print(submission['accident_risk'].describe())

# Save
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission saved to 'submission.csv'")



# Get feature importance from best model (LightGBM typically)
best_model_name = results_df.index[0]
best_model = models[best_model_name]

# Retrain on full data for feature importance
best_model.fit(X, y)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nğŸ“Š Top 20 Most Important Features ({best_model_name}):")
    print(feature_importance.head(20).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'], color='coral', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 20 Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n Final Results:")
print(f"   Best Single Model: {results_df.index[0]} (RMSE: {results_df['oof_score'].iloc[0]:.6f})")
print(f"   Ensemble Model:    RMSE: {ensemble_rmse:.6f}")
print(f"   Submission File:   submission.csv")




