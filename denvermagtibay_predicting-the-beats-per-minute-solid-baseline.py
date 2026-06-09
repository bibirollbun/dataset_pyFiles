import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# For modeling
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

print("="*50)
print("DATASET SHAPES")
print("="*50)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

print("\n" + "="*50)
print("FIRST 5 ROWS OF TRAINING DATA")
print("="*50)
print(train_df.head())

print("\n" + "="*50)
print("DATA TYPES")
print("="*50)
print(train_df.dtypes)

print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(train_df.describe())

print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
print("Training set missing values:")
print(train_df.isnull().sum().sum())
print("\nTest set missing values:")
print(test_df.isnull().sum().sum())

print("\n" + "="*50)
print("TARGET VARIABLE (BeatsPerMinute) ANALYSIS")
print("="*50)
print(f"Mean: {train_df['BeatsPerMinute'].mean():.2f}")
print(f"Median: {train_df['BeatsPerMinute'].median():.2f}")
print(f"Std: {train_df['BeatsPerMinute'].std():.2f}")
print(f"Min: {train_df['BeatsPerMinute'].min():.2f}")
print(f"Max: {train_df['BeatsPerMinute'].max():.2f}")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(train_df['BeatsPerMinute'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Beats Per Minute')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of BeatsPerMinute')
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(train_df['BeatsPerMinute'])
axes[1].set_ylabel('Beats Per Minute')
axes[1].set_title('Boxplot of BeatsPerMinute')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check for duplicate rows
print("\n" + "="*50)
print("DUPLICATE CHECK")
print("="*50)
duplicates = train_df.duplicated().sum()
print(f"Number of duplicate rows in training set: {duplicates}")

# Feature columns (excluding ID and target)
feature_cols = [col for col in train_df.columns if col not in ['ID', 'BeatsPerMinute']]
print("\n" + "="*50)
print(f"NUMBER OF FEATURES: {len(feature_cols)}")
print("="*50)
print("Feature columns:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")


# Step 2: Exploratory Data Analysis and Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Assuming data is already loaded from Step 1
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')

# Remove 'id' column as it's not a feature
feature_cols = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]

print("="*60)
print("CORRELATION ANALYSIS WITH TARGET")
print("="*60)

# Calculate correlations with target
correlations = train_df[feature_cols + ['BeatsPerMinute']].corr()['BeatsPerMinute'].sort_values(ascending=False)
print(correlations)

# Visualize correlations
plt.figure(figsize=(10, 6))
correlations[:-1].plot(kind='barh')
plt.xlabel('Correlation with BeatsPerMinute')
plt.title('Feature Correlations with Target Variable')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FEATURE DISTRIBUTIONS AND STATISTICS")
print("="*60)

# Create a comprehensive feature statistics table
feature_stats = pd.DataFrame()
for col in feature_cols:
    feature_stats.loc[col, 'mean'] = train_df[col].mean()
    feature_stats.loc[col, 'std'] = train_df[col].std()
    feature_stats.loc[col, 'skew'] = skew(train_df[col])
    feature_stats.loc[col, 'kurtosis'] = kurtosis(train_df[col])
    feature_stats.loc[col, 'min'] = train_df[col].min()
    feature_stats.loc[col, 'max'] = train_df[col].max()
    feature_stats.loc[col, 'corr_with_target'] = train_df[col].corr(train_df['BeatsPerMinute'])

print(feature_stats.round(4))

# Visualize feature distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    axes[idx].hist(train_df[col], bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{col}', fontsize=10)
    axes[idx].set_xlabel('')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    """Create new features based on domain knowledge and interactions"""
    df_new = df.copy()
    
    # Rhythm and Energy interactions
    df_new['Rhythm_Energy'] = df_new['RhythmScore'] * df_new['Energy']
    df_new['Rhythm_Energy_Ratio'] = df_new['RhythmScore'] / (df_new['Energy'] + 0.001)
    
    # Audio characteristics
    df_new['Loudness_Energy'] = df_new['AudioLoudness'] * df_new['Energy']
    df_new['Loudness_Scaled'] = df_new['AudioLoudness'] / df_new['AudioLoudness'].std()
    
    # Mood and acoustic features
    df_new['Mood_Acoustic'] = df_new['MoodScore'] * df_new['AcousticQuality']
    df_new['Mood_Energy'] = df_new['MoodScore'] * df_new['Energy']
    
    # Instrumental vs Vocal balance
    df_new['Instrumental_Vocal_Ratio'] = df_new['InstrumentalScore'] / (df_new['VocalContent'] + 0.001)
    df_new['Total_Sound_Content'] = df_new['InstrumentalScore'] + df_new['VocalContent']
    
    # Duration features
    df_new['Duration_Minutes'] = df_new['TrackDurationMs'] / 60000
    df_new['Duration_Category'] = pd.cut(df_new['TrackDurationMs'], 
                                          bins=[0, 180000, 240000, 300000, 500000],
                                          labels=[0, 1, 2, 3]).astype(int)
    
    # Live performance interaction
    df_new['Live_Energy'] = df_new['LivePerformanceLikelihood'] * df_new['Energy']
    df_new['Live_Mood'] = df_new['LivePerformanceLikelihood'] * df_new['MoodScore']
    
    # Polynomial features for most correlated features
    df_new['Energy_Squared'] = df_new['Energy'] ** 2
    df_new['RhythmScore_Squared'] = df_new['RhythmScore'] ** 2
    df_new['Energy_Cubed'] = df_new['Energy'] ** 3
    
    # Log transformations for skewed features
    df_new['Log_AcousticQuality'] = np.log1p(df_new['AcousticQuality'])
    df_new['Log_InstrumentalScore'] = np.log1p(df_new['InstrumentalScore'])
    
    # Binning for loudness
    df_new['Loudness_Bins'] = pd.cut(df_new['AudioLoudness'], 
                                      bins=[-30, -15, -10, -5, 0],
                                      labels=[0, 1, 2, 3]).astype(int)
    
    return df_new

# Apply feature engineering
train_fe = create_features(train_df)
test_fe = create_features(test_df)

# Get new feature columns
new_features = [col for col in train_fe.columns if col not in train_df.columns]
print(f"Created {len(new_features)} new features:")
for feat in new_features:
    print(f"  - {feat}")

print("\n" + "="*60)
print("CORRELATION OF NEW FEATURES WITH TARGET")
print("="*60)

# Check correlation of new features with target
new_feature_corrs = {}
for feat in new_features:
    new_feature_corrs[feat] = train_fe[feat].corr(train_fe['BeatsPerMinute'])

new_feature_corrs_df = pd.DataFrame.from_dict(new_feature_corrs, orient='index', columns=['Correlation'])
new_feature_corrs_df = new_feature_corrs_df.sort_values('Correlation', ascending=False)
print(new_feature_corrs_df)

# Visualize top new features correlation
plt.figure(figsize=(10, 6))
new_feature_corrs_df.head(10).plot(kind='barh')
plt.xlabel('Correlation with BeatsPerMinute')
plt.title('Top 10 New Features Correlation with Target')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("OUTLIER DETECTION")
print("="*60)

# Check for outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Check outliers in target variable
outliers_count, lower, upper = detect_outliers_iqr(train_fe, 'BeatsPerMinute')
print(f"BeatsPerMinute outliers: {outliers_count} ({outliers_count/len(train_fe)*100:.2f}%)")
print(f"  Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")

# Check outliers in key features
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    outliers_count, _, _ = detect_outliers_iqr(train_fe, col)
    print(f"{col} outliers: {outliers_count} ({outliers_count/len(train_fe)*100:.2f}%)")

# Save engineered datasets
train_fe.to_csv('train_engineered.csv', index=False)
test_fe.to_csv('test_engineered.csv', index=False)
print("\n" + "="*60)
print("ENGINEERED DATASETS SAVED")
print("="*60)
print(f"Training set shape: {train_fe.shape}")
print(f"Test set shape: {test_fe.shape}")
print(f"Total features (including new): {len([col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']])}")


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Load engineered datasets
train_fe = pd.read_csv('train_engineered.csv')
test_fe = pd.read_csv('test_engineered.csv')

# Prepare features and target
feature_cols = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
X = train_fe[feature_cols]
y = train_fe['BeatsPerMinute']
X_test = test_fe[feature_cols]

print("="*60)
print("DATA PREPARATION")
print("="*60)
print(f"Training features shape: {X.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Target shape: {y.shape}")

# Feature Scaling
scaler = RobustScaler()  # RobustScaler is better for outliers
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("CROSS-VALIDATION SETUP")
print("="*60)

# Setup cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
print(f"Using {n_folds}-fold cross-validation")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X, y, cv, model_name="Model"):
    """Evaluate model using cross-validation"""
    scores = []
    fold_predictions = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_predictions[val_idx] = y_pred
        
        score = rmse(y_val, y_pred)
        scores.append(score)
        print(f"  Fold {fold}: RMSE = {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  Mean RMSE: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return mean_score, std_score, fold_predictions

print("\n" + "="*60)
print("BASELINE MODELS EVALUATION")
print("="*60)

# Store results
results = {}

# 1. Simple Linear Models
print("\n--- Linear Models ---")

print("\nRidge Regression:")
ridge = Ridge(alpha=1.0, random_state=42)
ridge_score, ridge_std, ridge_preds = evaluate_model(ridge, X_scaled, y, kf, "Ridge")
results['Ridge'] = {'score': ridge_score, 'std': ridge_std}

print("\nLasso Regression:")
lasso = Lasso(alpha=0.01, random_state=42)
lasso_score, lasso_std, lasso_preds = evaluate_model(lasso, X_scaled, y, kf, "Lasso")
results['Lasso'] = {'score': lasso_score, 'std': lasso_std}

print("\nElasticNet:")
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
elastic_score, elastic_std, elastic_preds = evaluate_model(elastic, X_scaled, y, kf, "ElasticNet")
results['ElasticNet'] = {'score': elastic_score, 'std': elastic_std}

print("\nHuber Regressor (robust to outliers):")
huber = HuberRegressor(epsilon=1.35, alpha=0.01)
huber_score, huber_std, huber_preds = evaluate_model(huber, X_scaled, y, kf, "Huber")
results['Huber'] = {'score': huber_score, 'std': huber_std}


# 2. Tree-based Models
print("\n--- Tree-based Models ---")

print("\nRandom Forest:")
rf = RandomForestRegressor(n_estimators=100, max_depth=20, 
                          min_samples_split=10, min_samples_leaf=5,
                          random_state=42, n_jobs=-1)
rf_score, rf_std, rf_preds = evaluate_model(rf, X.values, y, kf, "RandomForest")
results['RandomForest'] = {'score': rf_score, 'std': rf_std}

print("\nExtra Trees:")
et = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                        min_samples_split=10, min_samples_leaf=5,
                        random_state=42, n_jobs=-1)
et_score, et_std, et_preds = evaluate_model(et, X.values, y, kf, "ExtraTrees")
results['ExtraTrees'] = {'score': et_score, 'std': et_std}


!pip install xgboost lightgbm catboost


# 3. Gradient Boosting Models
print("\n--- Gradient Boosting Models ---")

print("\nXGBoost:")
xgb_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_score, xgb_std, xgb_preds = evaluate_model(xgb_model, X.values, y, kf, "XGBoost")
results['XGBoost'] = {'score': xgb_score, 'std': xgb_std}

print("\nLightGBM:")
lgb_params = {
    'n_estimators': 300,
    'max_depth': -1,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_score, lgb_std, lgb_preds = evaluate_model(lgb_model, X.values, y, kf, "LightGBM")
results['LightGBM'] = {'score': lgb_score, 'std': lgb_std}

print("\nCatBoost:")
cb_params = {
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.05,
    'random_state': 42,
    'verbose': False
}
cb_model = cb.CatBoostRegressor(**cb_params)
cb_score, cb_std, cb_preds = evaluate_model(cb_model, X.values, y, kf, "CatBoost")
results['CatBoost'] = {'score': cb_score, 'std': cb_std}


# 4. KNN
print("\n--- K-Nearest Neighbors ---")
knn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
knn_score, knn_std, knn_preds = evaluate_model(knn, X_scaled, y, kf, "KNN")
results['KNN'] = {'score': knn_score, 'std': knn_std}

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

# Create results dataframe
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('score')
print("\nModels ranked by RMSE (lower is better):")
print(results_df)

# Visualize results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
models = results_df.index
scores = results_df['score'].values
stds = results_df['std'].values

x_pos = np.arange(len(models))
ax.bar(x_pos, scores, yerr=stds, alpha=0.7, capsize=5)
ax.set_xlabel('Models')
ax.set_ylabel('RMSE')
ax.set_title('Model Performance Comparison (5-Fold CV)')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, (score, std) in enumerate(zip(scores, stds)):
    ax.text(i, score + std + 0.2, f'{score:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("BEST BASELINE MODEL")
print("="*60)
best_model = results_df.index[0]
best_score = results_df.iloc[0]['score']
best_std = results_df.iloc[0]['std']
print(f"Best model: {best_model}")
print(f"RMSE: {best_score:.4f} (+/- {best_std:.4f})")

# Save baseline predictions for ensemble later
baseline_predictions = pd.DataFrame({
    'Ridge': ridge_preds,
    'Lasso': lasso_preds,
    'ElasticNet': elastic_preds,
    'Huber': huber_preds,
    'RandomForest': rf_preds,
    'ExtraTrees': et_preds,
    'XGBoost': xgb_preds,
    'LightGBM': lgb_preds,
    'CatBoost': cb_preds,
    'KNN': knn_preds,
    'Target': y
})
baseline_predictions.to_csv('baseline_predictions.csv', index=False)
print("\nBaseline predictions saved for ensemble modeling.")


# Step 4: OPTIMIZED Hyperparameter Tuning - Runs Much Faster

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, HuberRegressor
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_fe = pd.read_csv('train_engineered.csv')
test_fe = pd.read_csv('test_engineered.csv')

# Prepare features
feature_cols = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
X = train_fe[feature_cols]
y = train_fe['BeatsPerMinute']
X_test = test_fe[feature_cols]

# Scaling for linear models
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Setup - Using 3-fold CV for speed
kf = KFold(n_splits=3, shuffle=True, random_state=42)
rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))

print("="*60)
print("FAST HYPERPARAMETER TUNING")
print("="*60)
print("Using 3-fold CV and reduced parameter grids for speed")

# Store optimized models
optimized_models = {}

# 1. CatBoost - Focused parameter grid
print("\n[1/4] Tuning CatBoost...")
cb_param_grid = {
    'iterations': [300, 500],  # Just 2 options
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'l2_leaf_reg': [1, 3]
}

cb_model = cb.CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)
cb_random = RandomizedSearchCV(
    cb_model, 
    cb_param_grid, 
    n_iter=8,  # Only test 8 combinations
    cv=kf,
    scoring=rmse_scorer,
    random_state=42,
    n_jobs=1,  # CatBoost handles parallelism internally
    verbose=1
)

cb_random.fit(X, y)
print(f"Best CatBoost params: {cb_random.best_params_}")
print(f"Best CatBoost CV score: {-cb_random.best_score_:.4f}")
optimized_models['CatBoost'] = cb_random.best_estimator_

# 2. LightGBM - Focused parameter grid
print("\n[2/4] Tuning LightGBM...")
lgb_param_grid = {
    'n_estimators': [300, 500],
    'num_leaves': [25, 31],
    'learning_rate': [0.05, 0.1],
    'feature_fraction': [0.8],
    'bagging_fraction': [0.8],
    'reg_alpha': [0, 0.1]
}

lgb_model = lgb.LGBMRegressor(
    random_state=42, 
    verbose=-1, 
    n_jobs=-1,
    force_col_wise=True  # Faster for many features
)
lgb_random = RandomizedSearchCV(
    lgb_model,
    lgb_param_grid,
    n_iter=8,  # Only 8 combinations
    cv=kf,
    scoring=rmse_scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

lgb_random.fit(X, y)
print(f"Best LightGBM params: {lgb_random.best_params_}")
print(f"Best LightGBM CV score: {-lgb_random.best_score_:.4f}")
optimized_models['LightGBM'] = lgb_random.best_estimator_

# 3. Ridge - Quick grid search
print("\n[3/4] Tuning Ridge...")
ridge_param_grid = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]  # Reduced options
}

ridge_model = Ridge(random_state=42)
ridge_grid = GridSearchCV(
    ridge_model,
    ridge_param_grid,
    cv=kf,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1
)

ridge_grid.fit(X_scaled, y)
print(f"Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
print(f"Best Ridge CV score: {-ridge_grid.best_score_:.4f}")
optimized_models['Ridge'] = ridge_grid.best_estimator_

# 4. Huber - Quick grid search
print("\n[4/4] Tuning Huber...")
huber_param_grid = {
    'epsilon': [1.35, 1.5],
    'alpha': [0.001, 0.01]
}

huber_model = HuberRegressor(max_iter=200)
huber_grid = GridSearchCV(
    huber_model,
    huber_param_grid,
    cv=kf,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1
)

huber_grid.fit(X_scaled, y)
print(f"Best Huber params: {huber_grid.best_params_}")
print(f"Best Huber CV score: {-huber_grid.best_score_:.4f}")
optimized_models['Huber'] = huber_grid.best_estimator_

print("\n" + "="*60)
print("CREATING ENSEMBLE PREDICTIONS")
print("="*60)

# Train optimized models on full data
predictions = {}

print("\nTraining optimized models on full data...")
# CatBoost and LightGBM
for name in ['CatBoost', 'LightGBM']:
    print(f"Training {name}...")
    model = optimized_models[name]
    model.fit(X, y)
    predictions[name] = model.predict(X_test)

# Ridge and Huber (scaled data)
for name in ['Ridge', 'Huber']:
    print(f"Training {name}...")
    model = optimized_models[name]
    model.fit(X_scaled, y)
    predictions[name] = model.predict(X_test_scaled)

# Simple weighted average (skip stacking for speed)
print("\n" + "="*60)
print("WEIGHTED ENSEMBLE")
print("="*60)

weights = {
    'CatBoost': 0.40,
    'Ridge': 0.25,
    'LightGBM': 0.25,
    'Huber': 0.10
}

weighted_pred = np.zeros(len(X_test))
for name, weight in weights.items():
    weighted_pred += predictions[name] * weight

print(f"Ensemble weights: {weights}")

print("\n" + "="*60)
print("CREATING SUBMISSIONS")
print("="*60)

# Create submissions
submissions = {
    'ensemble': weighted_pred,
    'catboost': predictions['CatBoost'],
    'lightgbm': predictions['LightGBM']
}

for name, pred in submissions.items():
    submission = pd.DataFrame({
        'ID': test_fe['id'],
        'BeatsPerMinute': pred
    })
    
    filename = f'submission_{name}.csv'
    submission.to_csv(filename, index=False)
    print(f"\nCreated: {filename}")
    print(f"  Mean: {pred.mean():.2f}, Std: {pred.std():.2f}")
    print(f"  Min: {pred.min():.2f}, Max: {pred.max():.2f}")

# Also create the main submission.csv with ensemble
submission_final = pd.DataFrame({
    'ID': test_fe['id'],
    'BeatsPerMinute': weighted_pred
})
submission_final.to_csv('submission.csv', index=False)
print("\n✓ Main submission saved as: submission.csv")


# Quick Fix for Submission Format

import pandas as pd

# Read the existing submission
submission = pd.read_csv('submission.csv')

print("Current submission info:")
print(f"Columns: {list(submission.columns)}")
print(f"First few rows:")
print(submission.head())

# Check the sample submission format
sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
print("\nSample submission format:")
print(f"Columns: {list(sample.columns)}")
print(f"First few rows:")
print(sample.head())

# Fix the submission
# 1. Rename column if needed (case sensitive)
if 'id' in submission.columns:
    submission = submission.rename(columns={'id': 'ID'})
elif 'Id' in submission.columns:
    submission = submission.rename(columns={'Id': 'ID'})

# 2. Ensure ID is integer (no decimals)
submission['ID'] = submission['ID'].astype(int)

# 3. Ensure BeatsPerMinute column name is exact
if 'beatsperminute' in [col.lower() for col in submission.columns]:
    # Find the actual column name and rename it
    for col in submission.columns:
        if col.lower() == 'beatsperminute' and col != 'BeatsPerMinute':
            submission = submission.rename(columns={col: 'BeatsPerMinute'})

# 4. Ensure correct column order (ID first, BeatsPerMinute second)
submission = submission[['ID', 'BeatsPerMinute']]

# Save the fixed submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("FIXED SUBMISSION")
print("="*60)
print(f"Columns: {list(submission.columns)}")
print(f"Shape: {submission.shape}")
print(f"ID dtype: {submission['ID'].dtype}")
print(f"BeatsPerMinute dtype: {submission['BeatsPerMinute'].dtype}")
print("\nFirst 5 rows:")
print(submission.head())
print("\nLast 3 rows:")
print(submission.tail(3))

# Double-check by reading it back
print("\n" + "="*60)
print("VERIFICATION (reading back the file)")
print("="*60)
check = pd.read_csv('submission.csv')
print(f"Columns after saving: {list(check.columns)}")
print("First 3 rows:")
print(check.head(3))

print("\n✅ submission.csv is now ready for Kaggle!")

