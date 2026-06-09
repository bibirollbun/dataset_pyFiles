import warnings, ydata_profiling, shap
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

from tqdm import tqdm
from colorama import Fore, Back, Style

from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings

sns.set_style('dark')
warnings.filterwarnings('ignore')
shap.initjs()


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
df.head()


print("\nğŸ“Š DATA TYPES AND NON-NULL COUNTS\n")
df.info()


print("\nğŸ“ˆ DESCRIPTIVE STATISTICS")
display(df.describe().T.style.background_gradient(cmap='Blues'))


print("\nâœ… MISSING VALUE ANALYSIS\n")
display(test_df.isnull().sum())
msno.bar(df, sort='ascending')


ydata_profiling.ProfileReport(df)


print(f"\nğŸ”� Number of duplicate rows in training set: {df.duplicated().sum()}")
print(f"\nğŸ”� Number of duplicate rows in test set: {test_df.duplicated().sum()}")


target = 'BeatsPerMinute'
features = [col for col in df.columns if col not in ['id', target]]

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for i, col in enumerate(features):
    ax = axes[i]

    sns.kdeplot(data=df, x=col, ax=ax, fill=True, alpha=0.6, linewidth=2, label='Train')
    sns.kdeplot(data=test_df, x=col, ax=ax, fill=True, alpha=0.6, linewidth=2, label='Test')
    
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    skewness = stats.skew(df[col])
    ax.set_title(f'{col}\nSkewness: {skewness:.2f}', fontweight='bold', fontsize=12)
    ax.legend()

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
    
plt.suptitle('Distribution of Features: Train vs. Test with Skewness', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(df[target], kde=True, ax=ax1, color='purple', alpha=0.6)
ax1.axvline(df[target].mean(), color='red', linestyle='--', label=f'Mean: {df[target].mean():.2f}')
ax1.axvline(df[target].median(), color='green', linestyle='--', label=f'Median: {df[target].median():.2f}')
ax1.set_title(f'Distribution of {target}', fontweight='bold')
ax1.legend()

sns.boxplot(x=df[target], ax=ax2, color='lightblue')
ax2.set_title(f'Box Plot of {target}', fontweight='bold')

plt.tight_layout()
plt.show()


corr_with_target = df.corr()[target].sort_values(ascending=False)
corr_with_target.drop(target, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette='viridis')
plt.title('Feature Correlation with Beats Per Minute (BPM)', fontweight='bold', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    hb = ax.hexbin(x=df[feature], y=df[target], 
                   gridsize=50, cmap='Blues', bins='log', alpha=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f'{feature} vs. {target}', fontweight='bold')
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Log(Count)')

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
    
plt.suptitle('Hexbin Plots: Feature Relationships with BPM', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


corr_matrix = df.drop(columns=['id']).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))

sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, annot=True, fmt='.2f', 
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})

plt.title('Clustered Correlation Matrix (Lower Triangle)', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()


high_bpm_sample = df.loc[df[target].idxmax()]
low_bpm_sample = df.loc[df[target].idxmin()]
median_sample = df.iloc[(df[target] - df[target].median()).abs().argsort()[:1]].squeeze()

radar_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                  'AcousticQuality', 'InstrumentalScore', 'Energy', 'MoodScore']

def normalize_radar_data(sample, features):
    normalized_data = []
    for f in features:
        min_val = df[f].min()
        max_val = df[f].max()
        norm_val = (sample[f] - min_val) / (max_val - min_val)
        normalized_data.append(norm_val)
    return normalized_data

categories = radar_features
values_high = normalize_radar_data(high_bpm_sample, radar_features)
values_low = normalize_radar_data(low_bpm_sample, radar_features)
values_median = normalize_radar_data(median_sample, radar_features)

categories += categories[:1]
values_high += values_high[:1]
values_low += values_low[:1]
values_median += values_median[:1]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=values_high,
      theta=categories,
      fill='toself',
      name=f'High BPM Song ({high_bpm_sample[target]:.0f} BPM)',
      line_color='red'
))

fig.add_trace(go.Scatterpolar(
      r=values_low,
      theta=categories,
      fill='toself',
      name=f'Low BPM Song ({low_bpm_sample[target]:.0f} BPM)',
      line_color='blue'
))

fig.add_trace(go.Scatterpolar(
      r=values_median,
      theta=categories,
      fill='toself',
      name=f'Median BPM Song ({median_sample[target]:.0f} BPM)',
      line_color='green'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title='<b>Audio Feature Fingerprint: Comparison of Different Songs</b><br><i>Can you guess which is the fast and slow song?</i>',
  title_font_size=16
)

fig.show()


train_fe = df.copy()
test_fe = test_df.copy()

original_features = [f for f in features if f not in ['id', target]]

print("Original number of features:", len(original_features))
print("Original features:", original_features)


# Feature 1: Intensity - How much energy per second?
train_fe['EnergyDensity'] = train_fe['Energy'] / (train_fe['TrackDurationMs'] / 1000)
test_fe['EnergyDensity'] = test_fe['Energy'] / (test_fe['TrackDurationMs'] / 1000)

# Feature 2: Loudness Intensity - How loud per second?
train_fe['LoudnessPerSecond'] = train_fe['AudioLoudness'] / (train_fe['TrackDurationMs'] / 1000)
test_fe['LoudnessPerSecond'] = test_fe['AudioLoudness'] / (test_fe['TrackDurationMs'] / 1000)

# Feature 3: Rhythm-to-Energy Ratio - Does rhythm or energy dominate?
train_fe['RhythmEnergyRatio'] = train_fe['RhythmScore'] / (train_fe['Energy'] + 1e-5)  # Avoid division by zero
test_fe['RhythmEnergyRatio'] = test_fe['RhythmScore'] / (test_fe['Energy'] + 1e-5)

# Feature 4: Acoustic-Vocal Interaction - Typical of certain genres
train_fe['AcousticVocalInteraction'] = train_fe['AcousticQuality'] * train_fe['VocalContent']
test_fe['AcousticVocalInteraction'] = test_fe['AcousticQuality'] * test_fe['VocalContent']

# Feature 5: Mood-Energy Profile - Creates "genre-like" categories
train_fe['EnergeticMood'] = train_fe['Energy'] * train_fe['MoodScore']
test_fe['EnergeticMood'] = test_fe['Energy'] * test_fe['MoodScore']

# Feature 6: Instrumental Focus - How prominent are instruments vs vocals?
train_fe['InstrumentalFocus'] = train_fe['InstrumentalScore'] / (train_fe['VocalContent'] + 1e-5)
test_fe['InstrumentalFocus'] = test_fe['InstrumentalScore'] / (test_fe['VocalContent'] + 1e-5)

# Feature 7: Log transform of duration to handle skewness
train_fe['Log_TrackDuration'] = np.log1p(train_fe['TrackDurationMs'])
test_fe['Log_TrackDuration'] = np.log1p(test_fe['TrackDurationMs'])


cluster_features = original_features.copy()
X_cluster = train_fe[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Finding optimal clusters using Elbow Method...")
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Cluster Number', fontweight='bold')
plt.grid(True)
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_fe['Song_Cluster'] = kmeans.fit_predict(X_scaled)

X_test_scaled = scaler.transform(test_fe[cluster_features])
test_fe['Song_Cluster'] = kmeans.predict(X_test_scaled)

print("\nğŸ�¯ 4.3 Creating Target-Encoded Features...")
print("-" * 50)

cluster_means = train_fe.groupby('Song_Cluster')[target].mean().to_dict()
print("Cluster BPM Means:", cluster_means)

train_fe['Cluster_BPM_Mean'] = train_fe['Song_Cluster'].map(cluster_means)
test_fe['Cluster_BPM_Mean'] = test_fe['Song_Cluster'].map(cluster_means)

print("\nğŸ”— 4.4 Creating Polynomial Features for Key Interactions...")
print("-" * 50)

interaction_features = ['RhythmScore', 'Energy', 'MoodScore', 'AudioLoudness']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features_train = poly.fit_transform(train_fe[interaction_features])
poly_features_test = poly.transform(test_fe[interaction_features])

poly_feature_names = poly.get_feature_names_out(interaction_features)

for i, feature_name in enumerate(poly_feature_names):
    if ' ' in feature_name:
        train_fe[feature_name.replace(' ', '_')] = poly_features_train[:, i]
        test_fe[feature_name.replace(' ', '_')] = poly_features_test[:, i]

print("\nâœ… FEATURE ENGINEERING SUMMARY")
print("-" * 50)

engineered_features = [col for col in train_fe.columns if col not in original_features + ['id', target, 'Song_Cluster']]
print(f"Total engineered features: {len(engineered_features)}")
print("New features created:", engineered_features)

new_feature_corr = train_fe[engineered_features + [target]].corr()[target].sort_values(ascending=False)
new_feature_corr.drop(target, inplace=True)

plt.figure(figsize=(12, 8))
sns.barplot(x=new_feature_corr.values, y=new_feature_corr.index, palette='coolwarm')
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
plt.title('Correlation of Engineered Features with BPM', fontweight='bold', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

final_features = original_features + engineered_features
print(f"\nğŸ�‰ Final number of features for modeling: {len(final_features)}")


X = train_fe[final_features]
y = train_fe[target]
X_test = test_fe[final_features]

def rmsle_cv(model, X, y, n_folds=5):
    """
    Calculates RMSE scores for a model using K-Fold Cross-Validation.
    Returns the scores and the average RMSE.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)
        
        print(f"  Fold {fold+1} | RMSE: {rmse:.4f}")
    
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"  Average RMSE: {mean_rmse:.4f} (Â±{std_rmse:.4f})\n")
    return rmse_scores, mean_rmse


print('ğŸ§®' + Fore.LIGHTYELLOW_EX + ' Establishing a Baseline with Ridge Regression')
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_scores, ridge_avg_rmse = rmsle_cv(ridge_model, X, y)


print('ğŸ’¡' + Fore.LIGHTYELLOW_EX + ' Training LightGBM:')
lgbm_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}
lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_scores, lgbm_avg_rmse = rmsle_cv(lgbm_model, X, y)


print('ğŸŒ€' + Fore.LIGHTYELLOW_EX + ' Training XGBoost:')
xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}
xgb_model = XGBRegressor(**xgb_params)
xgb_scores, xgb_avg_rmse = rmsle_cv(xgb_model, X, y)


print('ğŸ�±' + Fore.LIGHTYELLOW_EX + ' Training CatBoost')
catb_params = {
    'iterations': 10000,
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False
}
catb_model = CatBoostRegressor(**catb_params)
catb_scores, catb_avg_rmse = rmsle_cv(catb_model, X, y)


model_comparison = pd.DataFrame({
    'Model': ['Ridge (Baseline)', 'LightGBM', 'XGBoost', 'CatBoost'],
    'Avg RMSE': [ridge_avg_rmse, lgbm_avg_rmse, xgb_avg_rmse, catb_avg_rmse]
}).sort_values('Avg RMSE')

plt.figure(figsize=(10, 6))
sns.barplot(x='Avg RMSE', y='Model', data=model_comparison, palette='viridis')
plt.title('Model Performance Comparison (Lower is Better)', fontweight='bold')
plt.xlim(model_comparison['Avg RMSE'].min() * 0.995, model_comparison['Avg RMSE'].max() * 1.005)
for i, v in enumerate(model_comparison['Avg RMSE']):
    plt.text(v, i, f' {v:.4f}', color='black', va='center', fontweight='bold')
plt.tight_layout()
plt.show()


# Create Ensemble Predictions (Averaging)
print("Retraining best models on full dataset...")
lgbm_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0}
lgbm_model_full = LGBMRegressor(**lgbm_params)
lgbm_model_full.fit(X, y, eval_set=[(X, y)])


xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0
}
xgb_model_full = XGBRegressor(**xgb_params)
xgb_model_full.fit(X, y, eval_set=[(X, y)], verbose=False)


catb_model_full = CatBoostRegressor(**catb_params)
catb_model_full.fit(X, y)


lgbm_preds = lgbm_model_full.predict(X_test)
xgb_preds = xgb_model_full.predict(X_test)
catb_preds = catb_model_full.predict(X_test)

ensemble_preds = (lgbm_preds + xgb_preds + catb_preds) / 3


# Analyze Model Agreement
# Create a DataFrame of test predictions for analysis
prediction_analysis = pd.DataFrame({
    'LightGBM': lgbm_preds,
    'XGBoost': xgb_preds,
    'CatBoost': catb_preds,
    'Ensemble': ensemble_preds
})

plt.figure(figsize=(12, 6))
for column in prediction_analysis.columns:
    sns.kdeplot(prediction_analysis[column], label=column, fill=True, alpha=0.6)
plt.title('Distribution of Test Predictions by Model', fontweight='bold')
plt.xlabel('Predicted BPM')
plt.legend()
plt.show()


prediction_analysis['Std_Dev'] = prediction_analysis[['LightGBM', 'XGBoost', 'CatBoost']].std(axis=1)
plt.figure(figsize=(10, 5))
sns.histplot(prediction_analysis['Std_Dev'], bins=30, kde=True)
plt.title('Distribution of Standard Deviation in Predictions\n(Measure of Model Disagreement)', fontweight='bold')
plt.xlabel('Standard Deviation of Model Predictions')
plt.show()

print(f"Average disagreement (std dev) between models: {prediction_analysis['Std_Dev'].mean():.4f}")


submission = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': ensemble_preds
})

submission_file_path = 'submission.csv'
submission.to_csv(submission_file_path, index=False)
print(f"âœ… Submission file saved as '{submission_file_path}'")
print(f"ğŸ“Š Submission head:")
display(submission.head())

