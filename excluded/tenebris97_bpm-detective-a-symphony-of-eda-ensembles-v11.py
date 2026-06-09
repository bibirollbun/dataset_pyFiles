import warnings, ydata_profiling
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


train_fe = df.copy()
test_fe = test_df.copy()
original_features = [f for f in features if f not in ['id', target]]

# 1. Polynomial Features
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    train_fe[f'{col}_sq'] = train_fe[col]**2
    test_fe[f'{col}_sq'] = test_fe[col]**2

# 2. Ratios
epsilon = 1e-6

train_fe['Vocal_Acoustic_Ratio'] = train_fe['VocalContent'] / (train_fe['AcousticQuality'] + epsilon)
test_fe['Vocal_Acoustic_Ratio'] = test_fe['VocalContent'] / (test_fe['AcousticQuality'] + epsilon)

train_fe['Loudness_per_Second'] = train_fe['AudioLoudness'] / (train_fe['TrackDurationMs'] / 1000 + epsilon)
test_fe['Loudness_per_Second'] = test_fe['AudioLoudness'] / (test_fe['TrackDurationMs'] / 1000 + epsilon)

# 3. Handling Skews
for col in ['InstrumentalScore', 'VocalContent', 'AcousticQuality']:
    train_fe[f'Log_{col}'] = np.log1p(train_fe[col])
    test_fe[f'Log_{col}'] = np.log1p(test_fe[col])


# 5. Binary flags for dominant characteristics
train_fe['Is_Instrumental'] = (train_fe['InstrumentalScore'] > 0.6).astype(int)
test_fe['Is_Instrumental'] = (test_fe['InstrumentalScore'] > 0.6).astype(int)

train_fe['Is_Acapella'] = ((train_fe['VocalContent'] > 0.15) & (train_fe['InstrumentalScore'] < 0.1)).astype(int)
test_fe['Is_Acapella'] = ((test_fe['VocalContent'] > 0.15) & (test_fe['InstrumentalScore'] < 0.1)).astype(int)


n_quantiles = 4
quantile_labels = ['Shortest', 'Short', 'Long', 'Longest']

train_fe['Duration_Quantile_Bin'] = pd.qcut(
    train_fe['TrackDurationMs'],
    q=n_quantiles,
    labels=quantile_labels,
    duplicates='drop'
)

test_fe['Duration_Quantile_Bin'] = pd.qcut(
    test_fe['TrackDurationMs'],
    q=n_quantiles,
    labels=quantile_labels,
    duplicates='drop'
)

train_fe = pd.get_dummies(train_fe, columns=['Duration_Quantile_Bin'], drop_first=True)
test_fe = pd.get_dummies(test_fe, columns=['Duration_Quantile_Bin'], drop_first=True)


display(train_fe.head())
display(test_fe.head())


# dropping features based on feature importance of previous versions
features_to_drop = ['AudioLoudness_sq', 'InstrumentalScore', 'Log_AcousticQuality', 'Duration_Quantile_Bin_Long', 'Duration_Quantile_Bin_Short', 'Is_Acapella', 'Is_Instrumental', 'Duration_Quantile_Bin_Longest']

X = train_fe.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_fe['BeatsPerMinute']
X_test = test_fe.drop(['id'], axis=1)

X = X.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)


X_test.head()


# from sklearn.model_selection import RandomizedSearchCV

# def run_random_search(model, params, X, y, model_name, n_iter=30, cv=5):
#     print(f"\nğŸ”� Tuning {model_name}...")
#     search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=params,
#         n_iter=n_iter,
#         cv=cv,
#         scoring='neg_root_mean_squared_error',
#         n_jobs=-1,
#         random_state=42,
#         verbose=1
#     )
#     search.fit(X, y)
#     print(f"âœ… Best {model_name} Params: {search.best_params_}")
#     print(f"âœ… Best {model_name} CV Score (RMSE): {-search.best_score_:.4f}")
#     return search.best_estimator_, -search.best_score_

# lgbm_params = {
#     'num_leaves': [9, 15, 31, 63],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [300, 500],
#     'subsample': [0.8, 0.9],
#     'colsample_bytree': [0.8, 0.9],
#     'reg_alpha': [0, 0.1, 1, 2],
#     'reg_lambda': [0.1, 1, 10, 20],
# }
# lgbm_model = LGBMRegressor(random_state=42, verbose=-1)
# best_lgbm, lgbm_score = run_random_search(lgbm_model, lgbm_params, X, y, "LightGBM")

# xgb_params = {
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [300, 500],
#     'subsample': [0.8, 0.9],
#     'colsample_bytree': [0.8, 0.9],
#     'reg_alpha': [0, 0.1, 1, 2],
#     'reg_lambda': [0.1, 1, 10, 20],
#     'gamma': [0, 0.1],
# }
# xgb_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
# best_xgb, xgb_score = run_random_search(xgb_model, xgb_params, X, y, "XGBoost")

# catb_params = {
#     'depth': [4, 6, 8],
#     'learning_rate': [0.01, 0.05],
#     'iterations': [500, 1000],
#     'l2_leaf_reg': [3, 5, 7, 9],
#     'random_strength': [0.1, 1],
# }
# catb_model = CatBoostRegressor(random_state=42, verbose=0)
# best_catb, catb_score = run_random_search(catb_model, catb_params, X, y, "CatBoost", n_iter=15)

# # --- CHOOSE THE BEST MODEL ---
# print("ğŸ�† MODEL COMPARISON")
# model_scores = {
#     'LightGBM': lgbm_score,
#     'XGBoost': xgb_score,
#     'CatBoost': catb_score
# }

# best_model_name = min(model_scores, key=model_scores.get)
# best_model = {'LightGBM': best_lgbm, 'XGBoost': best_xgb, 'CatBoost': best_catb}[best_model_name]

# print(f"ğŸ¥‡ BEST MODEL: {best_model_name} with RMSE: {model_scores[best_model_name]:.4f}")


best_catb_params = {'random_strength': 0.1, 'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 500, 'depth': 4}
best_catb = CatBoostRegressor(**best_catb_params, random_state=42, verbose=0)
best_catb.fit(X, y)

best_xgb_params = {'subsample': 0.8, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.8}
best_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, verbosity=0)
best_xgb.fit(X, y)

best_lgbm_params = {'subsample': 0.9, 'reg_lambda': 0.1, 'reg_alpha': 1, 'num_leaves': 15, 'n_estimators': 300, 'learning_rate': 0.01, 'colsample_bytree': 0.8}
best_lgbm = LGBMRegressor(**best_lgbm_params, random_state=42, verbose=-1)
best_lgbm.fit(X, y)


# # Assuming 'best_catb' is your trained CatBoost model and 'X' is your full training data
# feature_importances = pd.DataFrame({
#     'feature': X.columns,
#     'importance': best_model.feature_importances_
# }).sort_values('importance', ascending=False)

# # Plot the top 20 features
# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
# plt.title('Top 20 Feature Importances')
# plt.show()

# # Get the list of features to drop (e.g., bottom 40%)
# n_features = len(feature_importances)
# features_to_drop = feature_importances.tail(int(n_features * 0.4))['feature'].tolist()

# print(f"\nğŸ—‘ï¸� Dropping {len(features_to_drop)} least important features...")
# print(features_to_drop)

# # Create new dataframes with selected features
# X_selected = X.drop(columns=features_to_drop)
# X_test_selected = X_test.drop(columns=features_to_drop)


# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import RidgeCV

# estimators = [
#     ('lgbm', best_lgbm),
#     ('xgb', best_xgb),
#     ('catb', best_catb)
# ]

# meta_model = RidgeCV()

# stack = StackingRegressor(
#     estimators=estimators,
#     final_estimator=meta_model,
#     cv=5, # Use cross-validation to generate level-one features
#     n_jobs=-1,
#     passthrough=True # Allows the meta-model to also use the original features
# )

# print("Fitting the stacking model...")
# stack.fit(X, y)
# print("âœ… Stacking model fitted!")

# stacking_predictions = stack.predict(X_test)


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

all_predictions = []
seeds = [42, 123, 666, 888, 101]

for seed in seeds:
    print(f"--- Training with seed: {seed} ---")
    
    best_lgbm.set_params(random_state=seed)
    best_xgb.set_params(random_state=seed)
    # best_catb.set_params(random_state=seed)
    
    estimators = [('lgbm', best_lgbm), ('xgb', best_xgb), ('catb', best_catb)]
    stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5)
    stack.fit(X, y)
    
    predictions = stack.predict(X_test)
    all_predictions.append(predictions)

final_blended_predictions = np.mean(all_predictions, axis=0)


# # Ensemble
# pred_lgbm = best_lgbm.predict(X_test)
# pred_xgb = best_xgb.predict(X_test)
# pred_catb = best_catb.predict(X_test)

# final_predictions = (pred_lgbm + pred_xgb + pred_catb) / 3


submission = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_blended_predictions})
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"ğŸ“¤ Submission file '{submission_file}' created.")
submission.head()




