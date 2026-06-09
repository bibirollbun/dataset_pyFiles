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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, IsolationForest
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy import stats
from scipy.stats import normaltest, skew, kurtosis
import shap
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ENHANCED TERRAIN PRICE REGRESSION ANALYSIS")
print("=" * 80)

train_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/train.csv')
test_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/test.csv')
sample_submission = pd.read_csv('/kaggle/input/terrain-prices-reggression/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

target_col = 'target'
id_col = 'id'
categorical_cols = ['location_type', 'land_use', 'zoning_code']
numerical_cols = [col for col in train_df.columns if col not in categorical_cols + [id_col, target_col]]

print("\n" + "=" * 80)
print("ENHANCED EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n1. ENHANCED STATISTICAL SUMMARY")
print("-" * 40)

def enhanced_summary(df, numerical_cols):
    summary = pd.DataFrame()
    
    for col in numerical_cols:
        col_stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min(),
            'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
            'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0,
            'skewness': skew(df[col]),
            'kurtosis': kurtosis(df[col]),
            'missing': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().sum() / len(df) * 100,
            'unique': df[col].nunique(),
            'zeros': (df[col] == 0).sum(),
            'negative': (df[col] < 0).sum()
        }
        summary[col] = pd.Series(col_stats)
    
    return summary.T

enhanced_stats = enhanced_summary(train_df, numerical_cols[:10])
print(enhanced_stats.round(3))

print("\n2. DISTRIBUTION ANALYSIS")
print("-" * 40)

fig, axes = plt.subplots(5, 8, figsize=(24, 15))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < 40:
        ax = axes[idx]
        
        train_df[col].hist(ax=ax, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2 = ax.twinx()
        train_df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2)
        ax2.set_ylabel('')
        
        stat, p_value = normaltest(train_df[col])
        normality = "Normal" if p_value > 0.05 else "Not Normal"
        
        ax.set_title(f'{col}\n{normality} (p={p_value:.3f})', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Frequency', fontsize=8)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n3. ADVANCED OUTLIER DETECTION")
print("-" * 40)

def detect_outliers_multiple_methods(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    
    z_scores = np.abs(stats.zscore(df[col]))
    z_outliers = df[z_scores > 3]
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(df[[col]])
    iso_outliers = df[outlier_labels == -1]
    
    return {
        'IQR': len(iqr_outliers),
        'Z-score': len(z_outliers),
        'Isolation Forest': len(iso_outliers)
    }

outlier_summary = pd.DataFrame()
for col in ['area_sq_m', 'price_per_m2', 'distance_city_center_km', 'median_income_area']:
    outlier_summary[col] = pd.Series(detect_outliers_multiple_methods(train_df, col))

print(outlier_summary)

print("\n4. ADVANCED CORRELATION ANALYSIS")
print("-" * 40)

numerical_cols_with_target = numerical_cols + [target_col]
correlation_metrics = {
    'Pearson': train_df[numerical_cols_with_target].corr(method='pearson'),
    'Spearman': train_df[numerical_cols_with_target].corr(method='spearman'),
    'Kendall': train_df[numerical_cols_with_target].corr(method='kendall')
}

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for idx, (method, corr_matrix) in enumerate(correlation_metrics.items()):
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    sns.heatmap(corr_matrix.loc[target_corr.index[:20], target_corr.index[:20]], 
                annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .8},
                ax=axes[idx])
    axes[idx].set_title(f'{method} Correlation Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5. 3D VISUALIZATION ANALYSIS")
print("-" * 40)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(train_df['area_sq_m'], 
                     train_df['price_per_m2'], 
                     train_df['target'],
                     c=train_df['target'], 
                     cmap='viridis', 
                     alpha=0.6)

ax.set_xlabel('Area (sq m)')
ax.set_ylabel('Price per m²')
ax.set_zlabel('Target Price')
ax.set_title('3D Visualization: Area vs Price per m² vs Target')

plt.colorbar(scatter, ax=ax, label='Target Price')
plt.tight_layout()
plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n6. FEATURE RELATIONSHIPS ANALYSIS")
print("-" * 40)

key_features = ['area_sq_m', 'price_per_m2', 'distance_city_center_km', 
                'median_income_area', 'neighborhood_quality', 'target']

g = sns.pairplot(train_df[key_features], diag_kind='kde', 
                 plot_kws={'alpha': 0.6})

plt.suptitle('Feature Relationships Analysis', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n7. MULTICOLLINEARITY ANALYSIS")
print("-" * 40)

def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) 
                       for i in range(len(features))]
    return vif_data.sort_values('VIF', ascending=False)

vif_features = [col for col in numerical_cols[:20]]
vif_results = calculate_vif(train_df, vif_features)
print(vif_results.head(10))

print("\n8. MUTUAL INFORMATION ANALYSIS")
print("-" * 40)

X_mi = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
y_mi = train_df[target_col]

mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
mi_results = pd.DataFrame({
    'Feature': numerical_cols,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(mi_results['Feature'][:20], mi_results['MI_Score'][:20])
plt.xlabel('Mutual Information Score')
plt.title('Top 20 Features by Mutual Information with Target')
plt.tight_layout()
plt.savefig('mutual_information.png', dpi=300, bbox_inches='tight')
plt.show()

print(mi_results.head(10))

print("\n9. DIMENSIONALITY REDUCTION VISUALIZATION")
print("-" * 40)

X_dr = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dr)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X_scaled[:1000])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=train_df[target_col], cmap='viridis', alpha=0.6)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('PCA Visualization')
plt.colorbar(scatter, ax=axes[0])

scatter2 = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                           c=train_df[target_col][:1000], cmap='viridis', alpha=0.6)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].set_title('t-SNE Visualization')
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Explained Variance: {np.cumsum(pca.explained_variance_ratio_)[:3]}")

print("\n" + "=" * 80)
print("ENHANCED FEATURE ENGINEERING")
print("=" * 80)

X_train = train_df.copy()
X_test = test_df.copy()

print("\n1. Creating Polynomial Features...")

poly_features = ['area_sq_m', 'price_per_m2', 'land_area_m2']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

poly_train = poly.fit_transform(X_train[poly_features])
poly_test = poly.transform(X_test[poly_features])

poly_feature_names = poly.get_feature_names_out(poly_features)
for i, name in enumerate(poly_feature_names[len(poly_features):]):
    X_train[f'poly_{name}'] = poly_train[:, len(poly_features) + i]
    X_test[f'poly_{name}'] = poly_test[:, len(poly_features) + i]

print("\n2. Creating Domain-Specific Features...")

X_train['total_price_calc'] = X_train['area_sq_m'] * X_train['price_per_m2']
X_test['total_price_calc'] = X_test['area_sq_m'] * X_test['price_per_m2']

X_train['price_to_income_ratio'] = X_train['price_per_m2'] / (X_train['median_income_area'] + 1)
X_test['price_to_income_ratio'] = X_test['price_per_m2'] / (X_test['median_income_area'] + 1)

X_train['location_score'] = (X_train['neighborhood_quality'] * X_train['amenities_score'] * 
                             (1 - X_train['crime_rate']/10) * (1 - X_train['unemployment_rate']))
X_test['location_score'] = (X_test['neighborhood_quality'] * X_test['amenities_score'] * 
                           (1 - X_test['crime_rate']/10) * (1 - X_test['unemployment_rate']))

X_train['env_score'] = (X_train['air_quality_index'] + X_train['noise_pollution_index']) / 200
X_test['env_score'] = (X_test['air_quality_index'] + X_test['noise_pollution_index']) / 200

X_train['risk_score'] = X_train['flood_risk'] + X_train['earthquake_risk']
X_test['risk_score'] = X_test['flood_risk'] + X_test['earthquake_risk']

print("\n3. Creating Cluster-Based Features...")

location_features = ['distance_city_center_km', 'population_density', 
                    'median_income_area', 'employer_density']

kmeans = KMeans(n_clusters=5, random_state=42)
X_train['location_cluster'] = kmeans.fit_predict(X_train[location_features])
X_test['location_cluster'] = kmeans.predict(X_test[location_features])

print("\n4. Applying Log Transformations...")

log_features = ['area_sq_m', 'land_area_m2', 'distance_city_center_km', 
                'population_density', 'median_income_area']

for feat in log_features:
    X_train[f'log_{feat}'] = np.log1p(X_train[feat])
    X_test[f'log_{feat}'] = np.log1p(X_test[feat])

print("\n5. Encoding Categorical Variables...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[f'{col}_encoded'] = le.fit_transform(X_train[col])
    X_test[f'{col}_encoded'] = le.transform(X_test[col])
    label_encoders[col] = le

def target_encode(X_train_df, X_test_df, cat_col, target_series, smoothing=10):
    mean_target = target_series.mean()
    
    agg = X_train_df.groupby(cat_col)[target_series.name].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    smooth_mean = (counts * means + smoothing * mean_target) / (counts + smoothing)
    
    X_train_df[f'{cat_col}_target_encoded'] = X_train_df[cat_col].map(smooth_mean)
    X_test_df[f'{cat_col}_target_encoded'] = X_test_df[cat_col].map(smooth_mean)
    
    X_test_df[f'{cat_col}_target_encoded'].fillna(mean_target, inplace=True)
    
    return X_train_df, X_test_df

X_train['target_temp'] = train_df[target_col]
X_train, X_test = target_encode(X_train, X_test, 'location_type', X_train['target_temp'])
X_train.drop('target_temp', axis=1, inplace=True)

feature_cols = [col for col in X_train.columns 
                if col not in [id_col, target_col] + categorical_cols]

print(f"\nTotal features created: {len(feature_cols)}")

print("\n" + "=" * 80)
print("ENHANCED MODEL TRAINING")
print("=" * 80)

X_train_final = X_train[feature_cols]
X_test_final = X_test[feature_cols]
y_train = train_df[target_col]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

print("\n1. LEARNING CURVES ANALYSIS")
print("-" * 40)

def plot_learning_curves(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='r2', random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation score")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    return train_scores_mean, validation_scores_mean

models_to_analyze = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, model) in enumerate(models_to_analyze.items()):
    plt.sca(axes[idx])
    train_scores, val_scores = plot_learning_curves(model, X_train_scaled, y_train)
    axes[idx].set_title(f'Learning Curve - {name}')
    
    overfitting = np.mean(train_scores[-3:]) - np.mean(val_scores[-3:])
    axes[idx].text(0.02, 0.02, f'Overfitting: {overfitting:.4f}', 
                   transform=axes[idx].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n2. MODEL TRAINING WITH CROSS-VALIDATION")
print("-" * 40)

models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, 
        num_leaves=31, min_data_in_leaf=20, random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.1, 
        random_state=42, verbose=False
    )
}

cv_results = {}
trained_models = {}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')
    cv_results[name] = {
        'mean_r2': scores.mean(),
        'std_r2': scores.std(),
        'scores': scores
    }
    
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    print(f"{name} - Mean R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n3. SHAP ANALYSIS")
print("-" * 40)

rf_model = trained_models['Random Forest']

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_scaled[:1000])

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_final.iloc[:1000], 
                  feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTop 20 Features by SHAP Importance:")
print(shap_importance.head(20))

print("\n4. PERMUTATION IMPORTANCE ANALYSIS")
print("-" * 40)

perm_importance = permutation_importance(
    rf_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
)

perm_imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

plt.figure(figsize=(10, 8))
plt.errorbar(perm_imp_df['importance_mean'][:20], 
             range(20), 
             xerr=perm_imp_df['importance_std'][:20],
             fmt='o', capsize=5)
plt.yticks(range(20), perm_imp_df['feature'][:20])
plt.xlabel('Permutation Importance')
plt.title('Top 20 Features by Permutation Importance')
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5. UNCERTAINTY QUANTIFICATION")
print("-" * 40)

tree_predictions = np.array([tree.predict(X_test_scaled) 
                            for tree in rf_model.estimators_])

predictions_mean = tree_predictions.mean(axis=0)
predictions_std = tree_predictions.std(axis=0)

confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
predictions_lower = predictions_mean - z_score * predictions_std
predictions_upper = predictions_mean + z_score * predictions_std

uncertainty_df = pd.DataFrame({
    'prediction': predictions_mean,
    'std': predictions_std,
    'cv': predictions_std / (predictions_mean + 1e-8),
    'lower_bound': predictions_lower,
    'upper_bound': predictions_upper,
    'interval_width': predictions_upper - predictions_lower
})

print("\nPrediction Uncertainty Statistics:")
print(f"Mean Coefficient of Variation: {uncertainty_df['cv'].mean():.4f}")
print(f"Mean Interval Width: {uncertainty_df['interval_width'].mean():.2f}")
print(f"Predictions with CV > 0.1: {(uncertainty_df['cv'] > 0.1).sum()} ({(uncertainty_df['cv'] > 0.1).sum()/len(uncertainty_df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(uncertainty_df['cv'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(uncertainty_df['cv'].mean(), color='red', linestyle='--', 
                label=f'Mean CV: {uncertainty_df["cv"].mean():.4f}')
axes[0].set_xlabel('Coefficient of Variation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Prediction Uncertainty')
axes[0].legend()

sample_idx = np.random.choice(len(predictions_mean), 100, replace=False)
sample_idx_sorted = sample_idx[np.argsort(predictions_mean[sample_idx])]

axes[1].plot(range(100), predictions_mean[sample_idx_sorted], 'b-', label='Predictions')
axes[1].fill_between(range(100), 
                     predictions_lower[sample_idx_sorted],
                     predictions_upper[sample_idx_sorted],
                     alpha=0.3, label='95% CI')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Sample Predictions with 95% Confidence Intervals')
axes[1].legend()

plt.tight_layout()
plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n6. RESIDUAL ANALYSIS")
print("-" * 40)

y_val_pred = rf_model.predict(X_val)
residuals = y_val - y_val_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].scatter(y_val_pred, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted Values')

stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot of Residuals')

axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Residuals')

axes[1, 1].scatter(y_val, residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Actual Values')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean Residual: {residuals.mean():.2f}")
print(f"Std Residual: {residuals.std():.2f}")
print(f"Skewness: {skew(residuals):.4f}")
print(f"Kurtosis: {kurtosis(residuals):.4f}")

print("\n7. FEATURE INTERACTIONS")
print("-" * 40)

top_features = ['total_price_calc', 'area_sq_m', 'price_per_m2', 
                'distance_city_center_km', 'median_income_area']
top_features_idx = [feature_cols.index(feat) for feat in top_features if feat in feature_cols]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx[:5], top_features[:5])):
    display = PartialDependenceDisplay.from_estimator(
        rf_model, X_train_scaled, features=[feat_idx], 
        feature_names=feature_cols, ax=axes[idx]
    )
    axes[idx].set_title(f'PDP: {feat_name}')

if len(top_features_idx) >= 2:
    display = PartialDependenceDisplay.from_estimator(
        rf_model, X_train_scaled, features=[(top_features_idx[0], top_features_idx[1])],
        feature_names=feature_cols, ax=axes[5]
    )
    axes[5].set_title(f'PDP Interaction: {top_features[0]} x {top_features[1]}')

plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ADVANCED ENSEMBLE MODELING")
print("=" * 80)

base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, 
                              num_leaves=31, min_data_in_leaf=20, random_state=42, verbose=-1))
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=0.5),
    cv=5,
    n_jobs=-1
)

print("Training stacking ensemble...")
stacking_model.fit(X_train_scaled, y_train)

stacking_predictions = stacking_model.predict(X_test_scaled)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

performance_summary = pd.DataFrame(cv_results).T
performance_summary = performance_summary.sort_values('mean_r2', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.bar(performance_summary.index, performance_summary['mean_r2'], 
        yerr=performance_summary['std_r2'], capsize=5)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim(0.95, 1.0)
ax1.set_xticklabels(performance_summary.index, rotation=45)

cv_scores_data = [results['scores'] for results in cv_results.values()]
ax2.boxplot(cv_scores_data, labels=cv_results.keys())
ax2.set_ylabel('R² Score')
ax2.set_title('Cross-Validation Score Distribution')
ax2.set_ylim(0.95, 1.0)
ax2.set_xticklabels(cv_results.keys(), rotation=45)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print(performance_summary)

print("\n" + "=" * 80)
print("CREATING FINAL PREDICTIONS")
print("=" * 80)

submission_final = pd.DataFrame({
    'id': test_df['id'],
    'target': stacking_predictions,
    'uncertainty_std': predictions_std,
    'lower_bound': predictions_lower,
    'upper_bound': predictions_upper
})

submission_final[['id', 'target']].to_csv('submission_enhanced.csv', index=False)
submission_final.to_csv('submission_with_uncertainty.csv', index=False)

print("\nPrediction Summary:")
print(f"Mean: {stacking_predictions.mean():.2f}")
print(f"Std: {stacking_predictions.std():.2f}")
print(f"Min: {stacking_predictions.min():.2f}")
print(f"Max: {stacking_predictions.max():.2f}")
print(f"\nMean Uncertainty (CV): {(predictions_std/predictions_mean).mean():.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nFiles created:")
print("1. submission_enhanced.csv - Final predictions")
print("2. submission_with_uncertainty.csv - Predictions with uncertainty bounds")
print("\nVisualization files:")
print("- distribution_analysis.png")
print("- correlation_analysis.png")
print("- 3d_visualization.png")
print("- feature_relationships.png")
print("- mutual_information.png")
print("- dimensionality_reduction.png")
print("- learning_curves.png")
print("- shap_summary.png")
print("- permutation_importance.png")
print("- uncertainty_analysis.png")
print("- residual_analysis.png")
print("- partial_dependence.png")
print("- model_performance.png")

