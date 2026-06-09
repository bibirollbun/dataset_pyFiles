import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)


print("\nTrain dataset info: \n")
print(train_df.info())
print("\n\nFirst 5 rows of train data: \n")
train_df.head()


print("\nTarget variable (BeatsPerMinute) statistics: \n")
print(train_df['BeatsPerMinute'].describe())


print("Missing values in train dataset: \n")
print(train_df.isnull().sum())

print("\n\nMissing values in test dataset: \n")
print(test_df.isnull().sum())


print(f"\nDuplicate rows in train: {train_df.duplicated().sum()}")
print(f"Duplicate rows in test: {test_df.duplicated().sum()}")


print("Data types: \n")
print(train_df.dtypes)


print("Statistical summary: \n")
print(train_df.describe())


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers = detect_outliers(train_df, 'BeatsPerMinute')
print(f"\nNumber of outliers in BeatsPerMinute: {len(outliers)}")


plt.figure(figsize=(15, 10))

# Target variable distribution
plt.subplot(2, 3, 1)
plt.hist(train_df['BeatsPerMinute'], bins=50, alpha=0.7)
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
correlation_matrix = train_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

plt.subplot(2, 3, 3)
key_features = ['RhythmScore', 'AudioLoudness', 'Energy']
for i, feature in enumerate(key_features):
    plt.boxplot(train_df[feature], positions=[i], widths=0.6)
plt.xticks(range(len(key_features)), key_features, rotation=45)
plt.title('Box Plots of Key Features')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Target distribution
axes[0, 0].hist(train_df['BeatsPerMinute'], bins=50, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Distribution of BeatsPerMinute (Target)')
axes[0, 0].set_xlabel('BeatsPerMinute')
axes[0, 0].set_ylabel('Frequency')

# Correlation with target
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
           'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
           'TrackDurationMs', 'Energy']

correlations = []
for feature in features:
    corr = train_df[feature].corr(train_df['BeatsPerMinute'])
    correlations.append(corr)

# Plot correlations
axes[0, 1].barh(features, correlations, color='lightcoral')
axes[0, 1].set_title('Feature Correlations with BeatsPerMinute')
axes[0, 1].set_xlabel('Correlation')

# Top correlated features scatter plots
top_features = ['RhythmScore', 'Energy', 'MoodScore']
for i, feature in enumerate(top_features):
    row = (i + 2) // 3
    col = (i + 2) % 3
    if row < 2:
        axes[row, col].scatter(train_df[feature], train_df['BeatsPerMinute'], 
                              alpha=0.1, s=1)
        axes[row, col].set_title(f'{feature} vs BeatsPerMinute')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('BeatsPerMinute')

# Box plot for energy levels
axes[1, 2].boxplot([train_df[train_df['Energy'] < 0.3]['BeatsPerMinute'],
                   train_df[(train_df['Energy'] >= 0.3) & (train_df['Energy'] < 0.7)]['BeatsPerMinute'],
                   train_df[train_df['Energy'] >= 0.7]['BeatsPerMinute']], 
                  labels=['Low Energy', 'Medium Energy', 'High Energy'])
axes[1, 2].set_title('BPM Distribution by Energy Level')
axes[1, 2].set_ylabel('BeatsPerMinute')

plt.tight_layout()
plt.show()


feature_correlations = train_df.corr()['BeatsPerMinute'].abs().sort_values(ascending=False)
print("\nFeature correlations with BeatsPerMinute:")
print(feature_correlations[1:])


def engineer_features(df):
    """
    Create comprehensive engineered features for BPM prediction
    """
    df_engineered = df.copy()
    
    # 1. Rhythm and Energy Interactions (Most Important for BPM)
    df_engineered['RhythmEnergyProduct'] = df['RhythmScore'] * df['Energy']
    df_engineered['RhythmEnergySum'] = df['RhythmScore'] + df['Energy']
    df_engineered['RhythmEnergyDiff'] = df['RhythmScore'] - df['Energy']
    df_engineered['RhythmEnergyRatio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
    
    # 2. Audio Characteristics
    df_engineered['AudioIntensity'] = df['AudioLoudness'] * df['Energy']
    df_engineered['AudioQualityRatio'] = df['AcousticQuality'] / (abs(df['AudioLoudness']) + 1e-8)
    df_engineered['AudioEnergyProduct'] = df['AcousticQuality'] * df['Energy']
    
    # 3. Performance and Mood Features
    df_engineered['PerformanceMoodProduct'] = df['LivePerformanceLikelihood'] * df['MoodScore']
    df_engineered['TotalPerformanceScore'] = (df['LivePerformanceLikelihood'] * 
                                             df['InstrumentalScore'] * 
                                             df['MoodScore'])
    df_engineered['MoodEnergyProduct'] = df['MoodScore'] * df['Energy']
    df_engineered['MoodRhythmProduct'] = df['MoodScore'] * df['RhythmScore']
    
    # 4. Track Duration Features
    df_engineered['DurationMinutes'] = df['TrackDurationMs'] / 60000
    df_engineered['DurationSeconds'] = df['TrackDurationMs'] / 1000
    df_engineered['DurationCategory'] = pd.cut(df['TrackDurationMs'], bins=5, labels=False)
    df_engineered['DurationEnergyRatio'] = df['TrackDurationMs'] / (df['Energy'] + 1e-8)
    
    # 5. Vocal vs Instrumental Balance
    df_engineered['VocalInstrumentalRatio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-8)
    df_engineered['VocalInstrumentalProduct'] = df['VocalContent'] * df['InstrumentalScore']
    df_engineered['VocalEnergyProduct'] = df['VocalContent'] * df['Energy']
    
    # 6. Composite Scores
    df_engineered['OverallAudioScore'] = (df['RhythmScore'] + df['Energy'] + df['MoodScore']) / 3
    df_engineered['TechnicalQualityScore'] = (df['AcousticQuality'] + abs(df['AudioLoudness'])/30) / 2
    df_engineered['PerformanceQualityScore'] = (df['LivePerformanceLikelihood'] + 
                                               df['InstrumentalScore']) / 2
    
    # 7. Polynomial Features for Key Predictors
    df_engineered['RhythmScore_squared'] = df['RhythmScore'] ** 2
    df_engineered['Energy_squared'] = df['Energy'] ** 2
    df_engineered['MoodScore_squared'] = df['MoodScore'] ** 2
    df_engineered['RhythmScore_cubed'] = df['RhythmScore'] ** 3
    
    # 8. Logarithmic and Square Root Transformations
    df_engineered['log_TrackDuration'] = np.log1p(df['TrackDurationMs'])
    df_engineered['sqrt_Energy'] = np.sqrt(df['Energy'])
    df_engineered['sqrt_RhythmScore'] = np.sqrt(df['RhythmScore'])
    
    # 9. Binned Features
    df_engineered['EnergyBin'] = pd.cut(df['Energy'], bins=10, labels=False)
    df_engineered['RhythmBin'] = pd.cut(df['RhythmScore'], bins=10, labels=False)
    df_engineered['MoodBin'] = pd.cut(df['MoodScore'], bins=10, labels=False)
    
    # 10. Advanced Ratio Features
    df_engineered['EnergyToMoodRatio'] = df['Energy'] / (df['MoodScore'] + 1e-8)
    df_engineered['RhythmToAudioRatio'] = df['RhythmScore'] / (abs(df['AudioLoudness']) + 1e-8)
    
    return df_engineered


train_enhanced = engineer_features(train_df)
test_enhanced = engineer_features(test_df)

print(f"Original train features: {train_df.shape[1]}")
print(f"Enhanced train features: {train_enhanced.shape[1]}")
print(f"New features created: {train_enhanced.shape[1] - train_df.shape[1]}")


X_train = train_enhanced.drop(['id', 'BeatsPerMinute'], axis=1)
y_train = train_enhanced['BeatsPerMinute']
X_test = test_enhanced.drop(['id'], axis=1)


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=None
)


models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=1.0
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Ridge': Ridge(
        alpha=1.0,
        random_state=42
    )
}


model_scores = {}
trained_models = {}
feature_importances = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Handle Ridge regression separately (needs scaling)
    if name == 'Ridge':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        
        model.fit(X_train_scaled, y_train_split)
        y_pred_val = model.predict(X_val_scaled)
        
        # Store scaler for later use
        model.scaler = scaler
    else:
        model.fit(X_train_split, y_train_split)
        y_pred_val = model.predict(X_val_split)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val_split, y_pred_val))
    model_scores[name] = rmse
    trained_models[name] = model
    
    print(f"{name} Validation RMSE: {rmse:.4f}")
    
    # Cross-validation
    if name == 'Ridge':
        # For Ridge, we need to scale the full dataset
        X_train_full_scaled = StandardScaler().fit_transform(X_train)
        cv_scores = cross_val_score(Ridge(alpha=1.0), X_train_full_scaled, y_train, 
                                   cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(cv_scores.std() * 2)
    print(f"{name} CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")
    
    # Store feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_


best_model_name = min(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]
print(f"\nBest individual model: {best_model_name} with RMSE: {model_scores[best_model_name]:.4f}")


ensemble_models = []
weights = []

# Exclude Ridge from ensemble for simplicity (or include with proper scaling)
tree_models = {k: v for k, v in trained_models.items() if k != 'Ridge'}

for name, model in tree_models.items():
    ensemble_models.append((name.lower(), model))
    # Weight inversely proportional to RMSE
    weight = 1.0 / model_scores[name]
    weights.append(weight)

# Normalize weights
weights = np.array(weights)
weights = weights / weights.sum()

print("Ensemble composition and weights:")
for i, (name, _) in enumerate(ensemble_models):
    print(f"{name:15}: {weights[i]:.3f}")


ensemble = VotingRegressor(
    estimators=ensemble_models,
    weights=weights
)
print("\nTraining ensemble model...")
ensemble.fit(X_train, y_train)


ensemble_pred_val = ensemble.predict(X_val_split)
ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_pred_val))
print(f"Ensemble Validation RMSE: {ensemble_rmse:.4f}")


ensemble_cv_scores = cross_val_score(ensemble, X_train, y_train, 
                                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ensemble_cv_rmse = np.sqrt(-ensemble_cv_scores.mean())
ensemble_cv_std = np.sqrt(ensemble_cv_scores.std() * 2)
print(f"Ensemble CV RMSE: {ensemble_cv_rmse:.4f} (+/- {ensemble_cv_std:.4f})")


if best_model_name in feature_importances:
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances[best_model_name]
    }).sort_values('importance', ascending=False)
    
    print(f"Top 20 most important features ({best_model_name}):")
    print(feature_importance_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


final_predictions = ensemble.predict(X_test)


sample_submission.head()


submission = pd.DataFrame({
    'id': sample_submission['id'],
    'BeatsPerMinute': final_predictions
})
submission.to_csv('submission.csv', index=False)

