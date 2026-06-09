import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()

# For missing values
print("Missing values in train set:")
print(train.isnull().sum())
print("\nMissing values in test set:")
print(test.isnull().sum())


# Visualization layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Distributions', fontsize=16)

# Plot histograms for key features
features_to_plot = ['RhythmScore', 'AudioLoudness', 'Energy', 'BeatsPerMinute', 'MoodScore', 'TrackDurationMs']

for i, feature in enumerate(features_to_plot):
    row, col = i // 3, i % 3
    train[feature].hist(bins=30, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Target variable distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['BeatsPerMinute'], kde=True)
plt.title('Distribution of Beats Per Minute (BPM)')
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()

# Relationship between key features and target
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.scatterplot(data=train, x='RhythmScore', y='BeatsPerMinute', ax=axes[0, 0])
axes[0, 0].set_title('RhythmScore vs BPM')

sns.scatterplot(data=train, x='Energy', y='BeatsPerMinute', ax=axes[0, 1])
axes[0, 1].set_title('Energy vs BPM')

sns.scatterplot(data=train, x='AudioLoudness', y='BeatsPerMinute', ax=axes[1, 0])
axes[1, 0].set_title('AudioLoudness vs BPM')

sns.scatterplot(data=train, x='MoodScore', y='BeatsPerMinute', ax=axes[1, 1])
axes[1, 1].set_title('MoodScore vs BPM')

plt.tight_layout()
plt.show()


X = train.drop(['id', 'BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']
X_test = test.drop('id', axis=1)


def create_enhanced_features(df):
    
    df_enhanced = df.copy()
    
    # Interaction features
    df_enhanced['Rhythm_Energy'] = df_enhanced['RhythmScore'] * df_enhanced['Energy']
    df_enhanced['Loudness_Energy'] = df_enhanced['AudioLoudness'] * df_enhanced['Energy']
    df_enhanced['Mood_Energy'] = df_enhanced['MoodScore'] * df_enhanced['Energy']
    
    # Polynomial features
    df_enhanced['RhythmScore_sq'] = df_enhanced['RhythmScore'] ** 2
    df_enhanced['Energy_sq'] = df_enhanced['Energy'] ** 2
    df_enhanced['AudioLoudness_sq'] = df_enhanced['AudioLoudness'] ** 2
    
    # Ratio features (with small epsilon to avoid division by zero)
    df_enhanced['Energy_per_Rhythm'] = df_enhanced['Energy'] / (df_enhanced['RhythmScore'] + 1e-6)
    df_enhanced['Mood_per_Energy'] = df_enhanced['MoodScore'] / (df_enhanced['Energy'] + 1e-6)
    
    return df_enhanced


# Applying the features
X_enhanced = create_enhanced_features(X)
X_test_enhanced = create_enhanced_features(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")


cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

cat_train_pred = cat_model.predict(X_train)
cat_val_pred = cat_model.predict(X_val)

# Calculating metrics
cat_train_mae = mean_absolute_error(y_train, cat_train_pred)
cat_val_mae = mean_absolute_error(y_val, cat_val_pred)
cat_train_rmse = np.sqrt(mean_squared_error(y_train, cat_train_pred))
cat_val_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
cat_train_r2 = r2_score(y_train, cat_train_pred)
cat_val_r2 = r2_score(y_val, cat_val_pred)

print("CatBoost Results:")
print(f"Training MAE: {cat_train_mae:.4f}")
print(f"Validation MAE: {cat_val_mae:.4f}")
print(f"Training RMSE: {cat_train_rmse:.4f}")
print(f"Validation RMSE: {cat_val_rmse:.4f}")
print(f"Training RÂ²: {cat_train_r2:.4f}")
print(f"Validation RÂ²: {cat_val_r2:.4f}")


# Feature importance for CatBoost
cat_importance = pd.DataFrame({
    'feature': X_enhanced.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=cat_importance)
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()


test_predictions = cat_model.predict(X_test_enhanced)

# Creating submission file
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_predictions
})

print("\nSubmission BPM stats:")
print(submission['BeatsPerMinute'].describe())

submission.to_csv('/kaggle/working/submission.csv', index=False)

