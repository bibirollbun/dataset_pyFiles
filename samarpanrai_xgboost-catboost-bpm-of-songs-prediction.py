import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")


#CatBoost model
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# XGBoost model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# predictions for ensemble
cat_pred = cat_model.predict(X_val)
xgb_pred = xgb_model.predict(X_val)

# trying different weighting schemes
combined_pred_50_50 = 0.5 * cat_pred + 0.5 * xgb_pred
combined_pred_60_40 = 0.6 * cat_pred + 0.4 * xgb_pred
combined_pred_70_30 = 0.7 * cat_pred + 0.3 * xgb_pred

combined_rmse_50_50 = np.sqrt(mean_squared_error(y_val, combined_pred_50_50))
combined_rmse_60_40 = np.sqrt(mean_squared_error(y_val, combined_pred_60_40))
combined_rmse_70_30 = np.sqrt(mean_squared_error(y_val, combined_pred_70_30))

print(f"Combined 50-50 Ensemble RMSE: {combined_rmse_50_50:.4f}")
print(f"Combined 60-40 Ensemble RMSE: {combined_rmse_60_40:.4f}")
print(f"Combined 70-30 Ensemble RMSE: {combined_rmse_70_30:.4f}")

# using best combined approach
best_combined_rmse = min(combined_rmse_50_50, combined_rmse_60_40, combined_rmse_70_30)
print(f"Best Combined Ensemble RMSE: {best_combined_rmse:.4f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# CatBoost Feature Importance
cat_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(x='importance', y='feature', data=cat_importance, ax=ax1)
ax1.set_title('CatBoost - Top 10 Feature Importance')
ax1.set_xlabel('Importance')

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(x='importance', y='feature', data=xgb_importance, ax=ax2)
ax2.set_title('XGBoost - Top 10 Feature Importance')
ax2.set_xlabel('Importance')

plt.tight_layout()
plt.show()


# determine best weights based on validation performance
if combined_rmse_50_50 == best_combined_rmse:
    cat_weight, xgb_weight = 0.5, 0.5
    print("Using 50-50 ensemble for submission")
elif combined_rmse_60_40 == best_combined_rmse:
    cat_weight, xgb_weight = 0.6, 0.4
    print("Using 60-40 ensemble for submission")
else:
    cat_weight, xgb_weight = 0.7, 0.3
    print("Using 70-30 ensemble for submission")


cat_test_pred = cat_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

# Combining predictions with best weights
final_predictions = (cat_weight * cat_test_pred) + (xgb_weight * xgb_test_pred)


# submission file
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': final_predictions
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Ensemble submission file saved!")

