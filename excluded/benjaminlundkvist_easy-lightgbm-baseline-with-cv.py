import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# Playground dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")
sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


print("Playground Dataset Info:\n")
print(train.info())

# Check for missing values
print("\nMissing Values - Playground:\n", train.isna().sum())


plt.figure(figsize=(12, 6))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of Beats Per Minute', fontsize=16)
plt.xlabel('BPM', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(train['BeatsPerMinute'].mean(), color='red', linestyle='--', 
            label=f'Mean: {train["BeatsPerMinute"].mean():.2f}')
plt.axvline(train['BeatsPerMinute'].median(), color='green', linestyle='--', 
            label=f'Median: {train["BeatsPerMinute"].median():.2f}')
plt.legend()
plt.show()

# Print descriptive statistics for BPM
target_stats = train['BeatsPerMinute'].describe()
print("Target Statistics:")
print(target_stats)


plt.figure(figsize=(20, 16))
features = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))
correlation_matrix = train.drop('id', axis=1, errors='ignore').corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()

# sort features by correlation with target
target_correlation = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
print("Feature Correlation with BPM:")
print(target_correlation)


X = train.drop('BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']

X_combined = X
y_combined = y


X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=22
)


def train_lightgbm(train, test, target, n_splits=5):
    X = train
    y = target
    X_test = test.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_preds = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        
        # Initialize LightGBM with default parameters
        model = lgb.LGBMRegressor(
            n_estimators=30000,
            learning_rate=0.05,
            num_leaves=100,
            max_depth=12,
            min_child_samples=10,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            boosting_type='gbdt',
            metric='rmse'
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
        )
        
        models.append(model)
        y_preds += model.predict(X_test) / n_splits
    
    print("\nâœ… LightGBM training complete.")
    return y_preds, models


y_preds, models = train_lightgbm(X_combined, test, y_combined)


plt.figure(figsize=(10, 5))
sns.histplot(y_preds, bins=50, kde=True, color='green')
plt.title("Distribution of Predicted Beats Per Minute")
plt.xlabel("Predicted BPM")
plt.ylabel("Count")
plt.grid(True)
plt.show()


importances = np.mean([model.feature_importances_ for model in models], axis=0)
feat_imp = pd.DataFrame({'Feature': X_combined.columns, 'Importance': importances})
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20), palette='viridis')
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    'id': sub['id'],
    'BeatsPerMinute': y_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()

