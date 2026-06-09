import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import optuna



train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')


print("Train Dataset Shape:", train.shape)
print("Test Dataset Shape:", test.shape)
print("\nTrain Dataset Info:")
print(train.info())


print("\nMissing values in Train Dataset:")
print(train.isnull().sum())
print("\nMissing values in Test Dataset:")
print(test.isnull().sum())


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())
print(test.isnull().sum())


train.describe()


print(train['rainfall'].value_counts(normalize=True))


# Create visualizations for better understanding
plt.figure(figsize=(15, 8))

# 1. Target Distribution
plt.subplot(2, 2, 1)
sns.countplot(data=train, x='rainfall')
plt.title('Distribution of Rainfall')

# 2. Correlation Heatmap
plt.subplot(2, 2, 2)
correlation = train.corr()
sns.heatmap(correlation, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')

# 3. Box plots for key features
plt.subplot(2, 2, 3)
sns.boxplot(data=train, x='rainfall', y='humidity')
plt.title('Humidity vs Rainfall')

plt.subplot(2, 2, 4)
sns.boxplot(data=train, x='rainfall', y='pressure')
plt.title('Pressure vs Rainfall')

plt.tight_layout()
plt.show()


# Calculate correlation with target
correlations = train.corr()['rainfall'].sort_values(ascending=False)
print("\nFeature Correlations with Rainfall:")
print(correlations)


import warnings
warnings.filterwarnings('ignore')
# List of numerical features
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                     'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
# Plot distributions for train vs test
plt.style.use('seaborn')
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
fig.suptitle("Feature Distributions: Train vs Test", y=1.02, fontsize=16)

for idx, feature in enumerate(numerical_features):
    row = idx // 2
    col = idx % 2
    
    # Plot histograms
    sns.histplot(data=train, x=feature, label='Train', ax=axes[row, col], alpha=0.5)
    sns.histplot(data=test, x=feature, label='Test', ax=axes[row, col], alpha=0.5)
    

    
    axes[row, col].set_title(feature)
    axes[row, col].legend()

plt.tight_layout()
plt.show()

# Plot distributions by rainfall
plt.figure(figsize=(15, 25))
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
fig.suptitle("Feature Distributions by Rainfall", y=1.02, fontsize=16)

for idx, feature in enumerate(numerical_features):
    row = idx // 2
    col = idx % 2
    
    # Plot histograms
    sns.histplot(data=train[train['rainfall']==1], x=feature, label='Rain', ax=axes[row, col], alpha=0.5, color='green')
    sns.histplot(data=train[train['rainfall']==0], x=feature, label='No Rain', ax=axes[row, col], alpha=0.5, color='red')
    

    axes[row, col].set_title(feature)
    axes[row, col].legend()

plt.tight_layout()
plt.show()


def create_features(df):
    
    df_new = df.copy()
    
    # 1. Temperature-based Features
    df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
    df_new['temp_ratio'] = df_new['temparature'] / df_new['maxtemp']
    df_new['temp_from_dewpoint'] = df_new['temparature'] - df_new['dewpoint']
    df_new['max_min_temp_ratio'] = df_new['maxtemp'] / df_new['mintemp']
    
    # 2. Humidity and Temperature Interactions
    df_new['humid_temp_interaction'] = df_new['humidity'] * df_new['temparature']
    df_new['humid_pressure_interaction'] = df_new['humidity'] * df_new['pressure']
    df_new['humid_dewpoint_interaction'] = df_new['humidity'] * df_new['dewpoint']
    
    # 3. Cloud and Sunshine Features
    df_new['cloud_sunshine_ratio'] = df_new['cloud'] / (df_new['sunshine'] + 1)  # Adding 1 to avoid division by zero
    df_new['cloud_coverage_rate'] = df_new['cloud'] / 100  # Normalize to 0-1 range
    
    # 4. Pressure Changes (using only row-wise calculations)
    df_new['pressure_temp_ratio'] = df_new['pressure'] / df_new['temparature']
    df_new['pressure_humidity_ratio'] = df_new['pressure'] / df_new['humidity']
    
    # 5. Combined Weather Indicators
    df_new['weather_severity'] = (df_new['cloud'] * df_new['humidity']) / (df_new['pressure'] * (df_new['sunshine'] + 1))
    df_new['temp_humidity_index'] = (df_new['temparature'] * df_new['humidity']) / 100
    df_new['pressure_temp_humidity'] = (df_new['pressure'] * df_new['temparature']) / df_new['humidity']
    
    return df_new

train_processed = create_features(train)
test_processed = create_features(test)


y = train_processed['rainfall']
X = train_processed.drop('rainfall', axis=1)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 3000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize and train model
        model = CatBoostClassifier(
            **params,
            eval_metric='AUC',
            random_seed=42,
            verbose=False  
        )
        
        model.fit(
            X_fold_train, 
            y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        # Make predictions and calculate AUC
        y_pred = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_pred)
        auc_scores.append(auc)
    
    return np.mean(auc_scores)


study = optuna.create_study(direction="maximize")  # Maximize for AUC
study.optimize(objective, n_trials=20)


print("Best parameters:", study.best_trial.params)
print("Best CV AUC:", study.best_value)


# Train final model with best parameters
best_params = study.best_trial.params
final_model = CatBoostClassifier(
    **best_params,
    eval_metric='AUC',
    random_seed=42
)

final_model.fit(
    X, y,
    verbose=False
)


feature_names = train_processed.drop(columns=['rainfall']).columns
importance = final_model.feature_importances_

sorted_idx = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), importance[sorted_idx])

plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("CatBoost Feature Importance")

plt.tight_layout()
plt.show()


test_predictions = final_model.predict_proba(test_processed)[:, 1]

submission = pd.DataFrame({
    'id': test.index,
    'rainfall': test_predictions
})
submission.to_csv('submission.csv', index=False)

