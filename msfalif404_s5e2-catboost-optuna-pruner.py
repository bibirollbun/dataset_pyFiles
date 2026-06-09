!pip install xgboost
!pip install catboost
!pip install lightgbm
!pip install scikit-learn==1.3.1
!pip install optuna


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')

train_df = df.copy()
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

test_df_id = test_df['id']

train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


train_df.head()


test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Value Heatmap")
plt.show()


# Check missing values correlation
missing_corr = train_df.isnull().corr()

# Plot heatmap for missing values correlation
plt.figure(figsize=(12, 8))
sns.heatmap(missing_corr, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.title('Missing Values Correlation Heatmap')
plt.show()


# Identify categorical variables
categorical_features = train_df.select_dtypes(include=['object', 'category']).columns

# One-hot encode categorical variables
train_encoded = pd.get_dummies(train_df, columns=categorical_features)

# Compute correlation matrix
correlation_matrix = train_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()


# Plot boxplots for detecting outliers in numerical columns
numerical_features = train_df.select_dtypes(include=['number']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features) // 3 + 1, 3, i)
    sns.boxplot(y=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
plt.show()


# Plot histogram for Weight Capacity (kg) comparison
plt.figure(figsize=(12, 6))
sns.histplot(train_df['Price'], color='blue', label='Train', kde=True, bins=30, alpha=0.6)
plt.title('Distribution of Price in Train Dataset')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()


categorical_features = train_df.select_dtypes(include=['object', 'category', 'bool']).columns

train_df_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)


train_features = train_df_encoded.drop(columns=['Price'], errors='ignore')
train_target = train_df['Price']


train_features.info()


test_df_encoded.info()


knn_imputer = KNNImputer(n_neighbors=5)

train_df_imputed = pd.DataFrame(knn_imputer.fit_transform(train_features), columns=train_features.columns)
test_df_imputed = pd.DataFrame(knn_imputer.transform(test_df_encoded), columns=train_features.columns)

train_df_imputed['Price'] = train_target


train_df_imputed.isnull().sum()


test_df_imputed.isnull().sum()


# Create Weight Capacity Category using quartiles
q1 = train_df['Weight Capacity (kg)'].quantile(0.25)
q2 = train_df['Weight Capacity (kg)'].quantile(0.50)
q3 = train_df['Weight Capacity (kg)'].quantile(0.75)

bins = [0, q1, q3, np.inf]
labels = ['Low Capacity', 'Medium Capacity', 'High Capacity']

train_df_imputed['Weight Capacity Category'] = pd.cut(train_df['Weight Capacity (kg)'], bins=bins, labels=labels, include_lowest=True)
test_df_imputed['Weight Capacity Category'] = pd.cut(test_df['Weight Capacity (kg)'], bins=bins, labels=labels, include_lowest=True)


categorical_features = train_df_imputed.select_dtypes(include=['object', 'category', 'bool']).columns
categorical_features


train_df_encoded = pd.get_dummies(train_df_imputed, columns=categorical_features, drop_first=True)
test_df_encoded = pd.get_dummies(test_df_imputed, columns=categorical_features, drop_first=True)


X = train_df_encoded.drop(['Price'], axis=1)
y = train_df_encoded['Price']


cv = KFold(n_splits = 5)


xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.05,
    objective='reg:squarederror',
    missing=np.nan,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    tree_method='gpu_hist',
    gpu_id=0
)


catboost_model = CatBoostRegressor(
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.05,
    loss_function='RMSE',
    od_type='Iter',
    subsample=0.8,
    reg_lambda=1.5,
    random_seed=42,
    bootstrap_type='Bernoulli',
    verbose=False,
    task_type='GPU',
    devices='0'
)


lgbm_model = LGBMRegressor(
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.05,
    objective='regression',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    boosting_type='gbdt',
    verbosity=-1,
    device='gpu'
)


models = {
    'XGBoost': xgb_model,
    'CatBoost': catboost_model,
    'LightGBM': lgbm_model
}


# for model_name, model in models.items():
#     print(f"\nTraining {model_name} Model")

#     for fold, (train_index, test_index) in enumerate(cv.split(X), 1):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         # Train model
#         model.fit(X_train, y_train)

#         # Predict
#         y_pred = model.predict(X_test)

#         # Evaluate RMSE
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         print(f'RMSE for fold {fold}: {rmse:.4f}')


import optuna

def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 6, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'verbose': 0,
        'task_type': 'GPU'  # Gunakan GPU jika tersedia
    }

    model = CatBoostRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for step, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)

        # Laporkan nilai intermediate pada setiap langkah
        trial.report(np.mean(rmse_scores), step)

        # Pruning berdasarkan nilai intermediate
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(rmse_scores)

# Membuat objek study dengan pruner
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

# Menjalankan optimisasi
study.optimize(objective, n_trials=50)

# Mendapatkan hyperparameters terbaik
best_params = study.best_params

print("Best Hyperparameters:", best_params)


final_model = CatBoostRegressor(**best_params)


final_model.fit(X, y)
y_pred = final_model.predict(test_df_encoded)


test_df_encoded


y_pred


# Feature importance visualization
feature_importance = final_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Visualization')
plt.show()


pred_result = pd.DataFrame({
    'id': test_df_id,
    'Price': y_pred
})

pred_result.to_csv('final_submission.csv', index=False)


pred_result.shape


pred_result

