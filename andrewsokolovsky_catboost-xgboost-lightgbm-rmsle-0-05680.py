import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import KBinsDiscretizer
import time

warnings.filterwarnings('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Info:")
train.info()
print("\nTest Info:")
test.info()
print("\nTrain Describe:")
train.describe()


plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], bins=50, kde=True)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['Calories']), bins=50, kde=True)
plt.title('Distribution of Log(Calories + 1)')
plt.xlabel('Log(Calories + 1)')
plt.ylabel('Count')
plt.show()


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

print("\nSex Distribution:")
print(train['Sex'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=train)
plt.title('Distribution of Sex')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories by Sex')
plt.show()


train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = train.drop_duplicates(subset=cols, keep='first').reset_index(drop=True)
train = train.groupby(by=cols)['Calories'].min().reset_index()


unique_durations_train = train['Duration'].unique()
unique_durations_test = test['Duration'].unique()

for duration in unique_durations_train:
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    train[heart_rate_col] = np.where(train['Duration'] == duration, train['Heart_Rate'], 0)
    train[body_temp_col] = np.where(train['Duration'] == duration, train['Body_Temp'], 0)

for duration in unique_durations_test:
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    test[heart_rate_col] = np.where(test['Duration'] == duration, test['Heart_Rate'], 0)
    test[body_temp_col] = np.where(test['Duration'] == duration, test['Body_Temp'], 0)

unique_ages_train = train['Age'].unique()
unique_ages_test = test['Age'].unique()

for age in unique_ages_train:
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    train[heart_rate_col] = np.where(train['Age'] == age, train['Heart_Rate'], 0)
    train[body_temp_col] = np.where(train['Age'] == age, train['Body_Temp'], 0)

for age in unique_ages_test:
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    test[heart_rate_col] = np.where(test['Age'] == age, test['Heart_Rate'], 0)
    test[body_temp_col] = np.where(test['Age'] == age, test['Body_Temp'], 0)

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

def add_categorical_aggregations(df):
    categorical_cols = ['Sex']
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
    
    for i in range(1, len(categorical_cols) + 1):
        if i == 1:
            for cat_col in categorical_cols:
                aggs = df.groupby(cat_col).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                df = df.merge(aggs, on=cat_col, how='left')
        elif i == 2:
            for j in range(len(categorical_cols)):
                for k in range(j+1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    aggs = df.groupby([cat_col1, cat_col2]).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    df = df.merge(aggs, on=[cat_col1, cat_col2], how='left')
        elif i == 3:
            aggs = df.groupby(categorical_cols).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            df = df.merge(aggs, on=categorical_cols, how='left')
    return df

def add_log_interactions(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            col1 = numerical_features[i]
            col2 = numerical_features[j]
            df_new[f"{col1}_m_{col2}"] = np.log1p(df_new[col1] * df_new[col2])
            df_new[f"{col1}_d_{col2}"] = np.log1p(df_new[col1] / (df_new[col2] + 1e-5))
    return df_new


train['Sex_Reversed'] = 1 - train['Sex']
test['Sex_Reversed'] = 1 - test['Sex']
list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list2 = ['Sex', 'Sex_Reversed']
train = add_feature_cross_terms(train, list1, list2)
test = add_feature_cross_terms(test, list1, list2)
train.drop(columns=['Sex_Reversed'], inplace=True)
test.drop(columns=['Sex_Reversed'], inplace=True)

train = add_categorical_aggregations(train)
test = add_categorical_aggregations(test)


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = add_log_interactions(train, numerical_features)
test = add_log_interactions(test, numerical_features)

columns_match = train.columns.equals(test.columns.append(pd.Index(['Calories'])))
if not columns_match:
    train_without_calories = train.drop(columns=['Calories'])
    common_columns = [col for col in test.columns if col in train_without_calories.columns]
    train_without_calories = train_without_calories[common_columns]
    test = test[common_columns]
    train = pd.concat([train_without_calories, train['Calories']], axis=1)

train_without_calories = train.drop(columns=['Calories'])
columns_match_after_drop = train_without_calories.columns.equals(test.columns)

cat_features = ['Sex']
for col in cat_features:
    train[col] = train[col].astype('int32').astype('category')
    test[col] = test[col].astype('int32').astype('category')

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

X = train.drop(['Calories'], axis=1)
y = np.log1p(train['Calories'])


# CatBoost
n_bins = 10
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
duration_binned = discretizer.fit_transform(train[['Duration']]).astype(int).flatten()

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 12,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'cat_features': cat_features,
    'verbose': 1000,
    'task_type': 'GPU'
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_scores = []
cat_test_predictions = np.zeros(len(test))
cat_oof_predictions = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_binned)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    val_pred = model.predict(X_val)
    val_pred_original = np.expm1(val_pred)
    cat_oof_predictions[val_idx] = val_pred_original
    
    y_val_calories = train.iloc[val_idx]['Calories']
    fold_score = rmsle(y_val_calories, val_pred_original)
    cat_scores.append(fold_score)
    
    print(f"Fold {fold+1} - CatBoost RMSLE: {fold_score:.5f}")
    
    test_fold_preds = np.expm1(model.predict(test))
    cat_test_predictions += test_fold_preds / skf.n_splits

print(f"\nCatBoost Ortalama RMSLE: {np.mean(cat_scores):.5f}")
print(f"CatBoost Standart Sapma RMSLE: {np.std(cat_scores):.5f}")

# XGBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_scores = []
xgb_test_predictions = np.zeros(len(test))
xgb_oof_predictions = np.zeros(len(X))

xgb_params = {
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist'
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_pred = model.predict(X_val)
    val_pred_original = np.expm1(val_pred)
    xgb_oof_predictions[val_idx] = val_pred_original
    
    y_val_calories = train.iloc[val_idx]['Calories']
    fold_score = rmsle(y_val_calories, val_pred_original)
    xgb_scores.append(fold_score)
    
    print(f"Fold {fold+1} - XGBoost RMSLE: {fold_score:.5f}")
    
    test_fold_preds = np.expm1(model.predict(test))
    xgb_test_predictions += test_fold_preds / kf.n_splits


print(f"\nXGBoost Ortalama RMSLE: {np.mean(xgb_scores):.5f}")
print(f"XGBoost Standart Sapma RMSLE: {np.std(xgb_scores):.5f}")

sub_df = pd.DataFrame()
sub_df['id'] = submission['id']
sub_df['Calories_cat'] = np.clip(cat_test_predictions, 1, 314)
sub_df['Calories_xgb'] = np.clip(xgb_test_predictions, 1, 314)
sub_df['Calories'] = (sub_df['Calories_cat'] + sub_df['Calories_xgb']) / 2
sub_df[['id', 'Calories']].to_csv('submission.csv', index=False)

print(f"\nMedian Calories: {sub_df['Calories'].median():.2f}")

