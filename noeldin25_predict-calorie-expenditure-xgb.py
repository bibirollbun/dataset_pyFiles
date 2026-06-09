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


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(f'Training Data Shape: {train_df.shape}')
print(f'Testing Data Shape: {test_df.shape}')


train_df.head()


train_df.info()


train_df['Sex'].unique()


train_df = train_df.drop('id', axis=1)


train_df.describe()


train_df.isna().sum()


numeric_cols = train_df.select_dtypes(include='number').columns


corr = train_df[numeric_cols].corr()

plt.figure(figsize=(14, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.show()


fig, axes = plt.subplots(6, 2, figsize=(10, 14))

for i, col in enumerate(numeric_cols.drop('Calories')):
    
    sns.histplot(train_df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Histogram of {col}')

    sns.boxplot(x=train_df[col], ax=axes[i, 1], color='skyblue')
    axes[i, 1].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(x='Sex', data=train_df, palette='viridis', ax=axes[0])
sns.boxplot(x='Calories', y='Sex', data=train_df, palette='viridis', ax=axes[1])

plt.tight_layout()
plt.show()



numeric_cols = numeric_cols.drop('Calories')


def encode_sex(df):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df


def cross_cols(df, numeric_cols):
    for col_2 in numeric_cols:
        new_col = f"Sex_{col_2}"
        df[new_col] = df[col_2] * df['Sex']
    
    return df


def add_sex_aggregations(df, numeric_cols):
    agg_df = df.groupby('Sex')[numeric_cols]\
        .agg(['mean', 'std', 'max', 'min']).reset_index()

    agg_df.columns = ['Sex'] + [f'{col}_{stat}' for col, stat in agg_df.columns[1:]]

    df = df.merge(agg_df, on='Sex', how='left')

    return df
    


def bmi(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2 
    return df


def hr_temp_ratio(df):
    df['HR_Temp_Ratio'] = df['Heart_Rate'] / df['Body_Temp']
    return df


def age_category(df):
    df['Age_Category'] = pd.cut(df['Age'],
                                bins=[0, 18, 35, 55, 85],
                                labels=['Teen', 'Young_Adult', 'Adult', 'Elderly'])

    encoder = LabelEncoder()
    df['Age_Category'] = encoder.fit_transform(df['Age_Category'])
    return df



def bmi_age(df):
    df['BMI_Age'] = df['BMI'] * df['Age']
    return df


def heartrate_bodytemp(df):
    df['HeartRate_BodyTemp'] = df['Heart_Rate'] * df['Body_Temp']
    return df


def duration_squared(df):
    df['Duration_Squared'] = df['Duration'] ** 2
    return df


train_df = encode_sex(train_df)
test_df = encode_sex(test_df)

train_df = cross_cols(train_df, numeric_cols)
test_df = cross_cols(test_df, numeric_cols)

train_df = add_sex_aggregations(train_df, numeric_cols)
test_df = add_sex_aggregations(test_df, numeric_cols)

train_df = bmi(train_df)
test_df = bmi(test_df)

train_df = hr_temp_ratio(train_df)
test_df = hr_temp_ratio(test_df)

train_df = age_category(train_df)
test_df = age_category(test_df)

train_df = bmi_age(train_df)
test_df = bmi_age(test_df)

train_df = heartrate_bodytemp(train_df)
test_df = heartrate_bodytemp(test_df)

train_df = duration_squared(train_df)
test_df = duration_squared(test_df)


train_df.info()


train_df.head()


test_df.info()


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']


y_log = np.log1p(y)


import optuna


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.01, 0.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 25, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 7),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmsle'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model = XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_val_pred = np.expm1(model.predict(X_val))
        y_val_true = np.expm1(y_val)

        rmsle = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred))
        rmsle_scores.append(rmsle)

    return np.mean(rmsle_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=60)  

print("\nâœ… Best RMSLE:", study.best_value)
print("ðŸ”§ Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



model = XGBRegressor(**study.best_params)


kf = KFold(n_splits=10, shuffle=True, random_state=42)


fold = 1
train_scores = []
test_scores = []
feature_importance_list = np.zeros(X.shape[1]) 

for train_index, test_index in kf.split(X):
    print(f"Training fold {fold}...")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_log.iloc[train_index], y_log.iloc[test_index]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=100,
        verbose=False,
    )

    y_pred_train = model.predict(X_train)
    y_pred_train = np.expm1(y_pred_train)
    y_train_true = np.expm1(y_train)
    train_rmsle = np.sqrt(mean_squared_log_error(y_train_true, y_pred_train))
    print(f"Fold {fold} Train RMSLE: {train_rmsle:.4f}")

    y_pred_test = model.predict(X_test)
    y_pred_test = np.expm1(y_pred_test)
    y_test_true = np.expm1(y_test)
    test_rmsle = np.sqrt(mean_squared_log_error(y_test_true, y_pred_test))
    print(f"Fold {fold} Test RMSLE: {test_rmsle:.4f}")

    train_scores.append(train_rmsle)
    test_scores.append(test_rmsle)
    
    feature_importance_list += model.feature_importances_
    
    fold += 1

feature_importance_list /= fold - 1
print(f"\nAverage RMSLE across folds: {np.mean(train_scores):.4f} Â± {np.std(train_scores):.4f}")
print(f"\nAverage RMSLE across folds: {np.mean(test_scores):.4f} Â± {np.std(test_scores):.4f}")


importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance_list
})

top_features = importance_df.sort_values(by='Importance', ascending=False).head(25)

plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1]) 
plt.xlabel('Feature Importance')
plt.title('Top 25 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()


log_preds = model.predict(test_df.drop(columns=['id']))

test_df['predict'] = np.expm1(log_preds)

df_submission = pd.DataFrame({
    'id': test_df['id'], 
    'Calories': test_df['predict']
})



df_submission.to_csv('submission.csv', index = False)
df_submission.info()

