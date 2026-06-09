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


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


train.info()


train.describe()


import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('darkgrid')
sns.set_palette('husl')


numerical_features = train.select_dtypes(['int64','float64']).columns
numerical_features


categorical_features = train.select_dtypes(['object']).columns
categorical_features


# Set up subplots
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, 5 * len(numerical_features)))

for i, col in enumerate(numerical_features):
    # Histogram
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution of {col}")

    # Boxplot
    sns.boxplot(x=train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


# Set up subplots
fig, axes = plt.subplots(len(categorical_features), 1, figsize=(14, 5 * len(categorical_features)))

for i, col in enumerate(categorical_features):
    sns.countplot(x=train[col], order=train[col].value_counts().index, ax=axes[i], palette="inferno")
    axes[i].set_title(f"Count Plot of {col}")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)  # Rotate for readability

plt.tight_layout()
plt.show()



sns.heatmap(train[numerical_features].corr())
plt.title("Corelation Heatmap: ")


train.isnull().sum()


# Percentage of missing values in training data

train.isnull().sum()/train.count()*100


test.isnull().sum()


# Percentage of missing values in test data

test.isnull().sum()/test.count()*100


train


columns_to_fill_with_mean = ['Guest_Popularity_percentage', 'Episode_Length_minutes']
columns_to_fill_with_mode = ['Number_of_Ads']

for col in columns_to_fill_with_mean:
    train[col] = train[col].fillna(train[col].mean())

for col in columns_to_fill_with_mode:
    train[col] = train[col].fillna(train[col].mode()[0])

for col in columns_to_fill_with_mean:
    test[col] = test[col].fillna(test[col].mean())

for col in columns_to_fill_with_mode:
    test[col] = test[col].fillna(test[col].mode()[0])


train


train['Episode_Title'].str.split(" ").str[-1].unique()


train['Episode_Title'] = train['Episode_Title'].str.split(" ").str[-1]
test['Episode_Title'] = test['Episode_Title'].str.split(" ").str[-1]


train[categorical_features].nunique()


# Encoding using pd.get_dummies()

# train_encoded_columns = pd.get_dummies(train[['Genre','Publication_Day','Publication_Time']],dtype=int)
# test_encoded_columns = pd.get_dummies(test[['Genre','Publication_Day','Publication_Time']],dtype=int)

# train = pd.concat([train, train_encoded_columns], axis=1)
# test = pd.concat([test, test_encoded_columns], axis=1)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


# Define ordinal categories 
Episode_Sentiment = [['Negative', 'Neutral', 'Positive']]


one_hot_cols = ['Genre','Publication_Day','Publication_Time']
ordinal_cols = ['Episode_Sentiment']


numerical_features


num_cols = ['id', 'Episode_Length_minutes', 'Host_Popularity_percentage','Guest_Popularity_percentage', 'Number_of_Ads']


X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']



preprocessor = ColumnTransformer([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'), one_hot_cols),
    ('ordinal', OrdinalEncoder(categories=Episode_Sentiment), ordinal_cols),
    ('scaler', StandardScaler(), num_cols)
])


train_transformed = preprocessor.fit_transform(train)
test_transformed = preprocessor.transform(test)


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


y_test = sample_submission['Listening_Time_minutes']


import optuna


# Define the objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 50),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        'eval_metric': 'rmse',
    }

    # Train XGBoost model with these parameters
    model = XGBRegressor(**params, random_state=42)
    model.fit(train_transformed, y)

    # Predict on validation set and compute RMSE
    preds = model.predict(test_transformed)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return rmse  # Optuna minimizes RMSE


# Run Optuna study for hyperparameter tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)


# Best parameters from Optuna
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")


# Train XGBoost with best parameters
best_xgb = XGBRegressor(**best_params, random_state=42)
best_xgb.fit(train_transformed, y)


# Final evaluation on test set
xgb_preds = best_xgb.predict(test_transformed)
final_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"Final XGBoost RMSE: {final_rmse:.4f}")


sample_submission.shape,xgb_preds.shape


submission = pd.DataFrame({
    'id': sample_submission['id'],
    'Listening_Time_minutes': xgb_preds
})


submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")

