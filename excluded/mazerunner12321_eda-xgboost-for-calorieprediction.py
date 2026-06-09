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


!pip install optuna


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna  # Used for hyperparameter tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col='id')


common = pd.merge(train_df, test_df, how='inner')
common


# Gives all rows unique and common
# We will select only rows which are only "left_only" which will prevent data leakage
train_filtered = train_df.merge(test_df, how='outer', indicator=True)
train_unique = train_filtered[train_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])
train_df = train_unique
train_df.head()


train_df.shape


features_cols = [col for col in train_df.columns if col != 'Calories']

train_group = train_df.groupby(features_cols, as_index=False)['Calories'].mean()
train_group.shape


train_df = train_group
# train_df.to_csv('TRAIN_Calorie_data.csv')


sns.countplot(x='Sex',data=train_df, palette='hls')


num_cols = train_df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
  plt.figure(figsize=(8,5))
  plt.subplot(1,2,1)
  sns.histplot(train_df[col],kde=True, bins=30)
  plt.title(f'Distribution of {col}')
  plt.xlabel(col)
  plt.ylabel('Frequency')

  plt.subplot(1,2,2)
  sns.boxplot(train_df[col])
  plt.title(f'Boxplot of {col}')
  plt.xlabel(col)
  plt.ylabel('Value')

  plt.tight_layout()
  plt.show()


num_cols = train_df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    plt.figure(figsize=(10, 5))

    # Histogram with Sex hue
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x=col, hue='Sex', kde=True, bins=30, multiple='layer')
    plt.title(f'Distribution of {col} by Sex')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Boxplot with Sex split
    plt.subplot(1, 2, 2)
    sns.boxplot(data=train_df, x='Sex', y=col)
    plt.title(f'Boxplot of {col} by Sex')
    plt.xlabel('Sex')
    plt.ylabel(col)

    plt.tight_layout()
    plt.show()



def bmi_feature(df):
  df['bmi'] = df['Weight'] / (df['Height']/100)**2
  return df

train_df = bmi_feature(train_df)
test_df = bmi_feature(test_df)


train_df['bmi'] = train_df['bmi'].clip(lower=10, upper=70)
# boxplot
plt.figure(figsize=(8,5))
sns.boxplot(train_df['bmi'])
plt.title('Boxplot of bmi')
plt.xlabel('bmi')
plt.ylabel('Value')
plt.show()


train_df.describe()


# gender based clipping of Height and Weight
train_df.loc[train_df['Sex'] == 1, 'Height'] = train_df.loc[train_df['Sex'] == 1, 'Height'].clip(130, 200)
train_df.loc[train_df['Sex'] == 1, 'Weight'] = train_df.loc[train_df['Sex'] == 1, 'Weight'].clip(30, 180)

train_df.loc[train_df['Sex'] == 0, 'Height'] = train_df.loc[train_df['Sex'] == 0, 'Height'].clip(140, 220)
train_df.loc[train_df['Sex'] == 0, 'Weight'] = train_df.loc[train_df['Sex'] == 0, 'Weight'].clip(40, 200)



# gender based clipping of feature
test_df.loc[test_df['Sex'] == 1, 'Height'] = test_df.loc[test_df['Sex'] == 1, 'Height'].clip(130, 200)
test_df.loc[test_df['Sex'] == 1, 'Weight'] = test_df.loc[test_df['Sex'] == 1, 'Weight'].clip(30, 180)

test_df.loc[test_df['Sex'] == 0, 'Height'] = test_df.loc[test_df['Sex'] == 0, 'Height'].clip(140, 220)
test_df.loc[test_df['Sex'] == 0, 'Weight'] = test_df.loc[test_df['Sex'] == 0, 'Weight'].clip(40, 200)



sns.scatterplot(x=train_df['Duration'], y=train_df['Calories'])
plt.title('Duration vs Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')


train_df['Duration'].corr(train_df['Calories'])


train_df[(train_df['Duration'] < 5) & (train_df['Calories'] > 200)]


# Feature Engineering
train_df['Ratio_Weight/Height'] = train_df['Weight'] / train_df['Height']
test_df['Ratio_Weight/Height'] = test_df['Weight'] / test_df['Height']

train_df['Relative_Heart_Rate'] = train_df['Heart_Rate'] / (220 - train_df['Age'])
test_df['Relative_Heart_Rate'] = test_df['Heart_Rate'] / (220 - test_df['Age'])

train_df['hr_per_min'] = train_df['Heart_Rate'] / train_df['Duration']
test_df['hr_per_min'] = test_df['Heart_Rate'] / test_df['Duration']

train_df['temp_per_min'] = train_df['Body_Temp'] / train_df['Duration']
test_df['temp_per_min'] = test_df['Body_Temp'] / test_df['Duration']

train_df['duration_weight'] = train_df['Duration'] * train_df['Weight']
test_df['duration_weight'] = test_df['Duration'] * test_df['Weight']

train_df['age_group'] = pd.cut(train_df['Age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
test_df['age_group'] = pd.cut(test_df['Age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])



train_df.shape, test_df.shape


train_df.info()


train_df['Cal_per_min'] = train_df['Calories'] / train_df['Duration']


train_df = train_df.drop(columns=['Calories'])


# encode cat columns
le_gender = LabelEncoder()
train_df['Sex'] = le_gender.fit_transform(train_df['Sex'])
test_df['Sex'] = le_gender.transform(test_df['Sex'])

le_age = LabelEncoder()
train_df['age_group'] = le_age.fit_transform(train_df['age_group'])
test_df['age_group'] = le_age.transform(test_df['age_group'])



X = train_df.drop(columns=['Cal_per_min'])
y = train_df['Cal_per_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# # OPtuna Hyperparameter Tuning
# def objective(trial):
#   params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'booster': 'gbtree',
#         'tree_method': 'hist',
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
#         'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
#   }
#   dtrain = xgb.DMatrix(X_train, label=y_train)
#   dval = xgb.DMatrix(X_val, label=y_val)

#   model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=10,
#                     evals=[(dval, 'validation')],
#                     verbose_eval=0)
#   y_pred = model.predict(dval)
#   rmse = mean_squared_error(y_val, y_pred)
#   return rmse

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=1000)

# best_params = study.best_params
# print("Best RMSE:", study.best_value)
# print("Best hyperparameters:", best_params)



import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define fixed best parameters
fixed_best_params = {
    'max_depth': 10,
    'learning_rate': 0.015109052273050223,
    'subsample': 0.6285183690376833,
    'colsample_bytree': 0.9924388793330333,
    'min_child_weight': 9,
    'lambda': 0.16991523707036416,
    'alpha': 0.00014715503292600727,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',    
    'device': 'cuda'
}

# Prepare DMatrix
dtrain_full = xgb.DMatrix(pd.concat([X_train, X_val]), label=pd.concat([y_train, y_val]))
dtest = xgb.DMatrix(X_test, label=y_test)


final_model = xgb.train(
    fixed_best_params,
    dtrain_full,
    num_boost_round=1000,
    evals=[(dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False  
)

# Predict and evaluate
preds_test = final_model.predict(dtest)
rmse_test = mean_squared_error(y_test, preds_test, squared=False)  # RMSE

print("Test RMSE with fixed best params:", rmse_test)


importance_dict = final_model.get_score(importance_type='gain')
importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
print("Feature importance (gain):")
for feat, score in importance:
    print(f"  {feat:20s}: {score:.4f}")

# plot
xgb.plot_importance(final_model, importance_type='gain', show_values=False)
plt.title('XGBoost Feature Importance by Gain')
plt.tight_layout()
plt.show()


dtest_final = xgb.DMatrix(test_df)
preds_capermin = final_model.predict(dtest_final)

# Convert to calories
test_df['predicted_calorie'] = preds_capermin * test_df['Duration']

print(test_df[['predicted_calorie']].head())


sample =  pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample.head()


result_df = pd.DataFrame({'id': test_df.index, 'Calories': test_df['predicted_calorie']})
result_df.head()


result_df.to_csv('Submission_2.csv',index=False)

