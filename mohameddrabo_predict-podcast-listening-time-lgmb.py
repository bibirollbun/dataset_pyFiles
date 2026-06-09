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
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import  TransformerMixin
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import re


df  = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df.head()


numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df[col], ax=axes[i], color='blue', fill=True)
    if col in test.columns:
        sns.kdeplot(test[col], ax=axes[i], color='red', fill=True)
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.show()



visualize_data=df[['Number_of_Ads','Episode_Length_minutes']]
visualize_data_test=test[['Number_of_Ads','Episode_Length_minutes']]


visualize_data = visualize_data.dropna()
visualize_data_test = visualize_data_test.dropna()


visualize_data.head()


visualize_data_test.head()


lower= 0.03
upper= 0.99
Number_of_Ads_lower_quantile=  visualize_data['Number_of_Ads'].quantile(lower)
Number_of_Ads_upper_quantile=  visualize_data['Number_of_Ads'].quantile(upper)
Episode_Length_minutes_lower_quantile=  visualize_data['Episode_Length_minutes'].quantile(lower)
Episode_Length_minutes_upper_quantile=  visualize_data['Episode_Length_minutes'].quantile(upper)


visualize_data=  visualize_data[(Episode_Length_minutes_upper_quantile>visualize_data['Episode_Length_minutes'])]
visualize_data=  visualize_data[(Number_of_Ads_upper_quantile>visualize_data['Number_of_Ads'])]
visualize_data_test=  visualize_data_test[(Episode_Length_minutes_upper_quantile>visualize_data_test['Episode_Length_minutes'])]
visualize_data_test=  visualize_data_test[(Number_of_Ads_upper_quantile>visualize_data_test['Number_of_Ads'])]


numerical_cols = visualize_data.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(visualize_data[col], ax=axes[i], color='blue', fill=True)
    if col in visualize_data_test.columns:
        sns.kdeplot(visualize_data_test[col], ax=axes[i], color='red', fill=True)
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.show()



numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])


corr_matrix = df[numerical_cols].corr()

# Affichage avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix")
plt.show()


cat_columns = df.select_dtypes(exclude=np.number).columns
num_cols = 3  # Number of columns in the grid
num_rows = (len(cat_columns) + num_cols - 1) // num_cols

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()
palette = sns.color_palette("Set2", len(df.iloc[:, 0].value_counts()))

for i, col in enumerate(cat_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
for j in range(len(cat_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


cat_columns = test.select_dtypes(exclude=np.number).columns
num_cols = 3  # Number of columns in the grid
num_rows = (len(cat_columns) + num_cols - 1) // num_cols

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()
palette = sns.color_palette("Set2", len(df.iloc[:, 0].value_counts()))

for i, col in enumerate(cat_columns):
    test[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
for j in range(len(cat_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


test.isna().sum()


df.isna().sum()


Episode_Length_minutes_median =  df['Episode_Length_minutes'].median()
Guest_Popularity_percentage_median =  df['Guest_Popularity_percentage'].median()
Number_of_Ads_median  =  df['Number_of_Ads'].median()


def handle_missing_values(df):
    df =  df.copy()
    df['NA_Episode_Length_minutes'] = df['Episode_Length_minutes'].isna().astype(np.int_)
    df['NA_Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].isna().astype(np.int_)
    df['Episode_Length_minutes'] =  (df['Episode_Length_minutes'].fillna(Episode_Length_minutes_median))
    df['Guest_Popularity_percentage'] = ( df['Guest_Popularity_percentage'].fillna(Guest_Popularity_percentage_median))
    df['Number_of_Ads'] = df['Number_of_Ads']
    df  = df.dropna()
    return df

def hangle_outlier(df):
    df= df.copy()
    upper = 0.99
    Episode_Length_minutes_upper  = df['Episode_Length_minutes'].quantile(upper)
    Number_of_Ads_upper  = df['Number_of_Ads'].quantile(upper)
    df['Episode_Length_minutes'] =  np.log1p(df['Episode_Length_minutes'])
    return df


df  =  handle_missing_values(df)
test =  handle_missing_values(test)
df  =  hangle_outlier(df)
test =  hangle_outlier(test)


def create_features(df):
    df = df.copy()
    df['Host_Popularity_percentage_Guest_Popularity_percentage'] = df['Host_Popularity_percentage']/(df['Guest_Popularity_percentage']+1)
    df['Number_of_Ads_Episode_Length_minutes'] = df['Episode_Length_minutes']/(df['Number_of_Ads'] +1)
    df['Episode_Length_minutes_Guest_Popularity_percentage'] =  df['Guest_Popularity_percentage'] /(df['Episode_Length_minutes']+1)
    df['episode_number'] = df['Episode_Title'].apply(lambda x :  x.split(' ')[-1])
    df['Publication_Day'] =  df['Publication_Day'].map(day_mapping)
    df['Publication_Time'] =  df['Publication_Time'].map(time_mapping)
    df['Episode_Sentiment'] =  df['Episode_Sentiment'].map(sentiment_mapping)
    return df


df.head()


df['Episode_Sentiment'].unique()


day_mapping = {
    "Monday": 0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday": 6
}
time_mapping= {
    "Morning":0,
    "Afternoon":1,
    "Evening":2,
    "Night":3
}
sentiment_mapping={
    "Negative":0,
    "Neutral":1,
    "Positive":2
}


df= create_features(df)
test  = create_features(test )


df.head()


class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, except_col=[]):
        self.except_col = except_col
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self,X , y=None):
        df  = X.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.columns = final_col
        for col in self.columns:
            self.label_encoders[col]
            self.label_encoders[col].fit(df[col])
        return self

    def transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda s: '<unknown>' if s not in self.label_encoders[col].classes_ else s)
            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, '<unknown>')
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col])
        return X_copy


pipe  =  Pipeline([ ('label_encoder', MultiColumnLabelEncoder())])


transform_data  =  pipe.fit_transform(df)


test_transform =  pipe.transform(test.drop(['id'], axis='columns'))


test_transform  = test_transform





numerical_cols =  transform_data.columns
corr_matrix = transform_data[numerical_cols].corr()

# Affichage avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix")
plt.show()


X =  transform_data.drop(['id', 'Listening_Time_minutes'], axis='columns')
y = transform_data['Listening_Time_minutes']


features =  X.columns


# import optuna
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from category_encoders import MEstimateEncoder 

# feature_importances =[]
# def objective(trial):
#     params = {
#         "objective": "regression",
#         "metric": "rmse",
#         "boosting_type": "gbdt",   
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 300),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#         "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
#         "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
#         "device":"gpu", 
#         "num_gpu" : 2,
#         "verbose":-1
#     }
#     from sklearn.model_selection import KFold
#     skf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#         encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
#         for col in features:
#             encoder.fit(X_train[[col]], y_train)
#             X_train[f"encoder{col}"] =  encoder.transform(X_train[[col]])
#             X_test[f"encoder{col}"] =  encoder.transform(X_test[[col]])
#         encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
#         for col in features:
#             encoder.fit(X_train[[col]], X_train['Episode_Length_minutes'])
#             X_train[f"Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_train[[col]])
#             X_test[f"Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_test[[col]])
#         encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='median', seed=42)
#         for col in features:
#             encoder.fit(X_train[[col]], X_train['Episode_Length_minutes'])
#             X_train[f"median_Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_train[[col]])
#             X_test[f"median_Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_test[[col]])
#         model = LGBMRegressor(**params)
#         # EntraÃ®nement
#         model.fit(X_train, y_train)
        
#         # PrÃ©diction
#         y_pred = model.predict(X_test)
        
#         # Calcul du score
#         score = np.sqrt(mean_squared_error((y_test), (y_pred)))
#         print(score)
#         scores.append(score)
#         feature_importances.append(model.feature_importances_)
#         return score
#     # Afficher les rÃ©sultats
#     print(f"Scores pour chaque fold : {scores}")
#     print(f"Score moyen : {np.mean(scores):.4f}Â±{np.std(scores)}")
#     return np.mean(scores) + np.std(scores)


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
# print(study.best_value)
# print(study.best_params)


# pd.DataFrame({
#     "columns" :  list(X.columns) + [f"encoder{col}" for col in X.columns]+[f"Episode_Length_minutes_encoder{col}" for col in X.columns],
#     "importance" : np.mean(feature_importances, axis=0)
# }).sort_values(by="importance")


LGBM_params  =  {'learning_rate': 0.026559672283269154, 'n_estimators': 399, 'max_depth': 14, 'num_leaves': 300, 'min_child_samples': 96, 'subsample': 0.6254047524035968, 'colsample_bytree': 0.5063967157980618, 'reg_alpha': 0.0011379440745180492, 'reg_lambda': 0.001551205676382768,"verbose":-1,"device":"gpu", "num_gpu" : 2}


skf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
prediction =  []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for col in features:
            encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
            encoder.fit(X_train[[col]], y_train)
            X_train[f"encoder{col}"] =  encoder.transform(X_train[[col]])
            X_test[f"encoder{col}"] =  encoder.transform(X_test[[col]])
            test_transform[f"encoder{col}"] =  encoder.transform(test_transform[[col]])
    encoder =  TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean', seed=42)
    for col in features:
            encoder.fit(X_train[[col]], X_train['Episode_Length_minutes'])
            X_train[f"Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_train[[col]])
            X_test[f"Episode_Length_minutes_encoder{col}"] =  encoder.transform(X_test[[col]])
            test_transform[f"Episode_Length_minutes_encoder{col}"] =  encoder.transform(test_transform[[col]])
    model = LGBMRegressor(**LGBM_params)
    # EntraÃ®nement
    model.fit(X_train, y_train)
    # PrÃ©diction
    y_pred1 = model.predict(X_test)
    # Calcul du score
    score1 = np.sqrt(mean_squared_error((y_test), (y_pred1)))
    print(f"score model 1 : {score1}")
    scores.append(score1)
    prediction.append (model.predict(test_transform))
# Aficher les rÃ©sultats
print(f"Scores pour chaque fold model 1 : {scores}")
print(f"Score moyen model 2: {np.mean(scores):.4f}Â±{np.std(scores)}")


submission  = pd.DataFrame([], columns=['id', 'Listening_Time_minutes'])
submission.id  =  test.id
submission['Listening_Time_minutes']  =np.mean((prediction), axis=0)


submission.head()


submission.to_csv('submission.csv', index=False)

