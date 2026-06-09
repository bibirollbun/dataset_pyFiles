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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df.head()


df.shape


df.columns


df.info()


df['Genre'] = df['Genre'].astype('category')
df['Publication_Day'] = df['Publication_Day'].astype('category')
df['Publication_Time'] = df['Publication_Time'].astype('category')
df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')


df['Genre'].unique()


df['Publication_Day'].unique()


df['Publication_Time'].unique()


df['Episode_Sentiment'].unique()


df.select_dtypes(include='number').describe()


df.isna().sum()


missing_percentage = (df.isna().sum() / len(df)) * 100
missing_percentage[missing_percentage > 0]


def fillna_(df, feature):
    df[feature] = df[feature].fillna(df[feature].median())


fillna_(df, 'Episode_Length_minutes')
fillna_(df, 'Guest_Popularity_percentage')
fillna_(df, 'Guest_Popularity_percentage')


fillna_(df_test, 'Episode_Length_minutes')
fillna_(df_test, 'Guest_Popularity_percentage')
fillna_(df_test, 'Guest_Popularity_percentage')


corr = df.select_dtypes(include='number').drop('id', axis=1).corr()

plt.figure(figsize=(10, 4))

sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")

plt.xticks(rotation=-45)
plt.show()


columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                         'Guest_Popularity_percentage', 'Number_of_Ads']

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

ax = ax.flatten()

for i, col in enumerate(columns):
    sns.histplot(x=col, data=df, kde=True, ax=ax[i])

plt.tight_layout()  
plt.show()


columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                         'Guest_Popularity_percentage', 'Number_of_Ads']

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

ax = ax.flatten()

for i, col in enumerate(columns):
    sns.boxplot(y=col, data=df, ax=ax[i])

plt.tight_layout()  
plt.show()


columns = df.select_dtypes(include=['object', 'category'])\
        .drop(['Podcast_Name', 'Episode_Title'], axis=1)

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

ax = ax.flatten()

for i, col in enumerate(columns):
    sns.countplot(x=col, data=df, ax=ax[i])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
    
plt.tight_layout()  
plt.show()


columns = df.select_dtypes(include=['object', 'category'])\
        .drop(['Podcast_Name', 'Episode_Title'], axis=1)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax = ax.flatten()

for i, col in enumerate(columns):
    sns.boxplot(y=col, x='Listening_Time_minutes', data=df, ax=ax[i])
    
plt.tight_layout()  
plt.show()


def label_encode_columns(df, columns):

    df = df.copy()
    encoders = {}

    for col in columns:
        encoder = LabelEncoder()
        df[col + '_encoded'] = encoder.fit_transform(df[col])
        df = df.drop(col, axis=1)
        encoders[col] = encoder

    return df, encoders



columns_to_encode = ['Episode_Sentiment', 'Podcast_Name', 'Episode_Title']
df_encoded, encoders = label_encode_columns(df, columns_to_encode)

for col in columns_to_encode:
    df_test[col + '_encoded'] = encoders[col].transform(df_test[col])
    df_test.drop(col, axis=1)



df_dummies = pd.get_dummies(df_encoded, columns=['Genre', 'Publication_Day', 'Publication_Time'], prefix="_")
df_test = pd.get_dummies(df_test, columns=['Genre', 'Publication_Day', 'Publication_Time'], prefix="_")


df_dummies.columns


X = df_dummies.drop(columns=['id', 'Listening_Time_minutes'])
y = df_dummies['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = xgb.XGBRegressor(
#     tree_method='hist',
#     max_depth=12,
#     colsample_bytree=0.8,
#     subsample=0.8,
#     n_estimators=5000,
#     learning_rate=0.02,
#     enable_categorical=True,
#     min_child_weight=10,
#     early_stopping_rounds=50
#     n_jobs=-1,
#     verbose=1
# )

# kf = KFold(n_splits=5, shuffle=True, random_state=42)



model = xgb.XGBRegressor(
    tree_method='hist', 
    device='cuda',
    max_depth=8,               
    min_child_weight=20,        
    gamma=1,                   
    subsample=0.8,              
    colsample_bytree=0.7,      
    learning_rate=0.01,        
    n_estimators=10000,         
    early_stopping_rounds=100,  
    reg_alpha=1,                
    reg_lambda=5,               
    enable_categorical=True,    
    n_jobs=-1,
    random_state=42,
    verbosity=1
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)



fold = 1
scores = []

for train_index, test_index in kf.split(X):
    print(f"Training fold {fold}...")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Fold {fold} RMSE: {rmse:.4f}")
    scores.append(rmse)
    fold += 1

# Final average score
print(f"\nAverage RMSE across folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


df_test.info()


df_test = df_test.drop(['Podcast_Name', 'Episode_Title', 'Episode_Sentiment'], axis=1)


df_test['predict'] = model.predict(df_test.drop(columns = ['id']))
df_submission = pd.DataFrame({
    'id': df_test['id'], 
    'Listening_Time_minutes' : df_test['predict']
})


df_submission.to_csv('submission.csv', index = False)
df_submission.info()




