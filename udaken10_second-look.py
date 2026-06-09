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


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = 'id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.head()


# train['Episode_Title'] をEpisoce と　数値に分けます

train[['Episode', 'Episode_Number']] = train['Episode_Title'].str.split('Episode ', expand=True)
train['Episode_Number'] = train['Episode_Number'].astype(float)
train.drop(['Episode_Title','Episode'], axis=1, inplace=True)


test[['Episode', 'Episode_Number']] = test['Episode_Title'].str.split('Episode ', expand=True)
test['Episode_Number'] = test['Episode_Number'].astype(float)
test.drop(['Episode_Title','Episode'], axis=1, inplace=True)


target_col = 'Listening_Time_minutes'
target = train[target_col]
train.drop(target_col, axis=1, inplace=True)


for col in train.columns:
    print(col, train[col].unique())
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0],inplace = True)

train.describe()


train.isnull().sum()


train.head()


train_num = train.select_dtypes(include=np.number)
train_cat = train.select_dtypes(exclude=np.number)


train_cat


train['Podcast_Name'].unique()


train['Publication_Day'].unique()


train['Publication_Time'].unique()


all_data = pd.concat([train, test], axis=0)

for col in all_data.columns:
    print(col, all_data[col].unique())
    print('###'*25)


# publication_day とpublication_timeを合わせて一つのカラムにまとめます

# Combine 'Publication_Day' and 'Publication_Time' into a single column
all_data['Publication_DateTime'] = all_data['Publication_Day'] + ' ' + all_data['Publication_Time']

# Drop the original 'Publication_Day' and 'Publication_Time' columns
all_data.drop(['Publication_Day', 'Publication_Time'], axis=1, inplace=True)

# Now 'all_data' contains the combined 'Publication_DateTime' column


all_data['Episode_Length_Host_Popularity'] = all_data['Episode_Length_minutes'] * all_data['Host_Popularity_percentage']


# ganreとpodcast_episoceを合わせて一つのカラムにまとめます

all_data['podcast_genre'] = all_data['Podcast_Name'] + '_' + all_data['Genre']
all_data['podcast_genre'].unique()



train.groupby('Podcast_Name')['Episode_Number'].unique()


train.groupby('Podcast_Name')['Genre'].unique()


train.groupby('Podcast_Name')['Publication_Day'].unique()


all_data


all_data['podcast_name_and_number'] = all_data['Podcast_Name'] + '_' + all_data['Episode_Number'].astype(str)
all_data['podcast_name_and_number'].unique()


from sklearn.preprocessing import LabelEncoder


all_data_cat = all_data.select_dtypes(exclude=np.number)
all_data_num = all_data.select_dtypes(include=np.number)

for col in all_data_cat.columns:
    le = LabelEncoder()
    all_data_cat[col] = le.fit_transform(all_data_cat[col])
    all_data_cat[col] = all_data_cat[col].astype('category')


train_df_cat = all_data_cat.iloc[:len(train), :]
test_df_cat = all_data_cat.iloc[len(train):, :]

train_df_num = all_data_num.iloc[:len(train), :]
test_df_num = all_data_num.iloc[len(train):, :]


train_df_cat


train_df_num.drop('Listening_Time_minutes', axis = 1, inplace = True)


train_data = pd.concat([train_df_cat, train_df_num], axis=1)
test_data = pd.concat([test_df_cat, test_df_num], axis=1)


train_data_target = pd.concat([train_data, target], axis=1)
train_data_target


train_data_target.isnull().sum()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

# Define the objective function for Bayesian Optimization
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    val = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    model = RandomForestRegressor(n_estimators=val,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42,
                                  criterion='squared_error',
                                  n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    return -rmse  # BayesianOptimization maximizes, so we negate RMSE


# Define the parameter bounds for Bayesian Optimization
pbounds = {
    'n_estimators': (100, 1000),
    'max_depth': (5, 30),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
}

# Initialize the Bayesian Optimization object
optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=42)


# Perform Bayesian Optimization
optimizer.maximize(init_points=15, n_iter=45) # Adjust init_points and n_iter as needed

# Get the best parameters and RMSE
print(optimizer.max)

# Train the model with the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])


best_model = RandomForestRegressor(**best_params, random_state=42, criterion='squared_error')
best_model.fit(X_train, y_train)


# Make predictions on the test set
y_pred_best = best_model.predict(test_data)

# Create submission file
sub['Listening_Time_minutes'] = y_pred_best
sub.to_csv('/kaggle/working/submission.csv', index=False)




