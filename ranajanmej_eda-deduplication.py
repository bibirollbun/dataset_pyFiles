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
warnings.simplefilter('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.head()


# # Find duplicate groups
# duplicate_groups = train_df.groupby(['Podcast_Name', 'Episode_Title']).size().reset_index(name='Count')

# # Filter groups that have duplicates
# duplicates_only = duplicate_groups[duplicate_groups['Count'] > 1]

# print(duplicates_only)



# # Drop duplicates, keeping the first occurrence
# train_df_cleaned = train_df.drop_duplicates(subset=['Podcast_Name', 'Episode_Title'], keep='first')


# train_df_cleaned.shape,train_df.shape


# train_df_cleaned.head()


from sklearn.model_selection import train_test_split
X = train_df.drop(columns = 'Listening_Time_minutes')
y = train_df['Listening_Time_minutes']


X['null_ELM'] =X['Episode_Length_minutes'].apply(lambda x: 1 if pd.isna(x) else 0)
test_df['null_ELM'] = test_df['Episode_Length_minutes'].apply(lambda x: 1 if pd.isna(x) else 0)


X.null_ELM.unique()


test_df.null_ELM.unique()


X.head()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


X_train.shape,X_test.shape


X_train.isnull().sum()


test_df.isnull().sum()


test_df.isnull().sum()


# Step 1: Calculate group-wise median
group_median_df = X_train.groupby(['Podcast_Name', 'Genre'])['Episode_Length_minutes'].median().reset_index()

# Step 2: Rename the median column
group_median_df.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

# Step 3: Merge the group medians back to the original dataframe
X_train = X_train.merge(group_median_df, on=['Podcast_Name', 'Genre'], how='left')

X_train.head()
# Step 4: Fill missing values using the group median
X_train['Episode_Length_minutes'] = X_train['Episode_Length_minutes'].fillna(X_train['Group_Median'])

# Step 5: Drop the helper column
X_train.drop(columns=['Group_Median'], inplace=True)


# Merge group median with X_test and test_df
X_test = X_test.merge(group_median_df, on=['Podcast_Name', 'Genre'], how='left')
test_df = test_df.merge(group_median_df, on=['Podcast_Name', 'Genre'], how='left')

# Fill missing values
X_test['Episode_Length_minutes'] = X_test['Episode_Length_minutes'].fillna(X_test['Group_Median'])
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Group_Median'])

# Drop helper column
X_test.drop(columns=['Group_Median'], inplace=True)
test_df.drop(columns=['Group_Median'], inplace=True)



X_train.isnull().sum()


X_test.isnull().sum()


test_df.isnull().sum()


group_median = X_train.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

X_train = X_train.merge(group_median, on='Podcast_Name', how='left')
X_train['Episode_Length_minutes'] = X_train['Episode_Length_minutes'].fillna(X_train['Group_Median'])
X_train.drop(columns=['Group_Median'], inplace=True)


group_median = X_train.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

X_test = X_test.merge(group_median, on='Podcast_Name', how='left')
X_test['Episode_Length_minutes'] = X_test['Episode_Length_minutes'].fillna(X_test['Group_Median'])
X_test.drop(columns=['Group_Median'], inplace=True)


X_test.isnull().sum()


group_median = X_train.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

test_df = test_df.merge(group_median, on='Podcast_Name', how='left')
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Group_Median'])
test_df.drop(columns=['Group_Median'], inplace=True)


test_df.isnull().sum()


test_df.head()


test_ids = test_df['id']


test_df.columns


X_train.isnull().sum()


median_value = X_train.groupby('Podcast_Name')['Guest_Popularity_percentage'].median().reset_index()
median_value.rename(columns = {'Guest_Popularity_percentage':'GPP'},inplace =True)
X_train = X_train.merge(median_value ,on = 'Podcast_Name',how = 'left')
X_train['Guest_Popularity_percentage'] = X_train['Guest_Popularity_percentage'].fillna(X_train['GPP'])
X_train.drop(columns = ['GPP'],inplace = True)


X_train.head()


X_train.isnull().sum()


X_test = X_test.merge(median_value ,on = 'Podcast_Name',how = 'left')
X_test['Guest_Popularity_percentage'] = X_test['Guest_Popularity_percentage'].fillna(X_test['GPP'])
X_test.drop(columns = ['GPP'],inplace = True)


X_test.isnull().sum()


test_df = test_df.merge(median_value ,on = 'Podcast_Name',how = 'left')
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(test_df['GPP'])
test_df.drop(columns = ['GPP'],inplace = True)


test_df.isnull().sum()


categorical_cols = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time']


# # Extract numeric episode number (first occurrence of a number)
# X_train['episode_number'] = X_train['Episode_Title'].str.extract(r'(\d+)').astype(float)
# X_test['episode_number'] = X_test['Episode_Title'].str.extract(r'(\d+)').astype(float)
# test_df['episode_number'] = test_df['Episode_Title'].str.extract(r'(\d+)').astype(float)


# X_train.drop(columns = 'Episode_Title',inplace = True)
# X_test.drop(columns = 'Episode_Title',inplace = True)
# test_df.drop(columns = 'Episode_Title',inplace = True)


X_train.head()


len(X_train.Podcast_Name.unique())


# X_test.drop(columns = ['Podcast_Name','Episode_Title'],inplace = True)
# test_df.drop(columns = ['Podcast_Name','Episode_Title'],inplace = True)


X_train.head()


X_train.shape,X_test.shape,test_df.shape


X_train.columns


test_df.columns


# Using pandas
X_train = pd.get_dummies(X_train, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first=True,dtype = int) # drop_first avoids multicollinearity
X_test = pd.get_dummies(X_test, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first=True,dtype = int)  # drop_first avoids multicollinearity
test_df = pd.get_dummies(test_df, columns=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first=True,dtype = int)


X_train.shape,X_test.shape,test_df.shape


X_train.head()


X_train.drop(columns = 'id',inplace = True)
X_test.drop(columns = 'id',inplace = True)
test_df.drop(columns = 'id',inplace = True)


X_train.head()


cols_to_scale = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[cols_to_scale])
X_train[cols_to_scale] = X_train_scaled
X_train.head()


X_test_scaled = scaler.transform(X_test[cols_to_scale])
X_test[cols_to_scale] = X_test_scaled
X_test.head()


test_df_scaled = scaler.transform(test_df[cols_to_scale])
test_df[cols_to_scale] = test_df_scaled
test_df.head()


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Define the model
xgb_model = XGBRegressor(
    n_estimators=10000,         # Large number, let early stopping decide when to stop
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)

# Fit the model with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,    # Stop if validation score doesn't improve after 10 rounds
    verbose=True
)

# Predict
y_pred = xgb_model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


# ✅ Predict on test data
test_preds = xgb_model.predict(test_df)

# ✅ Prepare submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,         # Or use test IDs if present in your original test CSV
    'Listening_Time_minutes': test_preds         # Replace 'target' with the actual column name required by Kaggle
})

# ✅ Save to CSV
submission.to_csv('submission12.csv', index=False)

print("✅ Submission file 'submission.csv' created.")

