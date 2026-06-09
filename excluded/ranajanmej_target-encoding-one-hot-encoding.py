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


from sklearn.model_selection import train_test_split
X = train_df.drop(columns = 'Listening_Time_minutes')
y = train_df['Listening_Time_minutes']


X['null_ELM'] =X['Episode_Length_minutes'].apply(lambda x: 1 if pd.isna(x) else 0)
test_df['null_ELM'] = test_df['Episode_Length_minutes'].apply(lambda x: 1 if pd.isna(x) else 0)


X.null_ELM.unique()


test_df.null_ELM.unique()


X['null_GPP'] =X['Guest_Popularity_percentage'].apply(lambda x: 1 if pd.isna(x) else 0)
test_df['null_GPP'] = test_df['Guest_Popularity_percentage'].apply(lambda x: 1 if pd.isna(x) else 0)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


X_train.shape,X_test.shape


X_train.isnull().sum()


test_df.isnull().sum()


test_ids = test_df['id']


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


X_test.head()


X_test.isnull().sum()


group_median = X_train.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

test_df = test_df.merge(group_median, on='Podcast_Name', how='left')
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Group_Median'])
test_df.drop(columns=['Group_Median'], inplace=True)


test_df.isnull().sum()


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


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def kfold_target_encoding(X, y, categorical_cols, n_splits=5, seed=42):
    """
    Returns target encoded X using KFold encoding to prevent data leakage.
    """
    X_encoded = X.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in categorical_cols:
        X_encoded[col + '_te'] = np.nan

        for train_idx, val_idx in kf.split(X):
            train_fold_X = X.iloc[train_idx]
            train_fold_y = y.iloc[train_idx]
            val_fold_X = X.iloc[val_idx]

            # Compute means on the training fold
            means = train_fold_y.groupby(train_fold_X[col]).mean()

            # Map means to the validation fold
            X_encoded.loc[val_idx, col + '_te'] = val_fold_X[col].map(means)

        # For any unseen category (NaN after mapping), replace with global mean
        global_mean = y.mean()
        X_encoded[col + '_te'].fillna(global_mean, inplace=True)

    return X_encoded.drop(columns=categorical_cols)


X_train.head()


y_train.head()


# Suppose you already have X_train and y_train
categorical_cols = ['Podcast_Name','Episode_Title']

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_train_encoded = kfold_target_encoding(X_train, y_train, categorical_cols)


# You can use standard target encoding (fitted on full training data) for X_test
from category_encoders import TargetEncoder

te = TargetEncoder(cols=categorical_cols)
te.fit(X_train, y_train)
X_test_encoded = te.transform(X_test)
test_df_encoded = te.transform(test_df)


X_train_encoded.head()


X_train_encoded.rename(columns = {'Podcast_Name_te':'Podcast_Name','Episode_Title_te':'Episode_Title'},inplace = True)


X_train_encoded.head()


X_train_encoded['PD_name'] = X_train['Podcast_Name']
X_train_encoded['Ep_Title'] = X_train['Episode_Title']


X_train_encoded.head()





X_test_encoded.head()


X_test_encoded['PD_name'] = X_test['Podcast_Name']
X_test_encoded['Ep_Title'] = X_test['Episode_Title']


X_test_encoded.head()


X_train_encoded = X_train_encoded[X_test_encoded.columns]


X_train_encoded.head()


X_train_encoded.columns


X_test_encoded.columns


test_df_encoded.columns


test_df_encoded.head()


test_df_encoded['PD_name'] = test_df['Podcast_Name']
test_df_encoded['Ep_Title'] = test_df['Episode_Title']


X_train_encoded.shape,X_test_encoded.shape


X_train_encoded.head()


# Using pandas
X_train_encoded = pd.get_dummies(X_train_encoded, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment','PD_name','Ep_Title'], drop_first=True,dtype = int) # drop_first avoids multicollinearity
X_test_encoded = pd.get_dummies(X_test_encoded, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment','PD_name','Ep_Title'], drop_first=True,dtype = int)  # drop_first avoids multicollinearity
test_df_encoded = pd.get_dummies(test_df_encoded, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment','PD_name','Ep_Title'], drop_first=True,dtype = int)


X_train_encoded.shape,X_test_encoded.shape,test_df_encoded.shape


X_train_encoded.shape


X_train_encoded.columns


X_test_encoded.columns


X_train_encoded.drop(columns = 'id',inplace = True)
X_test_encoded.drop(columns = 'id',inplace = True)
test_df_encoded.drop(columns = 'id',inplace = True)


X_train_encoded.shape,X_test_encoded.shape,test_df_encoded.shape


X_train_encoded.head()


cols_to_scale = ['Podcast_Name','Episode_Title','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded[cols_to_scale])
X_train_encoded[cols_to_scale] = X_train_scaled
X_train_encoded.head()


X_test_scaled = scaler.transform(X_test_encoded[cols_to_scale])
X_test_encoded[cols_to_scale] = X_test_scaled
X_test_encoded.head()


test_df_scaled = scaler.transform(test_df_encoded[cols_to_scale])
test_df_encoded[cols_to_scale] = test_df_scaled
test_df_encoded.head()


X_train_encoded.head()


X_train_encoded.columns


X_test_encoded.columns


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
    X_train_encoded, y_train,
    eval_set=[(X_test_encoded, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,    # Stop if validation score doesn't improve after 10 rounds
    verbose=True
)

# Predict
y_pred = xgb_model.predict(X_test_encoded)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


# ✅ Predict on test data
test_preds = xgb_model.predict(test_df_encoded)

# ✅ Prepare submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,         # Or use test IDs if present in your original test CSV
    'Listening_Time_minutes': test_preds         # Replace 'target' with the actual column name required by Kaggle
})

# ✅ Save to CSV
submission.to_csv('submission13.csv', index=False)

print("✅ Submission file 'submission.csv' created.")

