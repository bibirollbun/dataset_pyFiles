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


# Imprt Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

 #Load Data set
sample=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


# simple imputer fill missing columns usinf for loop whwere object or categorical fill with most frequent value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
train['Guest_Popularity_percentage'] = imputer.fit_transform(train[['Guest_Popularity_percentage']])
train['Episode_Length_minutes'] = imputer.fit_transform(train[['Episode_Length_minutes']])
train['Number_of_Ads'] = imputer.fit_transform(train[['Number_of_Ads']])
# similarly for test
test['Guest_Popularity_percentage'] = imputer.fit_transform(test[['Guest_Popularity_percentage']])
test['Episode_Length_minutes'] = imputer.fit_transform(test[['Episode_Length_minutes']])


# Define bins by themes
bins = {
    'Tech': [
        'Tech Talks', 'Tech Trends', 'Gadget Geek', 'Innovators'
    ],
    'Health': [
        'Mind & Body', 'Fitness First', 'Healthy Living', 'Wellness Wave', 'Health Hour'
    ],
    'News': [
        'News Roundup', 'Global News', 'World Watch', 'Current Affairs', 'Daily Digest'
    ],
    'Sports': [
        'Sports Central', 'Sport Spot', 'Sports Weekly', "Athlete's Arena", 'Game Day'
    ],
    'Business': [
        'Money Matters', 'Finance Focus', 'Business Briefs', 'Business Insights', 'Market Masters'
    ],
    'Comedy': [
        'Joke Junction', 'Funny Folks', 'Comedy Corner', 'Humor Hub', 'Laugh Line'
    ],
    'Crime': [
        'Criminal Minds', 'True Crime Stories', 'Crime Chronicles', 'Detective Diaries'
    ],
    'Music': [
        'Music Matters', 'Melody Mix', 'Sound Waves', 'Tune Time'
    ],
    'Education': [
        'Study Sessions', 'Learning Lab', 'Life Lessons', 'Educational Nuggets', 'Brain Boost'
    ],
    'Lifestyle': [
        'Home & Living', 'Style Guide', 'Lifestyle Lounge', 'Fashion Forward'
    ]
}
# Invert bins to create a mapping from podcast name → bin label
podcast_bin_map = {podcast: group for group, names in bins.items() for podcast in names}

# Apply to dataset (assumes 'podcast_name' is the column name)
train['podcast_group'] = train['Podcast_Name'].map(podcast_bin_map).fillna('Other')
test['podcast_group'] = test['Podcast_Name'].map(podcast_bin_map).fillna('Other')

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Reshape to 2D array for encoder
train['podcast_encoded'] = encoder.fit_transform(train[['podcast_group']])
test['podcast_encoded'] = encoder.transform(test[['podcast_group']])



# Before removal
print("Train shape before removal:", train.shape)
print("Test shape before removal:", test.shape)

# Remove rows where Number_of_Ads > 10
train = train[train['Number_of_Ads'] <= 10]
test = test[test['Number_of_Ads'] <= 10]

# Reset index
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# After removal
print("Train shape after removal:", train.shape)
print("Test shape after removal:", test.shape)


# Remove rows where Episode_Length_minutes > 300
train = train[train['Episode_Length_minutes'] <= 300]
test=test[test['Episode_Length_minutes'] <= 300]
# Reset index
train = train.reset_index(drop=True)
test=test.reset_index(drop=True)
# Check the shape of the dataset after removal
print(f"Dataset shape after removing outliers: {train.shape}")



from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
train['Genre'] = Le.fit_transform(train['Genre'])
test['Genre'] = Le.transform(test['Genre'])

train['Episode_Sentiment'] = Le.fit_transform(train['Episode_Sentiment'])
test['Episode_Sentiment'] = Le.transform(test['Episode_Sentiment'])

train['Publication_Day'] = Le.fit_transform(train['Publication_Day'])
test['Publication_Day'] = Le.transform(test['Publication_Day'])

train['Publication_Time'] = Le.fit_transform(train['Publication_Time'])
test['Publication_Time'] = Le.transform(test['Publication_Time'])


# # remove columns from train and test of Podcast Name and Podcast Group becoause we alred encoded them
train = train.drop(['Podcast_Name', 'podcast_group'], axis=1)
test = test.drop(['Podcast_Name', 'podcast_group'], axis=1)
# drop Episode Title from train and test
train = train.drop(['Episode_Title'], axis=1)
test = test.drop(['Episode_Title'], axis=1)


# Remove rows where Episode_Length_minutes > 300
train = train[train['Host_Popularity_percentage'] <= 100]
test=test[test['Host_Popularity_percentage'] <= 100]
# Reset index
train = train.reset_index(drop=True)
test=test.reset_index(drop=True)
# Check the shape of the dataset after removal
print(f"Dataset shape after removing outliers: {train.shape}")
print(f"Dataset shape after removing outliers: {test.shape}")


# Remove rows where Episode_Length_minutes > 300
train = train[train['Guest_Popularity_percentage'] <= 100]
test=test[test['Guest_Popularity_percentage'] <= 100]
# Reset index
train = train.reset_index(drop=True)
test=test.reset_index(drop=True)
# Check the shape of the dataset after removal
print(f"Dataset shape after removing outliers: {train.shape}")



#  select X and y
target='Listening_Time_minutes'
train.drop(columns=['id'],inplace=True,errors='ignore')
X=train.drop(target,axis=1)
y=train[target]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor

# Re-initialize with best params
final_model = LGBMRegressor(learning_rate=0.1, n_estimators=50, num_leaves=31, verbose=-1)

# Fit on full training set
final_model.fit(X, y)


from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse:.4f}")



print(train.columns)
print(test.columns)




# Check the shape of the datasets
print("Training data shape:", train.shape)
print("Testing data shape:", test.shape)



# Load the Kaggle test data (without the target variable)

# test_data=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Drop the 'id' column (it will not be used for prediction)
X_test_kaggle = test.drop(columns=['id'])

# Generate predictions for Kaggle test set
y_pred_kaggle = final_model.predict(X_test_kaggle)

# Prepare the submission dataframe with 'id' and predicted 'Listening_Time_minutes'
submission = pd.DataFrame({
    'id': test['id'],  # 'id' from the test data
    'Listening_Time_minutes': y_pred_kaggle  # Predictions for 'Listening_Time_minutes'
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)



print(submission)




