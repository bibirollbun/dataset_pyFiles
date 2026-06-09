# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


file_path = '../input/playground-series-s5e4/train.csv'
test_path = '../input/playground-series-s5e4/test.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(file_path) 
test_data = pd.read_csv(test_path) 
sample_submission = pd.read_csv('../input/playground-series-s5e4/sample_submission.csv')



day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
           'Thursday': 4, 'Friday': 5, 'Saturday': 6}
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
podc_map = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
genr_map = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}

def preprocessing(df):
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode', '', regex=False).astype(int)
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    df['Have_Multiple_Ads'] = (df['Number_of_Ads'] > 1).astype(int)
    df['Length_Bucket'] = pd.cut(df['Episode_Length_minutes'],
                                  bins=[0, 30, 60, 90, 200],
                                  labels=[0, 30, 60, 90],
                                  right=True,
                                  include_lowest=True)    

    return df


data = preprocessing(data)
test_data = preprocessing(test_data)
print(data.head())


# Create target object and call it y
y = data.Listening_Time_minutes
# Create X
features = [
    'Episode_Length_minutes',
    'Have_Multiple_Ads',
    'Number_of_Ads', 
    'Length_Bucket',
    'Episode_Sentiment',
    'Host_Popularity_percentage', 
    'Guest_Popularity_percentage', 
    'Is_Long_Episode', 
    'Publication_Day', 
    'Publication_Time', 
    'Genre', 
    'Podcast_Name',
    'Episode_Title',
]

# Use .copy() to avoid potential SettingWithCopyWarning later
X = data[features].copy()
test_X = test_data[features].copy()


columns_to_impute = ['Episode_Length_minutes', 'Number_of_Ads', 'Length_Bucket', 'Guest_Popularity_percentage'] 

print(f"Imputing missing values in columns: {columns_to_impute}")

imputer_X = SimpleImputer(strategy='median')
imputer_X.fit(X[columns_to_impute])

X_imputed_subset_array = imputer_X.transform(X[columns_to_impute])
test_X_imputed_subset_array = imputer_X.transform(test_X[columns_to_impute])

X[columns_to_impute] = X_imputed_subset_array
test_X[columns_to_impute] = test_X_imputed_subset_array

if y.isnull().any():
    print(f"\nImputing {y.isnull().sum()} missing values in target 'y' using median.")
    y_median = y.median()
    y = y.fillna(y_median) # Use fillna for Series
    print(f"NaN check in y after imputation: {y.isnull().sum()}")


best_params = {
    'learning_rate': 0.025,
    'max_depth': 25,
    'min_child_weight': 5,
    'subsample': 0.666496972140272,
    'colsample_bytree': 0.6379829756397679,
    'gamma': 3.663052996753313,
    'reg_alpha': 9.089264890544545,
    'reg_lambda': 9.072490097887034,
    'random_state': 42,
    'objective': 'reg:squarederror',
}

# Split into validation and training data 
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.05, random_state=1)

# Define the XGBoost model
# Start with some reasonable parameters
dtrain = xgb.DMatrix(train_X, label=train_y, enable_categorical=True)
dval = xgb.DMatrix(val_X, label=val_y, enable_categorical=True)

evals_result = {}
xgb_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=1500,
    evals=[(dtrain, "train"), (dval, "valid")],
    early_stopping_rounds=100,
    evals_result=evals_result,
    verbose_eval=0
)


# Make predictions on the validation set
val_predictions = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration)) # Using the DMatrix approach

# Calculate the Validation RMSE (Competition Metric)
rmse = np.sqrt(mean_squared_error(val_y, val_predictions))

print("\nValidation RMSE for XGBoost Model: {:,.4f}".format(rmse))


dtest = xgb.DMatrix(test_X, enable_categorical=True) # Use test_X_processed (imputed test features)
preds = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration)) # Predict on the full test set DMatrix. 

# Use the original test_data DataFrame to get the full list of test IDs (750,000 rows)
all_test_preds_df = pd.DataFrame({
    "id": test_data['id'], # Original 750,000 IDs
    "Listening_Time_minutes": preds # Predictions for those 750,000 IDs
})

# Merge with sample_submission to get only the required 250,000 IDs
submission_df = sample_submission[['id']].merge(all_test_preds_df, on='id', how='left')

# Save the submission file
submission_df.to_csv("submission_ensemble.csv", index=False) # Using 'submission_ensemble.csv' as in your notebook

