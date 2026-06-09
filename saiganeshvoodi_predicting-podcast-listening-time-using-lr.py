import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


def preprocess_data(train, test):
    train['is_train'] = 1
    test['is_train'] = 0
    test['Listening_Time_minutes'] = np.nan
    full_data = pd.concat([train, test], axis=0)
    full_data.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)
    full_data['Episode_Length_minutes'] = full_data['Episode_Length_minutes'].fillna(full_data['Episode_Length_minutes'].median())
    full_data['Guest_Popularity_percentage'] = full_data['Guest_Popularity_percentage'].fillna(full_data['Guest_Popularity_percentage'].median())
    full_data['Number_of_Ads'] = full_data['Number_of_Ads'].fillna(full_data['Number_of_Ads'].mode()[0])
    cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for col in cat_cols:
        le = LabelEncoder()
        full_data[col] = le.fit_transform(full_data[col].astype(str))
    train_clean = full_data[full_data['is_train'] == 1].drop('is_train', axis=1)
    test_clean = full_data[full_data['is_train'] == 0].drop(['is_train', 'Listening_Time_minutes'], axis=1)
    return train_clean, test_clean


train_clean, test_clean = preprocess_data(train, test)
X = train_clean.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train_clean['Listening_Time_minutes']
X_test = test_clean.drop(['id'], axis=1)


# Train Multiple Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X, y)

# Predict
preds = mlr_model.predict(X_test)



# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv("submission.csv", index=False)


submission.isnull().sum()




