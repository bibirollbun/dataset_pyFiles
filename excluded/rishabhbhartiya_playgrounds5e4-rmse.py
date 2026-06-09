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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


print(f"The Shape of Train Data: {train_df.shape}")
print(f"The Shape of Test Data: {test_df.shape}")


print(f"The Columns of Train Data: {train_df.columns}")
print(f"The Columns of Test Data: {test_df.columns}")


train_df.info()


# Extract numeric values
train_df['Episode_Title'] = train_df['Episode_Title'].str.extract('(\d+)').astype(int)


train_df.info()


missing_percentage = train_df.isnull().sum() / len(train_df) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
print(missing_percentage)


train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Number_of_Ads'].fillna(float(train_df['Number_of_Ads'].mode()[0]), inplace=True)


train_df.isnull().sum()


train_df.info()


cat_cols = train_df.select_dtypes(include ="object")
for col in cat_cols:
    print(f"The {col} has {train_df[col].nunique()} : {train_df[col].unique()}")


train_df['Podcast_Name'] = train_df['Podcast_Name'].map(train_df['Podcast_Name'].value_counts() / len(train_df))


train_df['Podcast_Name']


train_df.info()


train_df = pd.get_dummies(train_df, columns=['Genre'], dtype='int8')

day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
               'Friday': 5, 'Saturday': 6, 'Sunday': 7}
train_df['Publication_Day'] = train_df['Publication_Day'].map(day_mapping).astype('int8')

time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
train_df['Publication_Time'] = train_df['Publication_Time'].map(time_mapping).astype('int8')

sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_mapping).astype('int8')


train_df.info()


train_df.corr()["Listening_Time_minutes"]*100


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = train_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



train_df.columns


plt.figure(figsize=(10, 5))
train_df.corr()['Listening_Time_minutes'].sort_values().plot(kind='barh', cmap='coolwarm')
plt.title("Feature Correlation with Target Variable")
plt.show()


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

X = train_df.drop(columns=['Listening_Time_minutes'])  # Features
y = train_df['Listening_Time_minutes']  # Target

# Use a simple model (Linear Regression)
model = LinearRegression()

# Recursive Feature Elimination
rfe = RFE(estimator=model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Get selected features
selected_features = X.columns[rfe.support_]
print("Selected Features by RFE:", selected_features)



from sklearn.model_selection import train_test_split

X = train_df[['Episode_Length_minutes', 'Number_of_Ads', 'Episode_Sentiment',
           'Genre_Comedy', 'Genre_Lifestyle', 'Genre_News', 'Genre_Sports',
           'Genre_Technology', 'Genre_True Crime']]
y = train_df['Listening_Time_minutes']  # Replace with your actual target column name

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



!pip install xgboost


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=200, 
    max_depth=5, 
    learning_rate=0.1, 
    tree_method='hist',  # Change from 'gpu_hist' to 'hist'
    n_jobs=-1  # Use all CPU cores for faster processing
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_val, y_pred_xgb))
print("MSE:", mean_squared_error(y_val, y_pred_xgb))
print("R² Score:", r2_score(y_val, y_pred_xgb))


rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))  # MSE from your XGBRegressor
print("RMSE (XGB Regressor):", rmse_xgb)


test_df




