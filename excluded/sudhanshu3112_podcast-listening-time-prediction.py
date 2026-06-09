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


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(train.head())



train.describe


train.info


test.head(3)


test.describe


test.isnull().sum()


sample.head()


# Extract numeric part from 'Episode_Title'
train['Episode_Number'] = train['Episode_Title'].str.extract('(\d+)').astype(float)
test['Episode_Number'] = test['Episode_Title'].str.extract('(\d+)').astype(float)

print(train[['Episode_Title', 'Episode_Number']].head())



# Fill missing Episode_Length
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean())
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].mean())

# Fill missing Guest_Popularity
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean())

# Verify no missing values
print("Train missing:\n", train.isnull().sum())
print("Test missing:\n", test.isnull().sum())



print(train['Genre'].unique())



print(train['Episode_Sentiment'].unique())


print(train['Publication_Time'].unique())


print(train['Publication_Day'].unique())


# Checking unique genres and their counts in the train dataset
genre_counts = train['Genre'].value_counts()

# Display the result
print(genre_counts)

# Plotting the genre distribution
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(train['Listening_Time_minutes'], kde=True, bins=50)
plt.title("Distribution of Listening Time")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Count")
plt.show()



import numpy as np

numeric_cols = train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='Genre', y='Listening_Time_minutes')
plt.title('Listening Time by Genre')
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(10, 5))
sns.scatterplot(data=train, x='Host_Popularity_percentage', y='Listening_Time_minutes')
plt.title('Host Popularity vs Listening Time')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=train, x='Guest_Popularity_percentage', y='Listening_Time_minutes')
plt.title('Guest Popularity vs Listening Time')
plt.show()



sns.boxplot(data=train, x='Publication_Day', y='Listening_Time_minutes')
plt.title('Listening Time by Day')
plt.show()

sns.boxplot(data=train, x='Publication_Time', y='Listening_Time_minutes')
plt.title('Listening Time by Time of Day')
plt.show()

sns.boxplot(data=train, x='Episode_Sentiment', y='Listening_Time_minutes')
plt.title('Listening Time by Episode Sentiment')
plt.show()



from sklearn.preprocessing import LabelEncoder

# List of categorical columns to encode
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Dictionary to store encoders for each column
encoders = {}

# Encode each column and store the encoder
for col in cat_cols:
    encoders[col] = LabelEncoder()  # Create a new encoder for this column
    train[col] = encoders[col].fit_transform(train[col])
    test[col] = encoders[col].transform(test[col])  # Use same encoder for test

# Print mapping of encoded values to original categories for each column
for col in cat_cols:
    print(f"{col} Mapping:")
    for i, label in enumerate(encoders[col].classes_):
        print(f"{i}: {label}")
    print()  # Add a blank line for readability

# Print first few rows of encoded columns
print("Encoded Train Data (First 5 Rows):")
print(train[cat_cols].head())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix with Listening_Time_minutes")
plt.show()






#Total Popularity (Host + Guest)
train['Total_Popularity'] = train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']



# Ads Per Minute (Ad density):
train['Ads_Per_Minute'] = train['Number_of_Ads'] / (train['Episode_Length_minutes'] + 1e-5)  # +1e-5 to avoid zero division



#Is Weekend (Saturday-Sunday detection):
train['Is_Weekend'] = train['Publication_Day'].apply(lambda x: 1 if x in [5, 6] else 0)



#Is Morning (Based on encoded Publication_Time, assume 0 = morning):

train['Is_Morning'] = train['Publication_Time'].apply(lambda x: 1 if x == 0 else 0)



# Apply same feature engineering to test set
# Total Popularity
test['Total_Popularity'] = test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']

# Ads Per Minute - handle division + inf + NaN cleanly
test['Ads_Per_Minute'] = test['Number_of_Ads'] / test['Episode_Length_minutes']
test['Ads_Per_Minute'] = test['Ads_Per_Minute'].replace([np.inf, -np.inf], 0)
test['Ads_Per_Minute'] = test['Ads_Per_Minute'].fillna(0)

# Is Weekend (Saturday=5, Sunday=6)
test['Is_Weekend'] = test['Publication_Day'].apply(lambda x: 1 if x in [5, 6] else 0)

# Is Morning (5 AM to 11 AM)
test['Is_Morning'] = test['Publication_Time'].apply(lambda x: 1 if 5 <= x <= 11 else 0)



features = [
    'Episode_Length_minutes',
    'Genre',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Sentiment',
    'Episode_Number',
    'Publication_Day',
    'Publication_Time',
    'Total_Popularity',          # engineered
    'Ads_Per_Minute',            # engineered
    'Is_Weekend',                # engineered
    'Is_Morning'                 # engineered
]

target = 'Listening_Time_minutes'



X = train[features]
y = train[target]



from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np

# Define parameter grid
param_grid = {
    'n_estimators': [100,200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Base model
xgb_model = XGBRegressor(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           scoring='neg_root_mean_squared_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

# Fit on training data
grid_search.fit(X, y)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)




final_model = XGBRegressor(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=8,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

final_model.fit(X, y)



preds = final_model.predict(test[features])



sample.head(5)



# Step 1: Copy sample submission
df_sub = sample.copy()

# Step 2: Insert predictions
df_sub['Listening_Time_minutes'] = preds   # ya test_preds, jo bhi variable hai tere predictions ka

# Step 3: Save as submission.csv
df_sub.to_csv('submission.csv', index=False)

# Step 4: (Optional) Check the first few rows
df_sub.head()







