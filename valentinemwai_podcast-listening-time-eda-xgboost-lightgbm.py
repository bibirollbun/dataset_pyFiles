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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(train.head())
print(test.head())


(train.info())


print(train.isnull().sum())
print(test.isnull().sum())


#dropping id column
train_df=train.copy()
train_df=train.drop(columns=['id'])
test_df=test.copy()
test_df=test_df.drop(columns=['id'])


train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)
test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)



print(train_df.isnull().sum())
print(test_df.isnull().sum())


#Numerical columns and Summary statistics
num_col=train_df.select_dtypes(include=["number"]).columns
print(num_col)
train_df[num_col].describe()


#Correlation Matrix and plot
corr_matrix= train_df[num_col].corr()
print(corr_matrix)
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, cmap="coolwarm",annot=True)
plt.show()


#Boxplot Numerical Variables
plt.figure(figsize=(10,6))
sns.boxplot(train_df[num_col])
plt.xticks(rotation=45)
plt.title("Boxplot for Numerical Variables")
plt.show()


sns.pairplot(train_df[num_col])
plt.show()


#Categorical variables 
cat_col = train_df.select_dtypes(include="object").columns
print(cat_col)
cat_col_test = test_df.select_dtypes(include="object").columns
print(cat_col_test)


#Count plot-Episode Sentiment
plt.figure(figsize=(12,6))
sns.countplot(x='Episode_Sentiment', data=train_df,hue='Episode_Sentiment')
plt.xlabel('Episode Sentiment')
plt.show()


#Count plot- Publication Day
plt.figure(figsize=(12,6))
sns.countplot(x='Publication_Day', data=train_df,hue='Publication_Day')
plt.title('Count plot of Publication Day')
plt.show()


#Count plot- Publication Time
plt.figure(figsize=(12,6))
sns.countplot(x='Publication_Time', data=train_df,hue='Publication_Time')
plt.title('Count plot of Publication Time')
plt.show()


#Count plot
plt.figure(figsize=(12,6))
sns.countplot(x='Genre', data=train_df,hue='Genre')
plt.title('Count plot of Genre')
plt.tight_layout()
plt.show()


# Calculate counts and get top 10 podcast
top10 = train_df['Podcast_Name'].value_counts().iloc[:10].index

# Create the count plot for the top 10 podcasts
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Podcast_Name', order=top10, palette='viridis')
plt.title('Top 10 Most Frequent Podcasts')
plt.xlabel('Postcard Names')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



train_df[cat_col] = train_df[cat_col].astype('category')
test_df[cat_col_test] = test_df[cat_col_test].astype('category')


cat_col


#Splitting the data
X = train_df.drop(columns=['Listening_Time_minutes'])  # Features
y = train_df['Listening_Time_minutes']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import xgboost as xgb


model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100, 
                         learning_rate=0.1, 
                         max_depth=5, 
                         tree_method="hist",  
                         enable_categorical=True)  

# Train the model
model.fit(X_train, y_train)


#Evaluating the model
# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")



#Using the test dataset
pred_xgb = model.predict(test_df)
pred_xgb


#Submission
sub['Listening_Time_minutes'] = pred_xgb
sub.to_csv('submission.csv', index=False)
sub.head()


import lightgbm as lgb


best_params = {
    'max_depth': 10,
    'n_estimators': 5000,
    'learning_rate': 0.03379119657569082,
    'colsample_bytree': 0.6588612968138808,
    'subsample': 0.8967584873358806,
    'min_child_weight': 4,
    'gamma': 1.6318053453600387,
    'reg_alpha':  5.521023013284561,
    'reg_lambda': 7.849683124657393
}
lgbm_model = lgb.LGBMRegressor(**best_params,
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0)

# Fit the model to the training data
lgbm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lgbm_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) to evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Optional: Check feature importance
import matplotlib.pyplot as plt
lgb.plot_importance(lgbm_model)
plt.show()


#Using the test dataset
pred_lgbm = lgbm_model.predict(test_df)
pred_lgbm


#Submission
sub['Listening_Time_minutes'] = pred_lgbm
sub.to_csv('submission.csv', index=False)
sub.head()

