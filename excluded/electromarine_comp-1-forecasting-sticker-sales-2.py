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

import warnings
warnings.filterwarnings("ignore")



train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
train_df.head()


print(f"The train dataset has",train_df.shape[0], "rows and", train_df.shape[1],"columns")


# Understanding the data type of each variable
train_df.info()


100*train_df.isnull().mean()


train_df['num_sold'].fillna(0,inplace=True)


100*train_df.isnull().mean()


# Feature engineering: Creating new variable to store day, month & yearr from date column 

# Converting date to Pandas datetime format
train_df['date'] = pd.to_datetime(train_df['date'])


train_df['day'] = train_df['date'].dt.day
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year

# droping date column
train_df.drop('date',axis=1,inplace=True)


train_df.head()


# Let's do descriptive analysis of the target variable- num_sold
train_df.num_sold.describe()


sns.pairplot(train_df)
plt.show()


train_df.store.value_counts()


train_df['product'].value_counts()


train_df.country.value_counts()


train_df["country"] = train_df["country"].map({"Canada":1,"Finland":2,"Italy":3, "Kenya":4,"Norway":5,"Singapore":6})
train_df["product"] = train_df["product"].map({"Holographic Goose":1, "Kaggle":2, "Kaggle Tiers":3, "Kerneler":4, "Kerneler Dark Mode":5})
train_df["store"] = train_df["store"].map({"Discount Stickers":1,"Stickers for Less":2, "Premium Sticker Mart":3})


train_df.info()


train_df.head()


from sklearn.model_selection import train_test_split

np.random.seed(0)

df_train, df_test = train_test_split(train_df,train_size=0.7,random_state=100)



df_train.shape, df_test.shape


from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()

df_train['num_sold'] =scaler.fit_transform(df_train[['num_sold']])
df_test['num_sold'] =scaler.transform(df_test[['num_sold']])


df_train.num_sold.describe()


df_test.num_sold.describe()


y_train=df_train.pop("num_sold")
X_train = df_train


y_test = df_test.pop("num_sold")
X_test = df_test


y_train.shape, y_test.shape


y_train.isnull().sum()


from sklearn.ensemble import RandomForestRegressor 
rf=RandomForestRegressor()


from sklearn.model_selection import GridSearchCV

# parameters to tune
params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV for regression
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=params, 
                           cv=4, 
                           n_jobs=-1, 
                           verbose=1, 
                           scoring={
                               "r2": "r2",
                               "neg_root_mean_squared_error":"neg_root_mean_squared_error"},
                          refit="r2") 


grid_search.fit(X_train, y_train)


score_df =pd.DataFrame(grid_search.cv_results_)
score_df.head()


score_df.nlargest(5,"mean_test_r2")


grid_search.best_score_ # avg score for different folds


grid_search.best_estimator_


dt_best = grid_search.best_estimator_


from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error


def evaluate_model(dt_classifier):
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    print("Train set performance")
    print("R-Squared:",r2_score(y_train, y_train_pred))
    print("MSE:",mean_squared_error(y_train, y_train_pred))
    print("MSELog:",mean_squared_log_error(y_train, y_train_pred))
    print("*"*50)
    print("Test set performance")
    print("R-Squared:",r2_score(y_test, y_test_pred)) 
    print("MSELog:",mean_squared_error(y_test, y_test_pred))
    print(mean_squared_log_error(y_test, y_test_pred))


evaluate_model(dt_best)


test_df.head()


test_df.shape


# Converting date to Pandas datetime format
test_df['date'] = pd.to_datetime(test_df['date'])


test_df['day'] = test_df['date'].dt.day
test_df['month'] = test_df['date'].dt.month
test_df['year'] = test_df['date'].dt.year

# droping date column
test_df.drop('date',axis=1,inplace=True)


test_df.head()


# imputing numerical values to replace categorical values 

test_df["country"] = test_df["country"].map({"Canada":1,"Finland":2,"Italy":3, "Kenya":4,"Norway":5,"Singapore":6})
test_df["product"] = test_df["product"].map({"Holographic Goose":1, "Kaggle":2, "Kaggle Tiers":3, "Kerneler":4, "Kerneler Dark Mode":5})
test_df["store"] = test_df["store"].map({"Discount Stickers":1,"Stickers for Less":2, "Premium Sticker Mart":3})


test_df.head()


predictions = dt_best.predict(test_df)


# Creating a DataFrame with 'id' and 'num_sold'
submission_df = pd.DataFrame({
    'id': test_df['id'],  
    'num_sold': predictions
})

# Saving the DataFrame as an .csv file
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

