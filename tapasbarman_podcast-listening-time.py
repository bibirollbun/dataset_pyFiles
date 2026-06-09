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
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


train.head(1)


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


test.head(1)


train.info()


train.isnull().sum()


train.set_index('id', inplace=True)
test.set_index('id', inplace=True)

train.head(1)


def fill_missing_val (df, column):
    df[column] = df.groupby(['Podcast_Name', 'Publication_Day', 'Publication_Time'])[column].transform(lambda x: x.fillna(x.median()))
    return 

fill_missing_val(train, 'Episode_Length_minutes')
fill_missing_val(test, 'Episode_Length_minutes')


train['Guest_Popularity_percentage'].sample(10)


train['Guest_Popularity_percentage'].fillna(0,inplace = True)
test['Guest_Popularity_percentage'].fillna(0,inplace = True)


train.isnull().sum()


train['Number_of_Ads'].fillna(0,inplace = True)
test['Number_of_Ads'].fillna(0,inplace = True)


train.duplicated().sum()
test.duplicated().sum()


train.columns


train.describe()


train['Number_of_Ads'].value_counts()


print(train['Genre'].value_counts())
sns.countplot(data=train, x='Genre')
plt.xticks(rotation=45)
plt.show()


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
train[num_cols].hist(bins=20, figsize=(12, 8))



sns.boxplot(data=train, x='Listening_Time_minutes')



sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm')
plt.xticks(rotation=45)
plt.show()



sns.barplot(data=train, x='Publication_Day', y='Listening_Time_minutes')


sns.boxplot(data=train, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.show()


train.columns


train.info()


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score


train = train.drop(['Episode_Title', 'Podcast_Name'], axis=1)
test = test.drop(['Episode_Title', 'Podcast_Name'], axis=1)


train = pd.get_dummies(train, columns=['Genre', 'Episode_Sentiment', 'Publication_Day'], drop_first=True)


test = pd.get_dummies(test, columns=['Genre', 'Episode_Sentiment', 'Publication_Day'], drop_first=True)



bool_cols = train.select_dtypes(include='bool').columns
train[bool_cols] = train[bool_cols].astype(int)

bool_cols = test.select_dtypes(include='bool').columns
test[bool_cols] = test[bool_cols].astype(int)


train.head(5)


train['Publication_Time'].unique()


from sklearn.preprocessing import OrdinalEncoder

time_order = [['Morning', 'Evening', 'Afternoon', 'Night']]
encoder = OrdinalEncoder(categories=time_order)
train['Publication_Time_Encoded'] = encoder.fit_transform(train[['Publication_Time']]).astype(int)
train = train.drop('Publication_Time', axis=1)



from sklearn.preprocessing import OrdinalEncoder

time_order = [['Morning', 'Evening', 'Afternoon', 'Night']]
encoder = OrdinalEncoder(categories=time_order)
test['Publication_Time_Encoded'] = encoder.fit_transform(test[['Publication_Time']]).astype(int)
test = test.drop('Publication_Time', axis=1)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
            'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time_Encoded']

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.fit_transform(test[num_cols])





X = train.drop(columns = ['Listening_Time_minutes'])
y = train['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd
# from xgboost import XGBRegressor

# # Assume you have already defined: X_train, y_train, test

# # Define model dictionary
# models = {
#     'XG Boost': XGBRegressor(),
#     'Random Forest': RandomForestRegressor(),
#     'Gradient Boosting': GradientBoostingRegressor()
# }

# # Evaluate models using cross-validation
# rmse_scores = {}

# for name, model in models.items():
#     neg_mse = cross_val_score(model, X_train, y_train, 
#                               scoring='neg_mean_squared_error', cv=2)
#     rmse = np.sqrt(-neg_mse.mean())
#     rmse_scores[name] = rmse

# # Print RMSE scores
# for model_name, rmse in rmse_scores.items():
#     print(f"{model_name}: RMSE = {rmse:.4f}")

# # Train models
# xgb_model = XGBRegressor()
# rf_model = RandomForestRegressor()
# gb_model = GradientBoostingRegressor()

# xgb_model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)
# gb_model.fit(X_train, y_train)

# # Make predictions
# y_pred_xgb = xgb_model.predict(test)
# y_pred_rf = rf_model.predict(test)
# y_pred_gb = gb_model.predict(test)

# # Define weights (make sure o + m + n = 1 for a proper weighted average)
# o = 0.4
# m = 0.3
# n = 0.3

# # Final ensemble prediction
# final_preds = (np.array(y_pred_xgb) * o + 
#                np.array(y_pred_rf) * m + 
#                np.array(y_pred_gb) * n)

# # Load test data for submission
# test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# # Prepare submission
# submission = pd.DataFrame({
#     'id': test_df['id'],
#     'prediction': final_preds
# })

# submission.to_csv('submission.csv', index=False)






from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Assume you have already defined: X_train, y_train, test

# Define model dictionary including CatBoost and LightGBM
models = {
    'XG Boost': XGBRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'LightGBM': LGBMRegressor()
}

# Evaluate models using cross-validation
rmse_scores = {}

for name, model in models.items():
    neg_mse = cross_val_score(model, X_train, y_train, 
                              scoring='neg_mean_squared_error', cv=2)
    rmse = np.sqrt(-neg_mse.mean())
    rmse_scores[name] = rmse

# Print RMSE scores
for model_name, rmse in rmse_scores.items():
    print(f"{model_name}: RMSE = {rmse:.4f}")

# Train models
xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()
cat_model = CatBoostRegressor(verbose=0)
lgbm_model = LGBMRegressor()

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(test)
y_pred_rf = rf_model.predict(test)
y_pred_gb = gb_model.predict(test)
y_pred_cat = cat_model.predict(test)
y_pred_lgbm = lgbm_model.predict(test)

# Define weights (should sum to 1)
weights = {
    'xgb': 0.25,
    'rf': 0.2,
    'gb': 0.2,
    'cat': 0.2,
    'lgbm': 0.15
}

# Final ensemble prediction
final_preds = (
    y_pred_xgb * weights['xgb'] +
    y_pred_rf * weights['rf'] +
    y_pred_gb * weights['gb'] +
    y_pred_cat * weights['cat'] +
    y_pred_lgbm * weights['lgbm']
)

# Load test data for submission
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': final_preds
})

submission.to_csv('submission.csv', index=False)



submission




