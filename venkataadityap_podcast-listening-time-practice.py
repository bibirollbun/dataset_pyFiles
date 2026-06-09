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


train_data=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col=0)
test_data=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col=0)


train_data.head()


train_data.info()


y=train_data.Listening_Time_minutes
X=train_data.drop(['Listening_Time_minutes'],axis=1)


y.head()


X.head()


X.shape


X.describe()


y.describe()


from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)


X_train.head()


X_train.info()


from sklearn.impute import SimpleImputer

mean_imputer_ads=SimpleImputer(strategy='mean')
X_train['Number_of_Ads']=mean_imputer_ads.fit_transform(X_train[['Number_of_Ads']]).flatten()
X_valid['Number_of_Ads']=mean_imputer_ads.transform(X_valid[['Number_of_Ads']]).flatten()


median_imputer=SimpleImputer(strategy='median')
X_train[['Episode_Length_minutes','Guest_Popularity_percentage']]=median_imputer.fit_transform(X_train[['Episode_Length_minutes','Guest_Popularity_percentage']])
X_valid[['Episode_Length_minutes','Guest_Popularity_percentage']]=median_imputer.transform(X_valid[['Episode_Length_minutes','Guest_Popularity_percentage']])


X_train.info()


print(f"Missing values after imputation in X_train: {X_train[['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']].isnull().sum()}")


print(f"Missing values after imputation in X_valid: {X_valid[['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']].isnull().sum()}")


X_train.nunique()


X_train.dtypes


data_summary=pd.DataFrame({'Unique_Count':X_train.nunique(),'Data_Types':X_train.dtypes})
data_summary


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

ordinal_cols=['Episode_Sentiment']
nominal_cols=['Genre','Publication_Day','Publication_Time']

sentiment_categories = [['Negative', 'Neutral', 'Positive']]

ordinal_encoder=OrdinalEncoder(categories=sentiment_categories)
X_train['Episode_Sentiment']=ordinal_encoder.fit_transform(X_train[['Episode_Sentiment']])
X_valid['Episode_Sentiment']=ordinal_encoder.transform(X_valid[['Episode_Sentiment']])


X_train.head()


onehot_encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first')
encoded_cols_train=onehot_encoder.fit_transform(X_train[nominal_cols])
encoded_cols_valid=onehot_encoder.transform(X_valid[nominal_cols])


encoded_cols_train


encoded_feature_names=onehot_encoder.get_feature_names_out(nominal_cols)


encoded_feature_names


encoded_df_train=pd.DataFrame(encoded_cols_train,index=X_train.index,columns=encoded_feature_names)
encoded_df_valid=pd.DataFrame(encoded_cols_valid,index=X_valid.index,columns=encoded_feature_names)


encoded_df_train.head()


print("Shape of encoded_cols_train:", encoded_cols_train.shape)
print("Shape of X_train index:", X_train.index.shape)
print("Shape of encoded_cols_valid:", encoded_cols_valid.shape)
print("Shape of X_valid index:", X_valid.index.shape)
print("Number of encoded features:", len(encoded_feature_names))


X_train=X_train.drop(nominal_cols,axis=1)
X_valid=X_valid.drop(nominal_cols,axis=1)

X_train=pd.concat([X_train,encoded_df_train],axis=1)
X_valid=pd.concat([X_valid,encoded_df_valid],axis=1)


X_train.head()


X_valid.head()





data_summary=pd.DataFrame({'Unique_Count':X_train.nunique(),'Data_Types':X_train.dtypes})
data_summary


columns_to_drop=['Podcast_Name','Episode_Title']
X_train=X_train.drop(columns_to_drop,axis=1)
X_valid=X_valid.drop(columns_to_drop,axis=1)


data_summary=pd.DataFrame({'Unique_Count':X_train.nunique(),'Data_Types':X_train.dtypes})
data_summary


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def model_score(n_est,X_train,X_valid,y_train,y_valid):
    model=RandomForestRegressor(n_estimators=n_est,random_state=0,n_jobs=-1)
    model.fit(X_train,y_train)
    preds=model.predict(X_valid)
    return mean_absolute_error(y_valid,preds)

mae_nest_10=model_score(10,X_train,X_valid,y_train,y_valid)
print(f"MAE with n_estimators = 10 is : {mae_nest_10:.2f}")


#n_estimators_values = [10, 100, 1000]
#results = {}

#for n_est in n_estimators_values:
    #mae = model_score(n_est, X_train, X_valid, y_train, y_valid)
   # results[f'n_estimators_{n_est}'] = mae
    #print(f"MAE with n_estimators = {n_est}: {mae:.2f}")


y.describe()


# # from xgboost import XGBRegressor

# xgb_model_1=XGBRegressor(n_estimators=100,learning_rate=0.1,n_jobs=-1,random_state=0)

# xgb_model_1.fit(X_train,y_train)
# preds=xgb_model_1.predict(X_valid)
# print(f"Mean Absolute error with initial XGB model: {mean_absolute_error(y_valid,preds)}")


# xgb_model_2=XGBRegressor(n_estimators=1000,learning_rate=0.1,n_jobs=-1,random_state=0)

# xgb_model_2.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],early_stopping_rounds=50,verbose=False)
# preds=xgb_model_2.predict(X_valid)
# print(f"MAE with early stopping: {mean_absolute_error(y_valid,preds)}")
# print(f"Best number of boosting rounds: {xgb_model_2.best_iteration}")


# #xgb_model_2=XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=-1,random_state=0)

# xgb_model_2.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],early_stopping_rounds=75,verbose=False)
# preds=xgb_model_2.predict(X_valid)
# print(f"MAE with early stopping: {mean_absolute_error(y_valid,preds)}")
# print(f"Best number of boosting rounds: {xgb_model_2.best_iteration}")


test_data['Number_of_Ads'] = mean_imputer_ads.transform(test_data[['Number_of_Ads']]).flatten()


test_data[['Episode_Length_minutes','Guest_Popularity_percentage']]=median_imputer.transform(test_data[['Episode_Length_minutes','Guest_Popularity_percentage']])


test_data.head()


test_data.info()


test_data['Episode_Sentiment']=ordinal_encoder.transform(test_data[['Episode_Sentiment']])


test_data.head()


encoded_cols_test = onehot_encoder.transform(test_data[nominal_cols])
encoded_df_test = pd.DataFrame(encoded_cols_test, index=test_data.index, columns=encoded_feature_names)
test_data = test_data.drop(nominal_cols, axis=1)
test_data = pd.concat([test_data, encoded_df_test], axis=1)


test_data.head()


columns_to_drop = ['Podcast_Name', 'Episode_Title']
test_data = test_data.drop(columns_to_drop, axis=1)


test_data.info()


X_train.info()


# rf_final_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
# rf_final_model.fit(X, y)


X['Number_of_Ads'] = mean_imputer_ads.transform(X[['Number_of_Ads']]).flatten()
X[['Episode_Length_minutes','Guest_Popularity_percentage']]=median_imputer.transform(X[['Episode_Length_minutes','Guest_Popularity_percentage']])


X['Episode_Sentiment']=ordinal_encoder.transform(X[['Episode_Sentiment']])


nominal_cols = ['Genre', 'Publication_Day', 'Publication_Time']
encoded_cols_X = onehot_encoder.transform(X[nominal_cols])
encoded_df_X = pd.DataFrame(encoded_cols_X, index=X.index, columns=encoded_feature_names)
X = X.drop(nominal_cols, axis=1)
X = pd.concat([X, encoded_df_X], axis=1)


columns_to_drop = ['Podcast_Name', 'Episode_Title']
X = X.drop(columns_to_drop, axis=1)


X.head()


X.info()


rf_final_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
rf_final_model.fit(X, y)


test_predictions = rf_final_model.predict(test_data)


submission = pd.DataFrame({'id': test_data.index, 'Listening_Time_minutes': test_predictions})
print(submission.head(10))


submission.to_csv('submission.csv', index=False)




