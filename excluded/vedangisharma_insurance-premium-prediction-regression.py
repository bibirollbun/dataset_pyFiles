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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessionos.path.join(dirname, filename))


kaggle_dir='/kaggle/input'
train_df=pd.read_csv(os.path.join(kaggle_dir, dirname, filenames[1]))
test_df=pd.read_csv(os.path.join(kaggle_dir, dirname, filenames[2]))


train_df.head(10)


test_df


train_df.info()


test_df.info()


df=pd.concat([train_df, test_df], axis=0).reset_index()


df.head()


df.drop('index', inplace=True, axis=1)


df


import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline 


num_cols = df.select_dtypes(np.number).columns.to_list()
cat_cols=df.select_dtypes('object').columns.to_list()


len(num_cols)


cat_cols


sns.boxplot(y=df['Age'],x=df['Gender'])


sns.boxplot(y=df['Annual Income'],x=df['Gender'])


df[num_cols[3]].value_counts()


sns.histplot(x=df[num_cols[4]])


sns.boxplot(y=df[num_cols[4]],x=df['Gender'])


df[num_cols[5]].value_counts()


sns.histplot(y=df[num_cols[6]])


sns.histplot(x=df[num_cols[7]])


df[num_cols[8]].value_counts()


sns.histplot(x=df['Premium Amount'])


cat_cols


sns.boxplot(y=df['Premium Amount'], x=df['Education Level'])


sns.boxplot(y=df['Premium Amount'], x=df['Marital Status'])


sns.boxplot(y=df['Premium Amount'], x=df['Location'])


sns.countplot(x=df['Policy Type'])


df['Policy Start Date']


df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
train_df['Policy Start Date']=pd.to_datetime(train_df['Policy Start Date'])
test_df['Policy Start Date']=pd.to_datetime(test_df['Policy Start Date'])


max(df['Policy Start Date'])


min(df['Policy Start Date'])


df['Year']=df['Policy Start Date'].dt.year
df['Month']=df['Policy Start Date'].dt.month 
df['Day']=df['Policy Start Date'].dt.day
train_df['Year']=train_df['Policy Start Date'].dt.year
train_df['Month']=train_df['Policy Start Date'].dt.month 
train_df['Day']=train_df['Policy Start Date'].dt.day
test_df['Year']=test_df['Policy Start Date'].dt.year
test_df['Month']=test_df['Policy Start Date'].dt.month 
test_df['Day']=df['Policy Start Date'].dt.day


df['Year'].value_counts()


df['Month'].value_counts()


df['Occupation'].value_counts()


sns.countplot(x=df['Customer Feedback'])


sns.countplot(x=df['Exercise Frequency'])


sns.countplot(x=df['Smoking Status'])


sns.countplot(x=df['Property Type'])


sns.heatmap(df[num_cols].corr())


from sklearn.impute import SimpleImputer


mean_imputer = SimpleImputer(strategy='mean')


train_df[num_cols].isna().sum()


test_df[num_cols[:-1]].isna().sum()


mean_imputer.fit(train_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']])


train_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']]=mean_imputer.transform(train_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']])


mean_imputer.fit(test_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']])


test_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']]=mean_imputer.transform(test_df[['Age', 'Number of Dependents', 'Health Score' ,'Previous Claims','Vehicle Age' ,'Credit Score' ,'Insurance Duration']])


median_imputer= SimpleImputer(strategy='median')


median_imputer.fit(pd.DataFrame(train_df['Annual Income']))


train_df['Annual Income'] = median_imputer.transform(pd.DataFrame(train_df['Annual Income']))


median_imputer.fit(pd.DataFrame(test_df['Annual Income']))


test_df['Annual Income'] = median_imputer.transform(pd.DataFrame(test_df['Annual Income']))


train_df[cat_cols].isna().sum()


test_df[cat_cols].isna().sum()


train_df['Customer Feedback'].value_counts()


train_df['Customer Feedback']= pd.DataFrame(train_df['Customer Feedback']).fillna('Average')
test_df['Customer Feedback']= pd.DataFrame(test_df['Customer Feedback']).fillna('Average')


train_df['Marital Status']= pd.DataFrame(train_df['Marital Status']).fillna(train_df['Marital Status'].mode())
test_df['Marital Status']= pd.DataFrame(test_df['Marital Status']).fillna(test_df['Marital Status'].mode())


train_df['Occupation']= pd.DataFrame(train_df['Occupation']).fillna(train_df['Occupation'].mode())
test_df['Occupation']= pd.DataFrame(test_df['Occupation']).fillna(test_df['Occupation'].mode())


train_df.columns


train_df.select_dtypes(include = ['int64','float64']).columns.tolist()


# train_df['Year']  = train_df['Year'].astype('int')
train_df['Month']  = train_df['Month'].astype('int')
train_df['Day']  = train_df['Day'].astype('int')


num_cols = train_df.select_dtypes(include = ['int64','float64']).columns.tolist()


num_cols.remove('Premium Amount')


num_cols.remove('id')


num_cols


cat_cols.remove('Policy Start Date')


cat_cols


from sklearn.preprocessing import MinMaxScaler


#Create the scaler
Scaler = MinMaxScaler()


Scaler.fit(train_df[num_cols])


train_df[num_cols] = Scaler.transform(train_df[num_cols])


train_df[num_cols].describe().loc[['min','max']]


test_df[num_cols] = Scaler.transform(test_df[num_cols])


train_df[cat_cols].nunique().sort_values(ascending=False)


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


encoder.fit(train_df[cat_cols])


encoded_cols = list(encoder.get_feature_names_out(cat_cols))


train_df[encoded_cols]=encoder.transform(train_df[cat_cols])


test_df[encoded_cols]=encoder.transform(test_df[cat_cols])


train_df


from sklearn.model_selection import train_test_split


X_train, val_train, X_targets, val_targets = train_test_split(train_df[num_cols + encoded_cols],
                                                                        train_df['Premium Amount'],
                                                                        test_size=0.2,
                                                                        random_state=28)


X_targets


from sklearn.linear_model import LinearRegression


LR_model = LinearRegression ()


LR_model.fit(X_train, X_targets)


X_predicts = LR_model.predict(X_train)


val_predicts = LR_model.predict(val_train)


from sklearn.metrics import mean_squared_error


train_rmse = np.sqrt(mean_squared_error(X_targets, X_predicts))


val_rmse = np.sqrt(mean_squared_error(val_targets, val_predicts))


print(train_rmse)
print(val_rmse)


LR_weight = LR_model.coef_


LR_model_Weights = pd.DataFrame({'Columns': X_train.columns, 'Weights': LR_weight})


LR_model_Weights


from sklearn.tree import DecisionTreeRegressor


DT_model = DecisionTreeRegressor(random_state = 28)


DT_model.fit(X_train, X_targets)


DT_X_predicts = DT_model.predict(X_train)


DT_val_predicts = DT_model.predict(val_train)


DT_X_rmse = np.sqrt(mean_squared_error(X_targets, DT_X_predicts))
DT_val_rmse = np.sqrt(mean_squared_error(val_targets, DT_val_predicts))


print(DT_X_rmse)
print(DT_val_rmse)


DT_model.tree_.max_depth


from sklearn.tree import plot_tree


plt.figure(figsize = (80,20))
plot_tree(DT_model, feature_names = X_train.columns, max_depth=2, filled= True)


DT_model = DecisionTreeRegressor(random_state = 28, max_depth = 10)


DT_model.fit(X_train, X_targets)


DT_X_predicts = DT_model.predict(X_train)
DT_val_predicts = DT_model.predict(val_train)


DT_X_rmse = np.sqrt(mean_squared_error(X_targets, DT_X_predicts))
DT_val_rmse = np.sqrt(mean_squared_error(val_targets, DT_val_predicts))
print(DT_X_rmse, DT_val_rmse)


DT_model = DecisionTreeRegressor(random_state = 28, max_depth = 8, max_features = 'sqrt')
DT_model1= DecisionTreeRegressor(random_state = 28, max_depth = 10, max_features = 0.6)


DT_model.fit(X_train, X_targets)
DT_model1.fit(X_train, X_targets)


DT_X_predicts = DT_model.predict(X_train)
DT_val_predicts = DT_model.predict(val_train)
DT1_X_predicts = DT_model1.predict(X_train)
DT1_val_predicts = DT_model1.predict(val_train)


DT_X_rmse = np.sqrt(mean_squared_error(X_targets, DT_X_predicts))
DT_val_rmse = np.sqrt(mean_squared_error(val_targets, DT_val_predicts))
DT1_X_rmse = np.sqrt(mean_squared_error(X_targets, DT1_X_predicts))
DT2_val_rmse = np.sqrt(mean_squared_error(val_targets, DT1_val_predicts))
print(DT_X_rmse, DT_val_rmse)
print(DT_X_rmse, DT_val_rmse)


from sklearn.ensemble import RandomForestRegressor


RF=RandomForestRegressor(random_state=28, n_jobs=-1, n_estimators = 10)


RF.fit(X_train, X_targets)


RF_X_predicts = RF.predict(X_train)
RF_val_predicts = RF.predict(val_train)


RF_X_rmse = np.sqrt(mean_squared_error(X_targets, RF_X_predicts))
RF_val_rmse = np.sqrt(mean_squared_error(val_targets, RF_val_predicts))
print(RF_X_rmse, RF_val_rmse)


RF1=RandomForestRegressor(random_state=28, n_jobs=-1, n_estimators = 10, max_depth = 7)


RF1.fit(X_train, X_targets)


RF1_X_predicts = RF1.predict(X_train)
RF1_val_predicts = RF1.predict(val_train)


RF1_X_rmse = np.sqrt(mean_squared_error(X_targets, RF1_X_predicts))
RF1_val_rmse = np.sqrt(mean_squared_error(val_targets, RF1_val_predicts))
print(RF1_X_rmse, RF1_val_rmse)


RF2=RandomForestRegressor(random_state=28, n_jobs=-1, n_estimators = 7, max_depth = 7)


RF2.fit(X_train, X_targets)


RF2_X_predicts = RF2.predict(X_train)
RF2_val_predicts = RF2.predict(val_train)


RF2_X_rmse = np.sqrt(mean_squared_error(X_targets, RF2_X_predicts))
RF2_val_rmse = np.sqrt(mean_squared_error(val_targets, RF2_val_predicts))
print(RF2_X_rmse, RF2_val_rmse)


RF3 = RandomForestRegressor(random_state=28, n_jobs=-1, n_estimators = 7, max_depth = 7, max_samples = 0.7)


RF3.fit(X_train, X_targets)


RF3_X_predicts = RF3.predict(X_train)
RF3_val_predicts = RF3.predict(val_train)


RF3_X_rmse = np.sqrt(mean_squared_error(X_targets, RF3_X_predicts))
RF3_val_rmse = np.sqrt(mean_squared_error(val_targets, RF3_val_predicts))
print(RF2_X_rmse, RF2_val_rmse)


from xgboost import XGBRegressor


Xgb_model = XGBRegressor(n_jobs=-1, random_state = 28, n_estimators = 10, max_depth = 8)


Xgb_model.fit(X_train,X_targets)


xgb_x_predicts=Xgb_model.predict(X_train)
xbg_val_predicts=Xgb_model.predict(val_train)


xgb_x_rmse = np.sqrt(mean_squared_error(X_targets, xgb_x_predicts))
xgb_val_rmse = np.sqrt(mean_squared_error(val_targets, xbg_val_predicts))


print(xgb_x_rmse, xgb_val_rmse)


test_df.head()


test_predict = Xgb_model.predict(test_df[num_cols + encoded_cols])


test_predict


import joblib


Insurance_premium_prediction = {
    'model': Xgb_model,
    'imputer': mean_imputer,
    'scaler': Scaler,
    'encoder': encoder,
    'input_cols': num_cols+encoded_cols,
    'target_col': 'Premium Amount',
    'numeric_cols': num_cols,
    'categorical_cols': cat_cols,
    'encoded_cols': encoded_cols
}


joblib.dump(Insurance_premium_prediction, 'Insurance_premium_prediction.joblib')




