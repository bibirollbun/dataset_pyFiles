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
import pandas as pd
import numpy as np
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


df.columns


df.shape


df.info()


df.describe()


sns.countplot(x='road_type',data=df)
plt.show()


sns.countplot(x='speed_limit',data=df)
plt.show()


sns.countplot(x='num_reported_accidents',data=df)
plt.show()


plt.scatter(df['curvature'],df['accident_risk'],alpha=0.1)
plt.show()


df.isna().sum()


plt.scatter(df['num_reported_accidents'],df['accident_risk'],alpha=0.1)
plt.show()


sns.histplot(df['accident_risk'], kde=True)


sns.boxplot(x='speed_limit', y='accident_risk', data=df)


sns.barplot(x='road_type', y='accident_risk', data=df)


sns.boxplot(x='time_of_day',y='accident_risk',data=df)


df['spped_curvature'] = df['speed_limit'] * df['curvature']
df['risk_compin'] = df['num_lanes'] * df['curvature']
df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
df['curvature_rain'] = df['curvature'] * (df['weather']=='rainy').astype(int)
df['curvature_night']=df['curvature'] * (df['lighting']=='night').astype(int)
df['speed_night'] = df['speed_limit'] * (df['lighting']=='night').astype(int)
df['no_signs_curvature'] = df['curvature'] *  (df['road_signs_present']==False).astype(int)
df['public_speed'] = df['speed_limit'] * df['public_road'].astype(int)
df['holiday_night'] =  (df['lighting']=='night').astype(int) * (df['holiday'] == True).astype(int)
df['school_season_day'] =  (df['school_season']==True).astype(int) * (df['time_of_day']=='afternoon').astype(int)


test['spped_curvature'] = test['speed_limit'] * test['curvature']
test['risk_compin'] = test['num_lanes'] * test['curvature']
test['speed_lanes'] = test['speed_limit'] * test['num_lanes']
test['curvature_rain'] = test['curvature'] * (test['weather']=='rainy').astype(int)
test['curvature_night'] = test['curvature'] * (test['lighting']=='night').astype(int)
test['speed_night'] = test['speed_limit'] * (test['lighting']=='night').astype(int)
test['no_signs_curvature'] = test['curvature'] *  (test['road_signs_present']==False).astype(int)
test['public_speed'] = test['speed_limit'] * test['public_road'].astype(int)
test['holiday_night'] =  (test['lighting']=='night').astype(int) * (test['holiday'] == True).astype(int)
test['school_season_day'] =  (test['school_season']==True).astype(int) * (test['time_of_day']=='afternoon').astype(int)




plt.figure(figsize=(20,12))
cor = df.select_dtypes(include='number').corr()
sns.heatmap(cor,annot=True,cmap='coolwarm')


sns.barplot(x='road_type', y='accident_risk', data=df)


sns.boxplot(x='holiday',y='accident_risk',data=df)


sns.boxplot(x='weather',y='accident_risk',data=df) 


sns.scatterplot(x=df['curvature']*df['speed_limit'], y=df['accident_risk'],alpha=0.1)


df =df.drop_duplicates()
print("After dropping, duplicates:", df.duplicated().sum())


num_cols = df.select_dtypes(include=['int64', 'float64']).columns 
for col in num_cols:
    Q1 = df[col].quantile(0.80)
    Q3 = df[col].quantile(0.10)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


#encoding # sacling
#df.drop(columns=['id'],inplace=True)
X = df.drop(columns='accident_risk')
y = df['accident_risk']
#X.info()
numeric = X.select_dtypes(include=['int64','float64']).columns
categorical = X.select_dtypes(include=['object','bool']).columns



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score,mean_squared_error

preprocessor = ColumnTransformer([
    ('scale',StandardScaler(),numeric),
    ('encode',OneHotEncoder(),categorical)
])


pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
        random_state=42,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_weight=1,
        gamma=0,
        n_jobs=-1,
        objective='reg:squarederror'
))
    ])





X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
pipeline.fit(X_train,y_train)
y_pred_val = pipeline.predict(X_val)
y_pred_test = pipeline.predict(test)


rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val,y_pred_val)


print('Validation Results')
print(f'RMSE: {rmse_val}')
print(f'R2_Val: {r2_val}')


submission = pd.DataFrame({
    'id': test['id'],            
    'accident_risk': y_pred_test
})


submission.to_csv('submission.csv',index=False)


0.05623
0.05579
0.05658
0.05649
0.05608

