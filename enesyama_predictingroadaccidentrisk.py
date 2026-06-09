import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)


DATA_PATH = '/kaggle/input/playground-series-s5e10/'

df1 = pd.read_csv(DATA_PATH + 'train.csv')
df2 = pd.read_csv(DATA_PATH + 'test.csv')


df1.head()


df1.tail()


df2.head()


df2.tail()


df1.info()


df2.info()


df1.isnull().sum()


df2.isnull().sum()


df1.shape


df2.shape


x=df1.drop(['accident_risk','id'], axis=1)


y=df1['accident_risk']


x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.20, random_state=42)


dummy=DummyRegressor(strategy='mean')
dummy.fit(x_train, y_train)


pred=dummy.predict(x_val)
rmse=np.sqrt(mean_squared_error(y_val, pred))


rmse


x.isna().mean().sort_values(ascending=False)


# Concat
df=pd.concat([df1.drop(['accident_risk', 'id'], axis=1),df2.drop(['id'], axis=1)],axis=0,ignore_index=True)


sns.kdeplot(x=df1['accident_risk'], fill=True);


df1['accident_risk'].describe()


df.select_dtypes('number').plot(kind='box', figsize=(16,10), vert=False);


plt.figure(figsize=(20,10))
sns.heatmap(df1.corr(numeric_only=True), annot=True, cmap='coolwarm')


for col in df.select_dtypes('number').columns:
    plt.figure(figsize=(4,1))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()


sns.scatterplot(x='num_reported_accidents', y='accident_risk', data=df1, alpha=0.2);


sns.boxplot(x='weather', y='accident_risk', data=df1)
plt.xticks(rotation=45);


sns.boxplot(x='holiday', y='accident_risk', data=df1);


# curve speed risk
df['curvature_speed']=df['curvature'] * df['speed_limit']


# lane speed load
df['lanes_speed']=df['num_lanes'] * df['speed_limit']


# accident density
df['accidents_lane']=df['num_reported_accidents'] / (df['num_lanes'] + 1)


# school holiday
df['holiday_school']=df['holiday'] & df['school_season']


# regulated public road
df['signs_public']=df['road_signs_present'] & df['public_road']


df.info()


bools=['road_signs_present','public_road','holiday','school_season','holiday_school','signs_public']


df[bools]=df[bools].astype(int)


df.info()


df=pd.get_dummies(df,drop_first=True)


df[df.select_dtypes('bool').columns]=df.select_dtypes('bool').astype(int)


df.info()


x=df[:517754]
x_test=df[517754:]


x.tail()


y.tail()


y=df1['accident_risk']


x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.20, random_state=42)


lr=LinearRegression(fit_intercept=True,n_jobs=-1)


lr.fit(x_train,y_train)


tahmin=lr.predict(x_val)


r2_score(y_val,tahmin)


mean_squared_error(y_val,tahmin)**.5


r=Ridge(alpha=1.0, random_state=42)


r.fit(x_train,y_train)


rtahmin=r.predict(x_val)


r2_score(y_val,rtahmin)


mean_squared_error(y_val,rtahmin)**.5


lgbm=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.05,max_depth=-1,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42)


lgbm.fit(x_train,y_train)


lgbmtahmin=lgbm.predict(x_val)


r2_score(y_val,lgbmtahmin)


mean_squared_error(y_val,lgbmtahmin)**.5


pl=(pd.DataFrame({'feature': x_train.columns,'importance': lgbm.feature_importances_}).sort_values('importance', ascending=False).head(15))


pl


plt.figure(figsize=(8, 6))
plt.barh(pl['feature'][::-1], pl['importance'][::-1])
plt.title('LightGBM Feature Importance (Top 15)')
plt.xlabel('Importance (split count)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


residuals=y_val - lgbmtahmin


# Residuals
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=60)
plt.title('Residuals Distribution')
plt.xlabel('Residual (y_true - y_pred)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))
plt.scatter(lgbmtahmin, residuals, alpha=0.25)
plt.axhline(0, linestyle='--')
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted accident_risk')
plt.ylabel('Residual (y_true - y_pred)')
plt.tight_layout()
plt.show()


xgb=XGBRegressor(n_estimators=800,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,objective='reg:squarederror',random_state=42)


xgb.fit(x_train,y_train)


xgbtahmin=xgb.predict(x_val)


r2_score(y_val,xgbtahmin)


mean_squared_error(y_val,xgbtahmin)**.5


sub=pd.read_csv(DATA_PATH + 'sample_submission.csv')


sub.head()


lgbm_final=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.05,max_depth=-1,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42)


lgbm_final.fit(x, y)


test_tahmin=lgbm_final.predict(x_test)


test_tahmin=np.clip(test_tahmin, 0, 1)


submission=pd.DataFrame({'id':df2['id'],'accident_risk':test_tahmin})


submission.head()


sub.isnull().sum()


submission.to_csv('submission.csv', index=False)


joblib.dump(x_train.columns.tolist(), 'feature_names.pkl')
joblib.dump(lgbm_final, 'lgbm_model.pkl')

