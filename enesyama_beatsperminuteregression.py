import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)


DATA_PATH = '/kaggle/input/playground-series-s5e9/'

df1=pd.read_csv(DATA_PATH + 'train.csv')
df2=pd.read_csv(DATA_PATH + 'test.csv')


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


plt.figure(figsize=(8,5))
plt.hist(df1['BeatsPerMinute'], bins=50)
plt.title('Distribution of Beats Per Minute')
plt.xlabel('Beats Per Minute')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8,2))
plt.boxplot(df1['BeatsPerMinute'], vert=False)
plt.title('Boxplot of Beats Per Minute')
plt.xlabel('Beats Per Minute')
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(
    df1.drop('id', axis=1).corr(),
    cmap='coolwarm',
    center=0
)
plt.title('Correlation Heatmap')
plt.show()


abs(df1.corr(numeric_only=True)['BeatsPerMinute'].sort_values(ascending=False))


for col in df1.select_dtypes('number').columns:
    plt.figure(figsize=(4,1))
    sns.boxplot(x=df1[col])
    plt.title(col)
    plt.show()


df1.hist(figsize=(14,10),bins=30)
plt.tight_layout()
plt.show()


def features(df):

    #rhythm × energy
    df['Rhythm_Energy']=df['RhythmScore'] * df['Energy']
    # mood × energy
    df['Mood_Energy']=df['MoodScore'] * df['Energy']
    #loudness × energy
    df['Loudness_Energy']=df['AudioLoudness'] * df['Energy']
    
    return df


df1=features(df1)
df2=features(df2)


df1.info()


df2.info()


x=df1.drop(['id','BeatsPerMinute'],axis=1)


y=df1['BeatsPerMinute']


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


rf=RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=42,n_jobs=-1)


rf.fit(x_train,y_train)


rftahmin=rf.predict(x_val)


r2_score(y_val,rftahmin)


mean_squared_error(y_val,rftahmin)**.5


lgbm=lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.05,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)


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


sub=pd.read_csv(DATA_PATH + 'sample_submission.csv')


sub.head()


x_final=df1.drop(['BeatsPerMinute', 'id'], axis=1)
y_final=df1['BeatsPerMinute']


lgbm_final=lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.05,max_depth=-1,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42)


lgbm_final.fit(x_final, y_final)


x_test=df2.drop('id', axis=1)


test_tahmin=lgbm_final.predict(x_test)


submission=pd.DataFrame({'id':df2['id'],'BeatsPerMinute':test_tahmin})


submission.head()


submission.to_csv('submission.csv',index=False)


joblib.dump(lgbm_final, 'lgbm_model.pkl')


joblib.dump(x_final.columns.tolist(), 'feature_names.pkl')


feature_names = joblib.load('feature_names.pkl')
print(feature_names)

