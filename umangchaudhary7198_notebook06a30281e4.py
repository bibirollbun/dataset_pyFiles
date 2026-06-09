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


import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



df_train.head()


df_train[(df_train['Podcast_Name']=='Joke Junction') & (df_train['Episode_Title']=='Episode 26')]


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer



from sklearn.feature_extraction.text import TfidfVectorizer


# tfid=TfidfVectorizer(min_df=2,max_df=0.8)
# name_vectors_train=tfid.fit_transform(df_train['Podcast_Name'])
# name_vectors_test=tfid.fit_transform(df_test['Podcast_Name'])



# name_df_train=pd.DataFrame(name_vectors_train.toarray(),columns=tfid.get_feature_names_out())
# name_df_test=pd.DataFrame(name_vectors_test.toarray(),columns=tfid.get_feature_names_out())

# name_df_test.head()


# from sklearn.decomposition import PCA
# pca=PCA(n_components=50)
# t_reduced=pca.fit_transform(name_vectors.toarray())
# t_reduced


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, axes = plt.subplots(2, 2)

sns.boxplot(data=df_train['Episode_Length_minutes'], ax=axes[0,0])
sns.boxplot(data=df_train['Host_Popularity_percentage'], ax=axes[0,1])
sns.boxplot(data=df_train['Guest_Popularity_percentage'], ax=axes[1,0])
sns.boxplot(data=df_train['Number_of_Ads'],  ax=axes[1,1])



# def find_outliers(column,df):
#     Q1=df[column].quantile(0.25)
#     Q3=df[column].quantile(0.75)
#     IQR=Q3-Q1
#     outliers=df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
#     l=list(outliers['id'].unique())
#     return l


# cols=['Number_of_Ads','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage']
# for i in cols:    
#     df_train=df_train[~df_train['id'].isin(find_outliers(i,df_train))]
#     df_test=df_test[~df_test['id'].isin(find_outliers(i,df_test))]


# plt.plot(df['Episode_Length_minutes'])
df_train['Episode_Length_minutes'].hist(bins=100)

plt.show()



trans = ColumnTransformer([
    ('imputer',SimpleImputer(strategy='mean'),[3,8,9]),
    ('sentiment_encoder',OrdinalEncoder(categories=[['Positive','Neutral','Negative']]),[10]),
    
],remainder='passthrough',verbose_feature_names_out=False).set_output(transform='pandas')


X_train=trans.fit_transform(df_train)
X_test=trans.fit_transform(df_test)
X_test.head()


df_train = pd.get_dummies(X_train, columns=['Genre','Publication_Day','Publication_Time'])
df_test = pd.get_dummies(X_test, columns=['Genre','Publication_Day','Publication_Time'])

df_train.head()


y=df_train['Listening_Time_minutes']
df_train.drop(['Podcast_Name','Episode_Title'],axis=1,inplace=True)
df_test.drop(['Podcast_Name','Episode_Title'],axis=1,inplace=True)



y_true=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
y_true.head()


f, axes = plt.subplots(2, 2)
sns.distplot(  df_train['Episode_Length_minutes'],  ax=axes[0,0])
sns.distplot(  df_train['Guest_Popularity_percentage'],  ax=axes[0,1])
sns.distplot(df_train['Number_of_Ads'],   ax=axes[1,0])
sns.distplot(   df_train['Host_Popularity_percentage'],   ax=axes[1,1])
plt.tight_layout()
plt.show()


# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state=42)
# regressor.fit(x_train, y_train)
# print(r2_score(y_test,regressor.predict(x_test)))


# from sklearn.ensemble import GradientBoostingRegressor
# reg = GradientBoostingRegressor(random_state=0)
# reg.fit(df_train, y)
# reg.predict(df_test)
# print(reg.score(x_test, y_test))



from sklearn.model_selection import KFold
from xgboost import XGBRegressor
kf = KFold(n_splits=5, shuffle=True, random_state=42)




params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'gpu',
        'n_estimators': 500,
       
    }

oof_preds = np.zeros(len(df_train))

for fold, (train_idx, valid_idx) in enumerate(kf.split(df_train)):
    x_train=df_train.iloc[train_idx].copy()
    x_test=df_train.iloc[valid_idx].copy()
    y_train=x_train['Listening_Time_minutes']
    y_test=x_test['Listening_Time_minutes']

    model = XGBRegressor(**params)
    model.fit(x_train.drop('Listening_Time_minutes',axis=1), y_train,
              eval_set=[(x_test.drop('Listening_Time_minutes',axis=1), y_test)],
                  early_stopping_rounds=50, verbose=0)
    oof_preds[valid_idx] = model.predict(x_test.drop('Listening_Time_minutes',axis=1))





from sklearn.metrics import mean_squared_error
print(mean_squared_error(df_train['Listening_Time_minutes'], oof_preds, squared=False))



y_pred=model.predict(df_test)



y_pred=pd.DataFrame(y_pred,columns=['Listening_Time_minutes'])
final=pd.concat([df_test['id'].reset_index(drop=True),y_pred.reset_index(drop=True)],axis=1)
final.to_csv('submission.csv',index=False)


# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV

# params={
#     'tree_method':['hist'],
#         'max_depth':[14],
#         'colsample_bytree':[0.5],
#         'subsample':[0.8],
#         'n_estimators':[50_000],
#         'learning_rate':[0.04],
#         'enable_categorical':[True],
#         # 'early_stopping_rounds':[100],
#         'min_child_weight':[10],
# }
# model = xgb.XGBRegressor()
# xg = GridSearchCV(estimator=model, param_grid=params, n_jobs = 3, cv= 3)

# xg_model = xg.fit(X=x_train, y=y_train)











