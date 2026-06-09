from IPython.display import display, HTML, Markdown


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


#customising the palette

saniya_palette = [
    "#fba1c2", "#152f64", "#886893", "#fff3f8", "#e58ab5",
    "#5c4b85", "#30477f", "#d88fad", "#cab0d5", "#9e7fb3",
    "#3f5d96", "#ffcadb", "#6f5a92", "#a3c3eb", "#b87aa7"
]

sns.set_palette(saniya_palette) #for setting the default palette


#custom cmap palette
import matplotlib.colors as mlc
saniya_pal=mlc.LinearSegmentedColormap.from_list("custom",["#fba1c2", "#152f64"])


data=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
data2=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df=pd.DataFrame(data)
df2=pd.DataFrame(data2)


df


#id and index is same
df=df.drop(columns=['id'])
df


df.dtypes


df.shape


df.nunique()


#splitting into two columns
df[['Episode','Title']]=df['Episode_Title'].str.split(' ',n=1, expand=True)


df=df.drop(columns=['Episode_Title','Episode'])


df


df.isnull().sum()


df['Title']=df['Title'].apply(lambda x:int(x))


fig,ax=plt.subplots(3,1,figsize=(40,20))
sns.countplot(x='Podcast_Name',data=df,hue='Podcast_Name',palette=saniya_palette,ax=ax[0])
ax[0].tick_params(rotation=90)
sns.countplot(x='Genre',data=df,hue='Genre',palette=saniya_palette,ax=ax[1])
ax[0].tick_params(rotation=90)
sns.countplot(x='Publication_Day',data=df,hue='Publication_Day',palette=saniya_palette,ax=ax[2])
ax[0].tick_params(rotation=90)
plt.show()


fig,ax=plt.subplots(2,2,figsize=(40,20))
sns.histplot(data=df,x='Episode_Length_minutes',palette=saniya_palette,ax=ax[0,0],bins=10,kde=True)
ax[0,0].tick_params(rotation=90)
sns.histplot(data=df,x='Host_Popularity_percentage',palette=saniya_palette,ax=ax[0,1],bins=10,kde=True)
ax[0,1].tick_params(rotation=90)
sns.histplot(data=df,x='Guest_Popularity_percentage',palette=saniya_palette,ax=ax[1,0],bins=10,kde=True)
ax[1,0].tick_params(rotation=90)
sns.histplot(data=df,x='Title',palette=saniya_palette,ax=ax[1,1],bins=10,kde=True)
ax[1,0].tick_params(rotation=90)
plt.show()


fig,ax=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='Publication_Time',data=df,hue='Publication_Time',palette=saniya_palette,ax=ax[0])
#ax[0].tick_params(rotation=90)
sns.countplot(x='Episode_Sentiment',data=df,hue='Episode_Sentiment',palette=saniya_palette,ax=ax[1])
#ax[0].tick_params(rotation=90)


sns.histplot(data=df, x='Listening_Time_minutes',palette=saniya_palette,bins=50,kde=True)
plt.show()


#dividing the listening time into two categories, for plotting
mean=df['Listening_Time_minutes'].mean()
df['Listener_Type']=df['Listening_Time_minutes'].apply(lambda x: "Long_Term" if x>mean else "Short_Term")
df


#Bi-variate Analysis
fig,ax=plt.subplots(2,2,figsize=(10,10))
sns.scatterplot(x=df['Episode_Length_minutes'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[0,0])
ax[0,0].tick_params(rotation=90)
sns.scatterplot(x=df['Host_Popularity_percentage'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[0,1])
ax[0,1].tick_params(rotation=90)
sns.scatterplot(x=df['Guest_Popularity_percentage'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[1,0])
ax[1,0].tick_params(rotation=90)
sns.scatterplot(x=df['Number_of_Ads'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[1,1])
ax[1,1].tick_params(rotation=90)
plt.show()


df


#imputation(filling in the null values)
df.isnull().sum()


df['Episode_Length_minutes']=df['Episode_Length_minutes'].fillna(value=df['Episode_Length_minutes'].mean())
df['Guest_Popularity_percentage']=df['Guest_Popularity_percentage'].fillna(value=df['Guest_Popularity_percentage'].mean())
df['Number_of_Ads']=df['Number_of_Ads'].fillna(value=df['Number_of_Ads'].mean())
df['Number_of_Ads']=df['Number_of_Ads'].apply(lambda x:int(x))


df.isnull().sum()


def feat_eng(df):
    df['Host/Guest']=df['Host_Popularity_percentage']/(df['Guest_Popularity_percentage']+1) #to avoid divide by 0 error.
    df['Interval_Length']=df['Episode_Length_minutes']/(df['Number_of_Ads']+1)
    df['Segments']=df['Number_of_Ads']+1
    #day and time relation
    df['Day_Time']=df['Publication_Day']/df['Publication_Time']
    #commercial effect 
    df['Host_Ads']=df['Host_Popularity_percentage']*df['Number_of_Ads']
    df['Guest_Ads']=df['Guest_Popularity_percentage']*df['Number_of_Ads']
    return df
    


mapping = { #map the values to new values
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}

df["Publication_Day"] = df["Publication_Day"].map(mapping)


mapping2 = { #map the values to new values
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4,
}

df["Publication_Time"] = df["Publication_Time"].map(mapping2)


df=feat_eng(df)
df


#EDA 2
fig,ax=plt.subplots(1,3,figsize=(20,5))
sns.histplot(data=df,x='Host/Guest',palette=saniya_palette,ax=ax[0],bins=50,kde=True)
ax[0].tick_params(rotation=90)
sns.histplot(data=df,x='Interval_Length',palette=saniya_palette,ax=ax[1],bins=10,kde=True)
ax[1].tick_params(rotation=90)
sns.countplot(data=df,x='Segments',palette=saniya_palette,ax=ax[2])
ax[2].tick_params(rotation=90)


#Bivariate analysis 2
fig,ax=plt.subplots(2,2,figsize=(10,10))
sns.scatterplot(x=df['Host/Guest'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[0,0])
ax[0,0].tick_params(rotation=90)
sns.scatterplot(x=df['Interval_Length'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[0,1])
ax[0,1].tick_params(rotation=90)
sns.scatterplot(x=df['Host_Ads'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[1,0])
ax[1,0].tick_params(rotation=90)
sns.scatterplot(x=df['Segments'],y=df['Listening_Time_minutes'],hue=df['Listener_Type'],ax=ax[1,1])
ax[1,1].tick_params(rotation=90)
plt.show()


from sklearn.preprocessing import LabelEncoder


df.dtypes


le=LabelEncoder()
cols=['Podcast_Name','Genre','Episode_Sentiment','Listener_Type']

for col in cols:
    df[col]=le.fit_transform(df[col])


df


from scipy import stats

z = np.abs(stats.zscore(df))

threshold = 3

#columns with outliers
cols = ['Interval_Length','Host_Ads', 'Guest_Ads','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']

#removing outliers
df = df[(z < 3).all(axis=1)]


df.shape


#chi squared test for segments
from scipy.stats import chi2_contingency

#sample data
samp=df.head(100)
data_c=pd.crosstab(df['Segments'],df['Listening_Time_minutes'], margins=False)
#data_s=pd.crosstab(samp['Listening_Time_minutes'],samp['Segments'], margins=False)
data_c


stat,p, dof, expected= chi2_contingency(data_c)
print('stat=%.3f,p=%.3f' % (stat,p))


fig,ax=plt.subplots(1,1,figsize=(20,20))
sns.heatmap(df.corr(),annot=True, cmap=saniya_pal)


import lightgbm as lgbm


lgb=lgbm.LGBMRegressor(n_estimators=100)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x=df.drop(columns=['Listening_Time_minutes','Listener_Type'])
y=df['Listening_Time_minutes']


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.4, random_state=42)


## Feature Selection

#initially wanted to get only the most important features but it was affecting the performance, indicating under-fitting.

'''from mlxtend.feature_selection import SequentialFeatureSelector
f_feat_selec=SequentialFeatureSelector(lgbm.LGBMRegressor(n_jobs=-1), #try without standardising,accuracy only for classification
                                      k_features=10,
                                      verbose=2,
                                      forward=True,
                                      floating=False,
                                      scoring='neg_mean_squared_error',
                                      cv=5
                                      ).fit(x_train,y_train)'''


#f_feat_selec.k_feature_names_


'''df_copy=df[['Podcast_Name',
 'Episode_Length_minutes',
 'Host_Popularity_percentage',
 'Publication_Day',
 'Publication_Time',
 'Guest_Popularity_percentage',
 'Episode_Sentiment',
 'Title',
 'Interval_Length',
 'Host_Ads']]'''


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=42)


from sklearn.preprocessing import StandardScaler
stan=StandardScaler()
x_train=stan.fit_transform(x_train)
x_test=stan.transform(x_test)


lgb.fit(x_train,y_train)


pred=lgb.predict(x_test)


pred


from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
r2_score(y_test, pred)
sqrt(mean_squared_error(y_test, pred))


'''from sklearn.model_selection import GridSearchCV

#parameters for grid search
param_grid = {
    "num_leaves": [27, 31, 63, 127],
    "max_depth": [-1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
# Grid Search Object
grid = GridSearchCV(lgb, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)

#fitting the grid search
grid.fit(x_train, y_train)

#best parameters
print(grid.best_params_)'''


#Applying the tuning
lgb2=lgbm.LGBMRegressor(n_estimators=100, colsample_bytree= 0.8, max_depth= -1, num_leaves= 127, subsample= 0.8)


lgb2.fit(x_train,y_train)


pred_2=lgb2.predict(x_test)


pred_2


r2_score(y_test, pred_2)
sqrt(mean_squared_error(y_test, pred_2))


df_test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_test


df_test.info()


df_test=df_test.drop(columns=['id'])
df_test


#splitting into two columns
df_test[['Episode','Title']]=df_test['Episode_Title'].str.split(' ',n=1, expand=True)


df_test=df_test.drop(columns=['Episode_Title','Episode'])


df_test.isnull().sum()
df_test['Title']=df_test['Title'].apply(lambda x:int(x))
df_test['Episode_Length_minutes']=df_test['Episode_Length_minutes'].fillna(value=df_test['Episode_Length_minutes'].mean())
df_test['Guest_Popularity_percentage']=df_test['Guest_Popularity_percentage'].fillna(value=df_test['Guest_Popularity_percentage'].mean())
df_test


df_test["Publication_Day"] = df_test["Publication_Day"].map(mapping)
df_test["Publication_Time"] = df_test["Publication_Time"].map(mapping2)


df_test=feat_eng(df_test)


df_test.shape


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
cols=['Podcast_Name','Genre','Episode_Sentiment']

for col in cols:
    df_test[col]=le.fit_transform(df_test[col])


df_test.dtypes


'''x=df_test[['Podcast_Name',
 'Episode_Length_minutes',
 'Host_Popularity_percentage',
 'Publication_Day',
 'Publication_Time',
 'Guest_Popularity_percentage',
 'Episode_Sentiment',
 'Title',
 'Interval_Length',
 'Host_Ads']]'''


x_trans=stan.fit_transform(df_test)


res=lgb.predict(x_trans)


res


res.shape


sub=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


sub


sub['Listening_Time_minutes']=res


sub


sub.to_csv('/kaggle/working/submission.csv',index=False)


feat_lgb=pd.DataFrame({'Features' :df_test.columns,'Importances' :lgb.feature_importances_})


figure, ax=plt.subplots(1,1,figsize=(20,5))
sns.barplot(x='Features',y='Importances',data=feat_lgb,palette=saniya_palette)
plt.xticks(rotation=90)

