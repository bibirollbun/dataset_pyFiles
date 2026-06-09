from IPython.display import display, HTML, Markdown


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score, f1_score, recall_score,confusion_matrix,roc_auc_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer




data=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df=pd.DataFrame(data)
df


df.info()


df.shape


df.dtypes


df.isnull().sum()


#dropping the id column since it's not contributing to the data
df.drop(columns=['id'],inplace=True)
df


#checking for unique values
df.nunique()


df.describe()


#customising the palette

saniya_palette = [
    "#fba1c2", "#152f64", "#886893", "#fff3f8", "#e58ab5",
    "#5c4b85", "#30477f", "#d88fad", "#cab0d5", "#9e7fb3",
    "#3f5d96", "#ffcadb", "#6f5a92", "#a3c3eb", "#b87aa7"
]

sns.set_palette(saniya_palette) #for setting the default palette


#plotting the data distribution
fig,ax=plt.subplots(2,3,figsize=(20,10))
sns.histplot(df['pressure'], ax=ax[0,0],bins=10)
#sns.histplot(df['temparature'],ax=ax[0,1],bins=50)#NEVER use x and y axis for histplots, it goes mad
sns.histplot(data=df,x="maxtemp", ax=ax[0,1],bins=10 )
sns.histplot(data=df,x="temparature", ax=ax[0,1],bins=10) #sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack"), stack to stack plots,layer to overlay
sns.histplot(data=df,x="mintemp", ax=ax[0,1],bins=10)
sns.histplot(data=df,x="dewpoint", ax=ax[0,2],bins=10)
sns.histplot(data=df,x="humidity", ax=ax[1,0],bins=10)
sns.histplot(data=df,x="cloud", ax=ax[1,1],bins=10)
sns.histplot(data=df,x="sunshine", ax=ax[1,2],bins=10)


#plotting the data distribution
fig,ax=plt.subplots(1,2,figsize=(20,10))
sns.histplot(df['winddirection'], ax=ax[0],bins=10)
#sns.histplot(df['temparature'],ax=ax[0,1],bins=50)#NEVER use x and y axis for histplots, it goes mad
sns.histplot(data=df,x="windspeed", ax=ax[1],bins=10 )


#rainfall distribution
fig,ax=plt.subplots(1,2,figsize=(15,10))
sns.countplot(data=df,x="rainfall", ax=ax[0], hue='rainfall')
label=['Yes','No']
p=df['rainfall'].value_counts().sort_values(ascending=False)
labels=['Yes','No']
plt.pie(p,labels=labels)
plt.show()


#rainfall dependency
fig,ax=plt.subplots(2,4,figsize=(20,10))
sns.kdeplot(x="temparature", data=df, hue="rainfall",multiple="layer",fill=True,ax=ax[0,0]) #stack to overlay plots
sns.kdeplot(data=df, x="pressure", hue="rainfall", multiple="layer",fill=True, ax=ax[0,1])
sns.kdeplot(data=df, x="humidity", hue="rainfall", multiple="layer",fill=True, ax=ax[0,2])
sns.kdeplot(data=df, x="cloud", hue="rainfall", multiple="layer",fill=True, ax=ax[0,3])
sns.kdeplot(data=df, x="sunshine", hue="rainfall", multiple="layer",fill=True,ax=ax[1,0])
sns.kdeplot(data=df, x="dewpoint", hue="rainfall", multiple="layer",fill=True, ax=ax[1,1])
sns.kdeplot(data=df, x="winddirection", hue="rainfall", multiple="layer",fill=True, ax=ax[1,2])
sns.kdeplot(data=df, x="windspeed", hue="rainfall", multiple="layer",fill=True, ax=ax[1,3])


#Bivariate Analysis
fig,ax=plt.subplots(1,3,figsize=(30,10))
sns.scatterplot(x=df['temparature'],y=df['pressure'],hue=df['rainfall'],ax=ax[0])
sns.scatterplot(x=df['sunshine'],y=df['cloud'], hue=df['rainfall'],ax=ax[1])
sns.scatterplot(x=df['cloud'],y=df['temparature'],hue=df['rainfall'],ax=ax[2])


def feature_eng(df):
    
    df['windcomp']=df['winddirection'].apply(lambda x:round(x/22.5+1)) #for applying the lambda function or any function
    #converting temparature into farenheit
    df['temp-faren']=df['temparature'] * 1.8 + 32
    #extracting the heat index
    df['Heat-Index']= 0.5* df['temp-faren'] + 61.0 + df['temp-faren']-68.0 * 1.2+ df['humidity']*0.094
    #calculating wind chill index
    df['wind-chill']=35.74 + 0.6215 * df['temp-faren']-35.75* df['windspeed']** 0.16 + 0.4275* df['windspeed']**0.16
    #finding temparature range
    df['temp-range']=df['maxtemp']-df['mintemp']
    #finding the temparature range
    df['avg-temp']=df['temp-range']/2
    df['sunshine-cloud']=df['sunshine']/df['cloud']
    df['temp-humid']=df['temparature']/df['humidity']
    df['dew-humid']=df['dewpoint']/df['humidity']



df_copy=df.copy()
feature_eng(df_copy)
df_copy


#preparing the test data
data=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test=pd.DataFrame(data)
df_test.drop(columns=['id'],inplace=True)
df_test


#preprocessing the data
df_test.info()


df_test.isnull().sum()


df_test.fillna(value=df_test['winddirection'].mean(),inplace=True)
df_test.isnull().sum()


#feature engineering
df_test_copy=df_test.copy()
feature_eng(df_test_copy)
df_test_copy


import matplotlib.colors as mlc
saniya_pal=mlc.LinearSegmentedColormap.from_list("custom",["#fba1c2", "#152f64"])
plt.figure(figsize=(15,10))
sns.heatmap(df_copy.corr(),cmap=saniya_pal, annot=True)


#splitting the data
from sklearn.model_selection import train_test_split
x=df_copy.drop(columns=['rainfall'])
y=df_copy['rainfall']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)


#standard scalar
stan=StandardScaler()
x_train=stan.fit_transform(x_train)
x_test=stan.transform(x_test)


rfc=RandomForestClassifier(random_state=42, max_features='sqrt',n_estimators=200, class_weight='balanced')
rfc.fit(x_train,y_train)


#training accuracy
rfc.score(x_train,y_train)


rfc_pred=rfc.predict(x_test)
rfc_pred


dtc=DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=6, min_samples_split=4,class_weight='balanced')
dtc.fit(x_train,y_train)


dtc.score(x_train,y_train)
dtc_pred=dtc.predict(x_test)
dtc_pred


lr=LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(x_train,y_train)


lr.score(x_train,y_train)
lr_pred=lr.predict(x_test)
lr_pred


## XGB Classifier
from xgboost import XGBClassifier
xgb=XGBClassifier(max_depth=4, subsample=0.2,n_estimators=200, learning_rate=0.01)
xgb.fit(x_train,y_train)


xgb.score(x_train,y_train)
xgb_pred=xgb.predict(x_test)
xgb_pred


fig,ax=plt.subplots(1,4,figsize=(20,5))
saniya_pal2=mlc.LinearSegmentedColormap.from_list("custom",["#152f64", "#fba1c2"])
sns.heatmap(confusion_matrix(y_test,rfc_pred),annot=True,fmt='d', ax=ax[0],cmap=saniya_pal2).set_title('Random Forest') #fmt to get integer results
sns.heatmap(confusion_matrix(y_test,dtc_pred),annot=True,fmt='d', ax=ax[1],cmap=saniya_pal2).set_title('Decision Tree')
sns.heatmap(confusion_matrix(y_test,lr_pred),annot=True, fmt='d', ax=ax[2],cmap=saniya_pal2).set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test,xgb_pred),annot=True, fmt='d', ax=ax[3],cmap=saniya_pal2).set_title('XGB Classifier')


from math import sqrt
print('Random Forest Scores')
print('Accuracy Score: ', accuracy_score(y_test, rfc_pred))
print('Precision Score: ', precision_score(y_test, rfc_pred))
print('Recall Score: ', recall_score(y_test, rfc_pred))
print('F1 Score: ', f1_score(y_test, rfc_pred))
print('AUC Score: ', roc_auc_score(y_test, rfc_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test,rfc_pred)))


print('Decision Tree Scores')
print('Accuracy Score: ', accuracy_score(y_test, dtc_pred))
print('Precision Score: ', precision_score(y_test, dtc_pred))
print('Recall Score: ', recall_score(y_test, dtc_pred))
print('F1 Score: ', f1_score(y_test, dtc_pred))
print('AUC Score: ', roc_auc_score(y_test, dtc_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test,dtc_pred)))


print('Logistic Regression Scores')
print('Accuracy Score: ', accuracy_score(y_test, lr_pred))
print('Precision Score: ', precision_score(y_test, lr_pred))
print('Recall Score: ', recall_score(y_test, lr_pred))
print('F1 Score: ', f1_score(y_test, lr_pred))
print('AUC Score: ', roc_auc_score(y_test, lr_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test,lr_pred)))


print('XGB Classifier Scores')
print('Accuracy Score: ', accuracy_score(y_test, xgb_pred))
print('Precision Score: ', precision_score(y_test, xgb_pred))
print('Recall Score: ', recall_score(y_test, xgb_pred))
print('F1 Score: ', f1_score(y_test, xgb_pred))
print('AUC Score: ', roc_auc_score(y_test, xgb_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test,xgb_pred)))


#test_df = df_test_copy[x.columns]
xgb_sub=xgb.predict_proba(df_test_copy)[:,1]
xgb_sub


np.isnan(df_test_copy.values.any())
df_test_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
s=SimpleImputer(missing_values=np.nan, strategy='mean')
s_test=s.fit_transform(df_test_copy)
rfc_sub=rfc.predict_proba(s_test)[:,1]
rfc_sub


dtc_sub=dtc.predict_proba(s_test)[:,1]
#dtc_sub=dtc.predict(df_test_copy)
dtc_sub


#choosing the final two best models with highest AUC scores and averaging the values
final_pred=(xgb_sub+dtc_sub)/2
final_pred


feat_dtc=pd.DataFrame({'Features' :x.columns,'Importances' :dtc.feature_importances_})
feat_xgb=pd.DataFrame({'Features' :x.columns,'Importances' :xgb.feature_importances_})


figure, ax=plt.subplots(1,2,figsize=(10,5))
sns.barplot(x='Features',y='Importances',data=feat_dtc,ax=ax[0],palette=saniya_palette)
ax[0].tick_params(axis='x', rotation=90)
sns.barplot(x='Features',y='Importances',data=feat_xgb,palette=saniya_palette)
ax[1].tick_params(axis='x', rotation=90)


data=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
df_sub=pd.DataFrame(data)
df_sub


df_sub['rainfall']=final_pred
df_sub


df_sub.to_csv('/kaggle/working/submission.csv',index=False)

