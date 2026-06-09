import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')





train_data.head()


train_data.describe().T


train_data.rename(columns={'temparature': 'temperature'}, inplace=True)
test_data.rename(columns={'temparature':'temperature'},inplace=True)


test_data


#Feature engineering
from math import sin, cos, pi

train_data['temp_range'] = train_data['maxtemp'] - train_data['mintemp']
train_data['lower_t_avg'] = train_data['temperature'] - train_data['mintemp']
train_data['higher_t_avg'] = train_data['maxtemp'] - train_data['temperature']
train_data['cloud_sunshine'] = train_data['cloud'] * train_data['sunshine']
train_data['temp_humidity'] = train_data['temperature'] * train_data['humidity']
train_data['cloud_sunshine_ratio'] = train_data['cloud'] / (train_data['sunshine'] + 1)  # Avoid division by zero
train_data['htp'] = (train_data['humidity'] * train_data['temperature']) / train_data['pressure']
train_data['humidity_dew_ratio'] = train_data['humidity'] / (train_data['dewpoint'] + 1)  # Avoid division by zero
train_data['chs'] = train_data['cloud'] + train_data['humidity'] + train_data['sunshine']
train_data['heat_index'] = 0.5 * (train_data['temperature'] + 61.0 + 
                                  ((train_data['temperature'] - 68.0) * 1.2) + 
                                  (train_data['humidity'] * 0.094))
train_data['winddirection_sin'] = np.sin(2 * np.pi * train_data['winddirection'] / 360)
train_data['winddirection_cos'] = np.cos(2 * np.pi * train_data['winddirection'] / 360)
train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 365)
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 365)
train_data['humidity_sunshine'] = train_data['humidity'] * train_data['sunshine']
train_data['humidity_pressure'] = train_data['humidity'] * train_data['pressure']

# Additional Feature Engineering
train_data['dewpoint_deficit'] = abs(train_data['temperature'] - train_data['dewpoint'])
train_data['sunshine_humidity_ratio'] = train_data['sunshine'] / (train_data['humidity'] + 1)  # Avoid division by zero
train_data['previous_day_temp'] = train_data['temperature'].shift(1).fillna(train_data['temperature'].mean())
train_data['previous_day_humidity'] = train_data['humidity'].shift(1).fillna(train_data['humidity'].mean())

# Apply same transformations to test data
test_data['temp_range'] = test_data['maxtemp'] - test_data['mintemp']
test_data['lower_t_avg'] = test_data['temperature'] - test_data['mintemp']
test_data['higher_t_avg'] = test_data['maxtemp'] - test_data['temperature']
test_data['cloud_sunshine'] = test_data['cloud'] * test_data['sunshine']
test_data['temp_humidity'] = test_data['temperature'] * test_data['humidity']
test_data['cloud_sunshine_ratio'] = test_data['cloud'] / (test_data['sunshine'] + 1)
test_data['htp'] = (test_data['humidity'] * test_data['temperature']) / test_data['pressure']
test_data['humidity_dew_ratio'] = test_data['humidity'] / (test_data['dewpoint'] + 1)
test_data['chs'] = test_data['cloud'] + test_data['humidity'] + test_data['sunshine']
test_data['heat_index'] = 0.5 * (test_data['temperature'] + 61.0 + 
                                 ((test_data['temperature'] - 68.0) * 1.2) + 
                                 (test_data['humidity'] * 0.094))
test_data['winddirection_sin'] = np.sin(2 * np.pi * test_data['winddirection'] / 360)
test_data['winddirection_cos'] = np.cos(2 * np.pi * test_data['winddirection'] / 360)
test_data['day_sin'] = np.sin(2 * np.pi * test_data['day'] / 365)
test_data['day_cos'] = np.cos(2 * np.pi * test_data['day'] / 365)
test_data['humidity_sunshine'] = test_data['humidity'] * test_data['sunshine']
test_data['humidity_pressure'] = test_data['humidity'] * test_data['pressure']

test_data['dewpoint_deficit'] = abs(test_data['temperature'] - test_data['dewpoint'])
test_data['sunshine_humidity_ratio'] = test_data['sunshine'] / (test_data['humidity'] + 1)  
test_data['previous_day_temp'] = test_data['temperature'].shift(1).fillna(test_data['temperature'].mean())
test_data['previous_day_humidity'] = test_data['humidity'].shift(1).fillna(test_data['humidity'].mean())



data_to_train=train_data.drop(columns=['rainfall',
                                       'humidity',
                                       'id',
                                       'temperature',
                                      'sunshine',
                                      'dewpoint',
                                      'mintemp',
                                       'cloud',
                                      'maxtemp']).copy()

data_to_test=test_data.drop(columns=['humidity',
                                       'id',
                                       'temperature',
                                      'sunshine',
                                      'dewpoint',
                                     'cloud',
                                      'mintemp',
                                      'maxtemp']).copy()


data_to_train.isna().sum()


#histogram plot to check distribution of individual columns
int_cols=data_to_train.select_dtypes(include=[np.number]).columns
col=3
rows=(len(int_cols)//col)+1
fig,axes=plt.subplots(rows,col,figsize=(12,13))
axes=axes.flatten()
for i,col in enumerate(int_cols):
    sns.histplot(data_to_train[col],kde=True,ax=axes[i],label=col)
    axes[i].set_xlabel(col)

for j in range(i,len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(data_to_train.corr(),fmt='.2')


data_to_test.isna().sum()


#there is one missing value in wind direction, lets check that one
from sklearn.impute import SimpleImputer
impute=SimpleImputer(strategy='mean').fit(data_to_test[int_cols])
data_to_test[int_cols]=impute.transform(data_to_test[int_cols])


#checking for outliers
col=3
rows=(len(int_cols)//col)+1
fig,axes=plt.subplots(rows,col,figsize=(10,8))
axes=axes.flatten()
for i,col in enumerate(int_cols):
    sns.boxplot(y=data_to_train[col],ax=axes[i])
    axes[i].set_ylabel(col)
plt.title('BoxPlot')

for j in range(i,len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#From boxplot its clear that humidity,cloud,dewpoint has outliers
#calculating IQ,IQ3 and IQR
for col in int_cols:
    Q1=np.percentile(data_to_train[col],25)
    Q3=np.percentile(data_to_train[col],75)
    IQR=Q3-Q1
    lowerbound=Q1-1.5*IQR
    upperbound=Q3+1.5*IQR
    print(f"""For {col} 
        IQ:{Q1} 
        IQ3:{Q3} 
        IQR:{IQR}
        lowerbound:{lowerbound}
        upperbound:{upperbound}
    """)
    data_to_train[col] = data_to_train[col].where(
        (data_to_train[col]>=lowerbound) & (data_to_train[col]<=upperbound),
        data_to_train[col].median()
    )
    


#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(data_to_train[int_cols])
data_to_train[int_cols]=scaler.transform(data_to_train[int_cols])
data_to_test[int_cols]=scaler.transform(data_to_test[int_cols])


#basemodel
def base_model(feature):
    return np.full(len(feature['rainfall']),1)


from sklearn.metrics import accuracy_score
base_pred=base_model(train_data)
print(base_pred)
print(accuracy_score(train_data['rainfall'],base_pred))


#train test split
from sklearn.model_selection  import train_test_split
train_inputs,valid_inputs=train_test_split(data_to_train[int_cols],random_state=42,test_size=0.2)
train_targets,valid_targets=train_test_split(train_data['rainfall'],random_state=42,test_size=0.2)


pip install xgboost --quiet


#KFold with stacking
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


kf=KFold(n_splits=5,shuffle=True,random_state=42)
base_models=[('lr',LogisticRegression(n_jobs=-1,random_state=42)),
           ('rf',RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42,max_depth=8)),
           ('xg',XGBClassifier(
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        alpha=1)),
            ('SVC',SVC(random_state=42,kernel='linear'))]

for name,model in base_models:
    scores=cross_val_score(model,train_inputs,train_targets,cv=kf)
    plt.figure(figsize=(9,8))
    sns.barplot(x=np.arange(1,len(scores)+1),y=scores)
    plt.axhline(scores.mean(),linestyle='--',c='r')
    plt.xlabel('KFOLDS')
    plt.ylabel('Scores')
    plt.title(f"Performance{name}")


stacking_model=StackingClassifier(estimators=base_models,final_estimator=LogisticRegression(),cv=5)
stacking_model.fit(train_inputs,train_targets)


valid_pred_stacking=stacking_model.predict(valid_inputs)


from sklearn.metrics import roc_curve, roc_auc_score

def AUC_ROC(y_test,y_pred,model_name):
    fpr, tpr, _ = roc_curve(valid_targets, valid_pred)
    
    auc_score = roc_auc_score(valid_targets, valid_pred)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {model_name}")
    plt.legend()
    plt.show()



pred=stacking_model.predict(data_to_test)


xgbModel=XGBClassifier(max_depth=5,learning_rate=0.01,random_state=42,n_estimators=100)
xgbModel.fit(train_inputs,train_targets)


valid_pred_xgb=xgbModel.predict(valid_inputs)



AUC_ROC(valid_targets,valid_pred_stacking,'Stacking')
AUC_ROC(valid_targets,valid_pred_xgb,'XGBClassifier')


predict=xgbModel.predict(data_to_test)


xgbOutput=pd.DataFrame({
    'id':test_data['id'],
    'rainfall':pred
})
print(xgbOutput)


# output=pd.DataFrame({
#     'id':test_data['id'],
#     'rainfall':pred
# })
# print(output)


xgbOutput.to_csv('submission.csv',index=False)

