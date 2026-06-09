import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#获取数据
properties_2016 = pd.read_csv('../input/zillow-prize-1/properties_2016.csv', low_memory=False)
properties_2017 = pd.read_csv('../input/zillow-prize-1/properties_2017.csv', low_memory=False)
train_2016 = pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv', low_memory=False)
train_2017 = pd.read_csv('../input/zillow-prize-1/train_2017.csv', low_memory=False)
sample_submission = pd.read_csv('../input/zillow-prize-1/sample_submission.csv', low_memory=False)


properties_2016.isnull().sum()



properties_2017.isnull().sum()


train_2016.isnull().sum()


train_2017.isnull().sum()


#buildingclasstypeid calculatedbathnbr heatingorsystemtypeid
print(len(properties_2016))
properties_2016.groupby('buildingclasstypeid')['parcelid'].count().sum()
#数据缺失太多删除
properties_2016['calculatedbathnbr'].unique()
properties_2016.groupby('calculatedbathnbr')['parcelid'].count().sum() #保留
properties_2016['heatingorsystemtypeid'].unique()
properties_2016.groupby('heatingorsystemtypeid')['parcelid'].count().sum() #加热系统2级较多,保留以加热系统作为阈值进行缺失特征的删除


#2016
heating_counts=properties_2016.groupby('heatingorsystemtypeid')['parcelid'].count()
plt.figure(figsize=(12,6))
plt.plot(heating_counts.index,heating_counts.values,marker='o',linewidth=2,markersize=8)
plt.title('group of heatingorsystem')
plt.xlabel('heatingorsystem')
plt.ylabel('counts')
plt.grid(True, alpha=0.3)
for x, y in zip(heating_counts.index, heating_counts.values):
    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
plt.show()


#2017
heating_counts=properties_2017.groupby('heatingorsystemtypeid')['parcelid'].count()
plt.figure(figsize=(12,6))
plt.plot(heating_counts.index,heating_counts.values,marker='o',linewidth=2,markersize=8)
plt.title('group of heatingorsystem')
plt.xlabel('heatingorsystem')
plt.ylabel('counts')
plt.grid(True, alpha=0.3)
for x, y in zip(heating_counts.index, heating_counts.values):
    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
plt.show()


threshold=1178816 #以加热系统等级作为阈值
def feature_drop(df,threshold):
    columns=df.columns
    for col in columns:
        if df[col].isnull().sum(axis=0)>threshold:
            df=df.drop(col,axis=1)
    return df
properties_2016=feature_drop(properties_2016,threshold)
properties_2017=feature_drop(properties_2017,threshold)


#针对于特殊特征（缺失较多，综合上述分析，heatingorsystemtypeid其主要为2.0）
properties_2016['heatingorsystemtypeid']=properties_2016['heatingorsystemtypeid'].fillna(2.0)
properties_2017['heatingorsystemtypeid']=properties_2017['heatingorsystemtypeid'].fillna(2.0)


def fillna_col(df):
    columns=df.columns
    for col in columns:
        if df[col].isnull().any():
            if df[col].dtype=='object':
                df[col]=df[col].fillna(df[col].mode())
            else:
                df[col]=df[col].fillna(df[col].mean())
    return df
properties_2016=fillna_col(properties_2016)
properties_2017=fillna_col(properties_2017)
#存在属性id属性描述和分区做合并作为同一维度无需重复考虑
properties_2016=properties_2016.drop(['propertycountylandusecode','propertyzoningdesc'],axis=1)
properties_2017=properties_2017.drop(['propertycountylandusecode','propertyzoningdesc'],axis=1)


#确保列相同
print(set(properties_2016)-set(properties_2017))
print(set(train_2016)-set(train_2017))

p_col=properties_2016.columns
properties_2017=properties_2017[p_col]
t_col=train_2016.columns
train_2017=train_2017[t_col]


train_2016_merge=pd.merge(train_2016,properties_2016,on='parcelid',how='left')
train_2017_merge=pd.merge(train_2017,properties_2017,on='parcelid',how='left')


#相关性分析
def corr_analy(num_df,num_name):
    corr_m=num_df.corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(
        corr_m,annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidth=0.5,
    )
    plt.title(f'heatmap of {num_name}')
    plt.tight_layout()
    plt.show()
num_2016=train_2016_merge.drop(['transactiondate'],axis=1)
corr_analy(num_2016,'2016')
num_2017=train_2017_merge.drop(['transactiondate'],axis=1)
corr_analy(num_2017,'2017')


#  lotsizesquarefeet regionidcity regionidzip unitcnt assessmentyear censustractandblock
#与各特征的相关性过低不存在间接相关和直接相关进行删除处理
to_drop=['lotsizesquarefeet','regionidcity','regionidzip','unitcnt','assessmentyear','censustractandblock']
train_2016_merge=train_2016_merge.drop(to_drop,axis=1)
train_2017_merge=train_2017_merge.drop(to_drop,axis=1)


properties_2017=properties_2017.drop(to_drop,axis=1)
properties_2016=properties_2016.drop(to_drop,axis=1)


train_2017_merge.head()


#错误日志分布图像核分布
import warnings
warnings.filterwarnings('ignore') 
train_2016_clean = train_2016.copy()
train_2017_clean = train_2017.copy()

train_2016_clean['logerror'] = train_2016_clean['logerror'].replace([np.inf, -np.inf], np.nan)
train_2017_clean['logerror'] = train_2017_clean['logerror'].replace([np.inf, -np.inf], np.nan)

plt.figure(figsize=(12,6))
sns.kdeplot(train_2016_clean['logerror'].dropna(), label='logerror-2016', fill=True, alpha=0.5)
sns.kdeplot(train_2017_clean['logerror'].dropna(), label='logerror-2017', fill=True, alpha=0.5)
plt.title('Distribution of Logerror ')
plt.xlabel("Logerror")
plt.ylabel("Density")
plt.legend()
plt.show()


train_properties=pd.concat([train_2016_merge,train_2017_merge],axis=0).reset_index(drop=True)
print(train_properties.shape)


train_properties.head()


#时间序列化
train_properties['transactiondate']=pd.to_datetime(train_properties['transactiondate'])
train_properties['transactiondate']=train_properties['transactiondate'].dt.strftime('%Y%m%d').astype(int)


train_properties.head()


from sklearn.model_selection import train_test_split
#2016的数据的logerror分布较为集中
time=train_properties['transactiondate']
early_data=train_properties[time%100<=9]
late_2016=train_properties[(time%100>=10)&(time//10000==2016)]
train_early,test_early=train_test_split(early_data,test_size=0.2,random_state=1)
train_late,test_late=train_test_split(late_2016,test_size=0.9,random_state=2)
train=pd.concat([train_early,train_late]).reset_index(drop=True)
test=pd.concat([test_early,test_late]).reset_index(drop=True)
print(train.shape)
print(test.shape)


train_final=train.drop(['parcelid'],axis=1)
test_final=test.drop(['parcelid'],axis=1)


train_final.head()


from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X_train=train_final.drop(['logerror'],axis=1)
y_train=train_final['logerror'].values
X_test=test_final.drop(['logerror'],axis=1)
y_test=test_final['logerror'].values

sc=StandardScaler()
X_train_scaler=sc.fit_transform(X_train)
X_test_scaler=sc.transform(X_test)
xr=XGBRegressor(n_estimators=1000,random_state=1,n_jobs=-1)
model=xr.fit(X_train_scaler,y_train)
y_train_pred=model.predict(X_train_scaler)
y_test_pred=model.predict(X_test_scaler)
print('train_r2_score:',r2_score(y_train,y_train_pred))
print('test_r2_score:',r2_score(y_test,y_test_pred))


#测试集的训练效果不佳，猜测：可能参数不行，进行模型调优交叉验证优化模型
from sklearn.model_selection import RandomizedSearchCV,KFold
X=train_final.drop(['logerror'],axis=1)
y=train_final['logerror']
cv=KFold(n_splits=5,shuffle=True,random_state=45)
models={
    'XGB':{
        'model':XGBRegressor(random_state=42,n_jobs=-1),
        'param_grid':{
            'n_estimators': [500, 700], 
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
    }
}
best_params={}
best_models={}
for model_name,model_info in models.items():
    Rs=RandomizedSearchCV(
    model_info['model'],model_info['param_grid'],
        n_iter=8,
        cv=cv,scoring='neg_mean_absolute_error',
        n_jobs=-1,random_state=42,verbose=2
)
    Rs.fit(X,y)
    best_params[model_name]=Rs.best_params_
    best_models[model_name]=Rs.best_estimator_
    print('best_params:',Rs.best_params_)



#调优模型评估
X_test=test_final.drop(columns=['logerror'], errors='ignore')
y_test=test_final['logerror']
for model_name,model in best_models.items():
    y_test_pre=model.predict(X_test)
    score=r2_score(y_test,y_test_pre)
    print(f'{model_name}:',score)


plt.figure(figsize=(8,7))
plt.hist(y_test_pre,bins=50,alpha=0.5,label='XGB')
plt.ylabel('frequency')
plt.xlabel('logerror')
plt.show()


properties_2017.head()


#使用最优参数模型再进行模型训练
model=best_models['XGB']
X_test_2016=properties_2016.drop(['parcelid'],axis=1)
X_test_2017=properties_2017.drop(['parcelid'],axis=1)
X_test_2016.insert(0, 'transactiondate', np.nan)
X_test_2017.insert(0, 'transactiondate', np.nan)
submission=pd.DataFrame()
submission['ParcelId'] = properties_2016['parcelid'].astype(int)


sample_submission.head()


#预测2016
for col in ['201610','201611','201612']:
    X_test_2016['transactiondate']=int(col)
    result_2016=model.predict(X_test_2016)
    submission[col]=result_2016


submission2=pd.DataFrame()
submission2['ParcelId']=properties_2017['parcelid'].astype(int)
for col in ['201710','201711','201712']:
    X_test_2017['transactiondate']=int(col)
    result_2017=model.predict(X_test_2017)
    submission2[col]=result_2017

submission_final=pd.merge(submission,submission2,on='ParcelId',how='left')


submission_final.head()


submission_final.to_csv('/kaggle/working/submission.csv',index=False)




