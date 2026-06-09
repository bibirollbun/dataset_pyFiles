import pandas as pd
data=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
data


data.isnull().sum()


print(f"Len of the rainfall dataset is {len(data)}")
print('------------------------')
print(f"variabel of  dataset are {data.columns.tolist()}")



data.dtypes


days=data['day']
#get date 2024 and matching with day
date_per=pd.to_datetime("2024-01-01") + pd.to_timedelta(days,unit='D')
matching_data=pd.DataFrame({
    'day':days,
    'date':date_per
})
#convert to period Q and acces string using. str and show only last two words
matching_data['quarter']=date_per.dt.to_period('Q').astype(str).str[-2:]
matching_data.head()



data['date']=pd.to_datetime("2024-01-01") + pd.to_timedelta(days,unit='D')
data['quarter']=data['date'].dt.to_period('Q').astype(str).str[-2:]
data.head()


group_pressure=data.groupby('quarter')['pressure'].mean()
group_pressure.values


#defining column x variable
kol_x=[i for i in data.columns.tolist() if i!='id' and i!='day' and i!='date' and i!='rainfall' and i!='quarter']
#make line plot for all x variable
from matplotlib import pyplot as plt
nrow=2
ncol=5
fig,ax=plt.subplots(nrow,ncol,figsize=(11,8))
for (i,j),k in zip([(i,j) for i in range(nrow) for j in range(ncol)], kol_x):
    group_data=data.groupby('quarter')[k].mean()
    xtic=[i for i in range(4)]
    plotting=ax[i][j].plot(group_data.index,group_data.values,marker='o')
    ax[i][j].set_xticks(xtic)
    ax[i][j].spines['top'].set_visible(False)
    ax[i][j].spines['right'].set_visible(False)
    ax[i][j].spines['left'].set_visible(False)
    ax[i][j].set_yticks([])
    xval=plotting[0].get_xdata()
    yval=plotting[0].get_ydata()
    for l,m in zip(xval,yval):
        ytex=round(m,2)
        ax[i][j].text(l,m,f"{ytex}",fontsize=8)
    ax[i][j].set_title(f"{k}",loc='left',fontweight='bold',fontsize=8)
    
plt.tight_layout()




fig,ax=plt.subplots(1,4,figsize=(10,10))
qua=data['quarter'].unique().tolist()
for i,j in zip(ax,qua):
    tabel_freq=data[data['quarter']==j]['rainfall'].value_counts()
    i.pie(tabel_freq.values,labels=tabel_freq.index,wedgeprops=dict(width=0.3))
    i.set_title(f"distribution of rainfall in \n{j}", fontsize=10)
plt.tight_layout()


#check corelation
corr=data[kol_x].corr()
corr


#temperature vs dew point
rainfall_type=data['rainfall'].unique().tolist()
for i,j in zip(rainfall_type,['blue','yellow']):
    ambil=data[data['rainfall']==i]
    plt.scatter(ambil['temparature'],ambil['dewpoint'],color=j,label=i)
    plt.xlabel('temparature')
    plt.ylabel('dewpoint')
    plt.title('temparature vs dewpoint')
    plt.legend()


#presuere vs dew point
fig,ax=plt.subplots(1,2,figsize=(5,3))
for i,j in zip(range(2),['temparature','dewpoint']):
    rainfall_type=data['rainfall'].unique().tolist()
    for k,l in zip(rainfall_type,['blue','yellow']):
        ambil=data[data['rainfall']==k]
        ax[i].scatter(ambil['pressure'],ambil[j],color=l,label=i)
        ax[i].set_xlabel('presurre')
        ax[i].set_ylabel(j)
        ax[i].set_title(f'presurre vs {j}')
        ax[i].legend()
plt.tight_layout()



#cloud and shunshine
rainfall_type=data['rainfall'].unique().tolist()
for i,j in zip(rainfall_type,['blue','yellow']):
    ambil=data[data['rainfall']==i]
    plt.scatter(ambil['cloud'],ambil['sunshine'],color=j,label=i)
    plt.xlabel('cloud')
    plt.ylabel('sunshine')
    plt.title('cloud vs sunshine')
    plt.legend()


data['rainfall'].unique().tolist()


fig,ax=plt.subplots(2,5,figsize=(8,5))
for (i,j),k in zip([(i,j) for i in range (2) for j in range (5)],kol_x):
    kum=[]
    for l in data['rainfall'].unique().tolist():
        ambil=data[data['rainfall']==l]
        get_data=ambil[k]
        kum.append(get_data)
    ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    ax[i][j].set_title(k)
plt.tight_layout()


print(f"class label in this dataset is {data['rainfall'].unique().tolist()}")
print('---------------------')
data['rainfall'].value_counts()


data_new=data.drop(['id','day'],axis=1)


from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
data_cl=data_new.copy()
kol_log=['cloud','sunshine','windspeed','humidity']
X=data_cl[kol_log]
y=data_cl['rainfall']
#split teh dara into validation set, stratify keep distribution of class is same
xtrain,xval,ytrain,yval=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
xtrain.head()


#make kf
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
#since imbalance class we use roc_auc score
roc_train=[]
roc_val=[]
for tr_index,val_index in kf.split(xtrain,ytrain):
    xtrain_fold,xval_fold=xtrain.iloc[tr_index],xtrain.iloc[val_index]
    ytrain_fold,yval_fold=ytrain.iloc[tr_index],ytrain.iloc[val_index]
     #scaler
    scaler=StandardScaler()
    tr_clean=scaler.fit_transform(xtrain_fold)
    val_clean=scaler.transform(xval_fold)
    #make model
    model_log=LogisticRegression(class_weight='balanced',solver='liblinear',random_state=42)
    model_log.fit(tr_clean,ytrain_fold)
    #extract only columns 1 column 1 is positive class
    tr_pred=model_log.predict_proba(tr_clean)[:,1]
    val_pred=model_log.predict_proba(val_clean)[:,1]
    #score
    score_tr=roc_auc_score(ytrain_fold,tr_pred)
    score_val=roc_auc_score(yval_fold,val_pred)
    roc_train.append(score_tr)
    roc_val.append(score_val)
print('roc auc score training :')
print('-----------------------------------------')
print(roc_train)
print('-----------------------------------------')
print('roc auc score validation :')
print('-----------------------------------------')
print(roc_val)


#perform logistic regression in validation
standar=StandardScaler()
val_new=standar.fit_transform(xval)
pred=model_log.predict_proba(val_new)[:,1]
rocscore=roc_auc_score(yval,pred)
rocscore


import pandas as pd
data_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
data_test.tail()


X_test=data_test[kol_log]
xtest_str=standar.transform(X_test)
predict_test=model_log.predict_proba(xtest_str)[:,1]
#threshold 0.5 if more than 0,5 we conver to 1
threshold=0.5
def binary_class(x):
    if x>=0.5:
        return 1
    else:
        return 0
binary=list(map(binary_class,predict_test))
datakum=pd.DataFrame({
    'id':data_test['id'],
    'class':binary
})
datakum


datakum['class'].value_counts()


from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost
from sklearn.metrics import roc_auc_score
data_xg=data_new.copy()
X=data_xg[kol_x]
y=data_xg['rainfall']
#split teh dara into validation set, stratify keep distribution of class is same
xtrain,xval,ytrain,yval=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
ytrain.shape


#make kf
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
tab_count=y.value_counts()
scale=tab_count.loc[1]/tab_count.loc[0]
#since imbalance class we use roc_auc score
roc_train=[]
roc_val=[]
for tr_index,val_index in kf.split(xtrain,ytrain):
    xtrain_fold,xval_fold=xtrain.iloc[tr_index],xtrain.iloc[val_index]
    ytrain_fold,yval_fold=ytrain.iloc[tr_index],ytrain.iloc[val_index]
    #make model
    model_xg=xgboost.XGBClassifier(scale_pos_weight=scale,
                                    eval_metric='auc',
                                    max_depth=2,
                                    reg_alpha=0.6,
                                    reg_lambda=9,
                                    n_estimators=200,
                                   gamma=5)
    model_xg.fit(xtrain_fold,ytrain_fold)
    #extract only columns 1 column 1 is positive class
    tr_pred=model_xg.predict_proba(xtrain_fold)[:,1]
    val_pred=model_xg.predict_proba(xval_fold)[:,1]
    #score
    score_tr=roc_auc_score(ytrain_fold,tr_pred)
    score_val=roc_auc_score(yval_fold,val_pred)
    roc_train.append(score_tr)
    roc_val.append(score_val)
print('roc auc score training :')
print('-----------------------------------------')
print(roc_train)
print('-----------------------------------------')
print('roc auc score validation :')
print('-----------------------------------------')
print(roc_val)


#perform logistic regression in validation
pred=model_xg.predict_proba(xval)[:,1]
rocscore=roc_auc_score(yval,pred)
rocscore


import pandas as pd
data_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
data_test.tail()


X_test=data_test[kol_x]
predict_test=model_xg.predict_proba(X_test)[:,1]
#threshold 0.5 if more than 0,5 we conver to 1
threshold=0.5
def binary_class(x):
    if x>=0.5:
        return 1
    else:
        return 0
binary=list(map(binary_class,predict_test))
datakum=pd.DataFrame({
    'id':data_test['id'],
    'class':binary
})
datakum


datakum['class'].value_counts()

