%load_ext cudf.pandas

import gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

VER=1


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",index_col='id')
print("Train shape", train.shape )
train.head()


train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')
print("Extra Train shape", train2.shape )
train2.head()


train = pd.concat([train,train2],axis=0,ignore_index=True)
print("Combined Train shape", train.shape)


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv",index_col='id')
print("Test shape", test.shape )
test.head()


del train2


target_col='Price'
target=train.pop(target_col)


catcols = train.columns.to_list()[:-1]
numcols = [train.columns.to_list()[-1]]


catcol2=[]
for col in catcols:
    if train[col].nunique()>4:
        catcol2.append(col)
    print(f'{col}: {train[col].nunique()}')


for col in catcols:
    temp=pd.concat([train[col],test[col]],axis=0)
    temp,_=pd.factorize(temp)
    train[col]=temp[:len(train)]
    test[col]=temp[len(train):]


train[catcols]=train[catcols].astype(np.int16)
test[catcols]=test[catcols].astype(np.int16)
train[numcols]=train[numcols].astype(np.float32)
test[numcols]=test[numcols].astype(np.float32)
target=target.astype(np.float32)


train=train.fillna(-1)
test=test.fillna(-1)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS

#These are cudF "slow functions"
Q25=lambda x: x.quantile(0.25)
Q75=lambda x: x.quantile(0.75)
IQR=lambda x: x.quantile(0.75)-x.quantile(0.25)

def mode(serie):
    s=serie.value_counts().to_dict()
    s=sorted(list(s.items()),key=lambda x:x[1])
    return s[-1][0]

def antimode(serie):
    s=serie.value_counts().to_dict()
    s=sorted(list(s.items()),key=lambda x:x[1])
    return s[0][0]
    
def entropy(serie):
    p=serie.value_counts().values
    p=p/p.sum()
    return -(p*np.log(p)).sum()

num_functions=["mean","std","count","nunique","median","min","max","skew"]#,('Q25',Q25),('Q75',Q75),('IQR',IQR)]
cat_functions=["count","nunique","median",('entropy',entropy),('mode',mode),('antimode',antimode)]


def FT_encode(train,test,target=None,col1=[],col2=[],cat_functions=[],num_functions=[],numcols=[],catcols=[],nfolds=5,debug=False):

    def filter_agg(temp):
        todrop=temp.describe().T
        todrop=todrop[todrop['std']<1e-16].index.to_list()
        return temp.drop(todrop,axis=1)

    X_train = train.reset_index(drop=True).copy()
    train_columns=X_train.columns.to_list()
    if type(target)!= type(None):
        y_train = target.copy()

    X_test = test.reset_index(drop=True).copy()
    if debug and col2:
        print('Computing FE:')
    for c1 in col1:
        for c2 in col2:
            if c1==c2:
                continue
            print(f'\t{c1} - {c2}')
            if c2 in numcols:
                functions=num_functions
            else:
                functions=cat_functions
            tmp=X_train.groupby(c1)[c2].agg(functions)
            tmp=filter_agg(tmp)
            tmp.columns=[f'FE_{c1}_{c2}_{item}' for item in tmp.columns]
            X_train = X_train.merge(tmp, on=c1, how="left")
            X_test = X_test.merge(tmp, on=c1, how="left")

    if type(target)!= type(None):
        if debug:
            print('Computing TE for train')
        c2='target_col'
        X_train['target_col']=y_train
        if y_train.nunique()>30:
            functions=num_functions
        else:
            functions=cat_functions
        
        skf2=KFold(n_splits=nfolds, shuffle=True, random_state=42)
        for j, (train_index2, test_index2) in enumerate(skf2.split(X_train)):
            if debug:
                print(f"\tFold {j+1}",end=': ')
            X_train2 = X_train.loc[train_index2,train_columns+['target_col']].copy()
            X_valid2 = X_train.loc[test_index2,train_columns+['target_col']].copy()
            for c1 in col1:
                if debug:
                    print(f'{c1}',end=', ')
                tmp=X_train2.groupby(c1)['target_col'].agg(functions)
                tmp=filter_agg(tmp)
                tmp.columns=[f'TE_{c1}_{item}' for item in tmp.columns]
                
                X_valid2 = X_valid2.merge(tmp, on=c1, how="left")
                
                for c in tmp.columns:
                    X_train.loc[test_index2,c] = X_valid2[c].values
            if debug:
                print() 
        del X_train2, X_valid2
        if debug:
            print('Computing TE for test',end='\n\t')
        for c1 in col1:
            if debug:
                print(f'{c1}',end=', ')
            tmp=X_train.groupby(c1)['target_col'].agg(functions)
            tmp=filter_agg(tmp)
            tmp.columns=[f'TE_{c1}_{item}' for item in tmp.columns]
            X_test = X_test.merge(tmp, on=c1, how="left")
        X_train=X_train.drop('target_col',axis=1)
    return X_train,X_test


enc_train,enc_test=FT_encode(train, test, target=target,
          cat_functions=cat_functions, num_functions=num_functions,
          col1=catcol2+numcols, col2=numcols,
          numcols=numcols, catcols=catcols,
          nfolds=7, debug=True)


print(f'#features {enc_train.shape[1]}\n NaN values: {enc_train.isnull().sum().sum()}')



for col in enc_train.columns:
    if (nnull:=enc_train[col].isnull().sum())>0:
        print(f'{col}: {nnull}')


for col in enc_train.columns:
    if (nnull:=enc_test[col].isnull().sum())>0:
        print(f'{col}: {nnull}')


temp_train,temp_test=FT_encode(train[catcols], test[catcols], target=None,
          cat_functions=cat_functions, num_functions=num_functions,
          col1=catcol2, col2=catcol2,
          numcols=numcols, catcols=catcols,
          nfolds=7, debug=True)


print(f'#features {temp_train.shape[1]}\n NaN values: {temp_train.isnull().sum().sum()}')


columns=[item for item in temp_train.columns if not item in catcols+numcols]
enc_train[columns]=temp_train[columns].values
enc_test[columns]=temp_test[columns].values


print(f'#features {enc_train.shape[1]}')



del train,test, temp_train,temp_test
gc.collect()


for col in enc_train.columns:
    if '64' in str(enc_train[col].dtype):
        print(f'{col}: {enc_train[col].min()} - {enc_train[col].max()}')
        if 'int' in str(enc_train[col].dtype):
            enc_train[col]=enc_train[col].astype(np.int32)
            enc_test[col]=enc_test[col].astype(np.int32)
        else:
            enc_train[col]=enc_train[col].astype(np.float32)
            enc_test[col]=enc_test[col].astype(np.float32)


enc_train[catcols]=enc_train[catcols].astype('category')
enc_test[catcols]=enc_test[catcols].astype('category')


%%time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(enc_train)))
pred = np.zeros((len(enc_test)))


# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(enc_train)):
    print(f"### Fold {i+1} ###")

    X_train = enc_train.loc[train_index].reset_index(drop=True).copy()
    y_train = target.loc[train_index]

    X_valid = enc_train.loc[test_index].reset_index(drop=True).copy()
    y_valid = target.loc[test_index]


    # BUILD MODEL
    model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
    
    # TRAIN MODEL
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],  
        verbose=300,
    )
    # PREDICT OOF AND TEST
    oof[test_index] = model.predict(X_valid).astype(np.float32)
    pred += model.predict(enc_test).astype(np.float32)

pred /= FOLDS


# COMPUTE OVERALL CV SCORE

s = np.sqrt(np.mean( (oof-target)**2.0 ) )
print(f"=> Overall CV Score = {s}")


# SAVE OOF TO DISK FOR ENSEMBLES
np.save(f"oof_v{VER}",oof)
print("Saved oof to disk")


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()




