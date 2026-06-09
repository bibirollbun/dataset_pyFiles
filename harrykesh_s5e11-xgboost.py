import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib inline
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import warnings 
from sklearn.metrics import *


from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
org = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
df.head()


org.head()


print(f'Nulls in training: {df.isnull().sum().sum()}')


nums = df.select_dtypes(include='number').columns.tolist()
cats = df.select_dtypes(exclude='number').columns.tolist()


fig,ax = plt.subplots(3,int(len(nums)//3)+1,figsize=(12,8))
ax = ax.flatten()

for idx,feats in enumerate(nums):
    df[feats].plot(kind='hist',ax=ax[idx])
    ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(cats)//3),figsize=(15,6))
ax = ax.flatten()

for idx,feats in enumerate(cats):
    # df[feats].plot(kind='hist',ax=ax[idx])
    sns.countplot(
        data=df,
        x = feats,
        hue=df['loan_paid_back'],
        ax = ax[idx]
    )
    # ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(3,int(len(nums)//3)+1,figsize=(12,8))
ax = ax.flatten()

for idx,feats in enumerate(nums):
    sns.histplot(
        data=df,
        x = feats,
        hue='loan_paid_back',
        ax=ax[idx]
    )
    ax[idx].set_xlabel(feats)
plt.tight_layout()
plt.show()


def Process(data,org_data,MAP=None,cats=cats,nums=nums,target='loan_paid_back'):
    df = data.copy()
    eps=1e-5
    ##target encoding  from the original dataset 
    Map1,Map2 = {},{}
    for feat in cats:
        Map1[feat] = org_data.groupby(feat)[target].mean()
        df[f'_org_TE_{feat}'] = df[feat].map(Map1[feat])

    #numerical_features 
    df['loanxanual_income'] = df['loan_amount']/(eps+df['annual_income'])
    df['credit_utilizisation'] = df['debt_to_income_ratio'] /(eps+df['annual_income'])

    #target encoding from the training dataset
    if MAP is None:
        for feat in cats:
            Map2[feat] = df.groupby(feat)[target].mean()
            df[f'_TE_{feat}'] = df[feat].map(Map2[feat])
    else:
        for feat in cats:
            # Map2[feat] = df.groupby(feat)[target].mean()
            df[f'_TE_{feat}'] = df[feat].map(MAP[feat])

    #Cross Binning
    cross_columns = [
        ('gender','marital_status'),
        ('education_level','employment_status'),
        # ('loan_purpose','grade_subgrade'),
        ('loan_purpose','marital_status'),
        ('loan_purpose','gender')
    ]

    for cross in cross_columns:
        f1 = cross[0]
        f2 = cross[1]
        df[f'{f1}x{f2}'] = df[f1] + "_" +df[f2]
    cats = df.select_dtypes(exclude='number').columns.tolist()

    df= pd.get_dummies(df,prefix=cats,columns=cats)
    df.drop(columns=['id'],inplace=True)
    return df,Map2


splitter = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)


score = []
for fold_idx,(train_idx,val_idx) in enumerate(splitter.split(np.zeros(len(df)),df.loan_paid_back)):
    print(f'Fold:{fold_idx+1}')
    train,val = df.iloc[train_idx],df.iloc[val_idx]

    train,mAp = Process(train,org)
    X,y = train.drop(columns=['loan_paid_back']),train.loan_paid_back

    val,_ = Process(val,org,MAP=mAp)
    valx,valy = val.drop(columns=['loan_paid_back']),val.loan_paid_back

    neg_count = (y==0).sum()
    pos_count = (y==1).sum()

    model = xgb.XGBClassifier(scale_pos_weight=neg_count/pos_count,
                              n_estimators=10000,
                              subsample=0.8,
                              learning_rate=0.01,
                              device='gpu',
                              early_stopping_rounds=100,
                              verbose=0,
                              random_state=42)

    model.fit(X,y,
             eval_set=[(valx,valy)],
             verbose=0)
    PROB = model.predict_proba(valx)[:,1]

    score.append(roc_auc_score(valy,PROB))

print(np.mean(score))
print(score)





