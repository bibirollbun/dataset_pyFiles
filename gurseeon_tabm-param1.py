!pip install  scikit-learn==1.6.1
!pip install -qq pytabkit
#numpy==2.0.2 matplotlib==3.10.0 seaborn==0.13.2 pandas==2.2.2  xgboost==3.0.5 lightgbm==4.6.0 catboost==1.2.8 optuna==4.5.0


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns,sklearn, warnings
#import optuna
import pytabkit as tab
#import sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os,sys,shutil,warnings
from copy import  deepcopy
warnings.simplefilter("ignore")



print(f"{os.getcwd()}")
if os.getcwd() != "/kaggle/working":
  path=path2='/content/drive/MyDrive/Kaggle/PG-S/5e10/data/'
  out_path='/'
  print('>>> Colab DIR')

else:
  path='/kaggle/input/playground-series-s5e10/'
  path2='/kaggle/input/simulated-roads-accident-data/'
  out_path='/kaggle/working/'
  print('--+++--++ Kaggle DIR')



train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')

train.info()


orig=[]
for k in [2,10,100]:
  df=pd.read_csv(f'{path2}synthetic_road_accidents_{k}k.csv')
  orig.append(df)

orig=pd.concat(orig,axis=0)
orig.reset_index(drop=True,inplace=True)
orig.info()


target='accident_risk'
base=[col for col in test.columns if col not in ['id']]
cats=train.select_dtypes(include=[np.object_,np.bool_]).columns.to_list()
print(f'Base: {len(base)}\n{base}\nCat: {len(cats)}\n{cats}')


ORIG=[]
for col in base:
  tmp=orig.groupby(col)[target].mean()
  new_col_name=f'orig_{col}'
  tmp.name=new_col_name
  train=train.merge(tmp,on=col,how='left')
  test=test.merge(tmp,on=col,how='left')
  orig=orig.merge(tmp,on=col,how='left')
  ORIG.append(new_col_name)
print(f'ORIG: {len(ORIG)} \n{ORIG}')
train


from scipy.stats import norm


def f(X):
  return 0.3*X['curvature']+\
  0.2*(X['lighting']=='night').astype(int)+\
  0.1*(X['weather']!='clear').astype(int)+\
  0.2*(X['speed_limit']>=60).astype(int)+\
  0.1*(X['num_reported_accidents']>2).astype(int)

def clip(f):
  def clip_f(X):
    sigma=0.05
    mu=f(X)
    a=-mu/sigma
    b=(1-mu)/sigma
    Phi_a,Phi_b=norm.cdf(a),norm.cdf(b)
    phi_a,phi_b=norm.pdf(a),norm.pdf(b)

    return mu*(Phi_b- Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b

  return clip_f

clipped_f=clip(f)
train['y']=clipped_f(train)
test['y']=clipped_f(test)
orig['y']=clipped_f(orig)


train['orig_curvature']=train['orig_curvature'].fillna(orig[target].mean())
test['orig_curvature']=test['orig_curvature'].fillna(orig[target].mean())


features=base+ORIG+['y']
print(f'features: {len(features)} \n{features}')


both_data=False

if both_data:
  X=pd.concat([train[features],orig[features]],axis=0)
  #X.reset_index(drop=True,inplace=True)
  y=pd.concat([train[target]-train['y'],orig[target]-orig['y']],axis=0)
  #y.reset_index(drop=True,inplace=True)
  X_test=test[features]

else:
  X=train[features]
  #X.reset_index(drop=True,inplace=True)
  y=train[target]-train['y']
  #y.reset_index(drop=True,inplace=True)
  X_test=test[features]

print(f'{both_data=}')

X.shape,y.shape,X_test.shape





#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from contextlib import contextmanager
import sys
@contextmanager
def suppress_stdout():
  with open(os.devnull,'w') as devnull:
    old_stdout=sys.stdout
    sys.stdout=devnull
    try:
      yield
    finally:
      sys.stdout=old_stdout




params_tab={'patience': 18,"allow_amp":True,'arch_type':'tabm-mini',
            'tabm_k': 125,"share_training_batches": False,
            'gradient_clipping_norm': 0.055795068607357085,
            'lr': 0.0006139606665573677,"batch_size":'auto' ,
            'weight_decay': 0.008993214640470116,
            'n_blocks': 5, 'd_block': 503,"num_emb_type": 'pwl',
            'num_emb_n_bins': 119,
            'dropout': 0.015170216720376377, 'd_embedding': 16}
param_2={'patience': 18,"allow_amp":True,'arch_type':'tabm-mini',
            'tabm_k': 125,"share_training_batches": False,
            'gradient_clipping_norm': 0.055795068607357085,
            'lr': 0.0006139606665573677,"batch_size":'auto' ,
            'weight_decay': 0.008993214640470116,
            'n_blocks': 5, 'd_block': 503,"num_emb_type": 'pwl',
            'num_emb_n_bins': 119,"device":'cuda',
            'dropout': 0.015170216720376377, 'd_embedding': 16
         }  # Best is trial 0 with value: 0.05605149096498026
param_3=deepcopy(param_2)

param_3.update({'patience': 14, 'tabm_k': 54,
                'gradient_clipping_norm': 0.42520050162003964,
                'lr': 0.00046318583248775225, 'weight_decay': 0.005279674345228175,
                'n_blocks': 4, 'd_block': 161, 'dropout': 0.13650844749411428,
                'd_embedding': 23})# Best is trial 4 with value: 0.05605021387151689.

param_2_1=deepcopy(param_2) # this will use
param_2_1.update({'patience': 14, 'tabm_k': 90,
                  'gradient_clipping_norm': 0.8301321817909664,
                  'lr': 0.00015662898808778292,
                  'weight_decay': 0.002536089631584974,  'n_blocks': 3,
                  'd_block': 326, 'dropout': 0.05679846432052377,
                  'd_embedding': 12}) #0.05603729889691921

N_splits=3


#param_3


from pytabkit import TabM_D_Regressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


from tqdm import tqdm


kf=KFold(n_splits=N_splits,shuffle=True,random_state=42)
oof_preds=np.zeros(len(X))
test_preds=np.zeros(len(test))
for fold,(train_idx,val_idx) in tqdm(enumerate(kf.split(X,y))):

  X_train,y_train=X.iloc[train_idx],y.iloc[train_idx]
  X_val,y_val=X.iloc[val_idx],y.iloc[val_idx]
  with suppress_stdout():
    model=TabM_D_Regressor(**param_2_1)
    model.fit(X_train,y_train,X_val,y_val,cat_col_names=cats)

  oof_pred=model.predict(X_val)
  test_preds+=model.predict(X_test)
  oof_preds[val_idx]=oof_pred
  print(f">> Fold: {fold+1}/{N_splits}",end='  ')
  print(f"RMSE: {root_mean_squared_error(y_val+X.iloc[val_idx].y,oof_pred+X.iloc[val_idx].y):.6f}",end=' ')
  print(f"R2: {r2_score(y_val+X.iloc[val_idx].y,oof_pred+X.iloc[val_idx].y):.6f}")

test_preds/=N_splits
y_true_final=y.to_numpy()+X.y.to_numpy()
y_pred_final=oof_preds+X.y.to_numpy()
print(f"\nOverall RMSE: {root_mean_squared_error(y_true_final,y_pred_final):.6f}",end="  ")
print(f"R2: {r2_score(y_true_final,y_pred_final):.6f}")


sub=pd.DataFrame({'id':test['id'],target:test_preds+test['y'].to_numpy()})
sub.to_csv('sub_TabM_param_3.csv',index=False)
sub


sub.to_csv('submission.csv',index=False)


#!kaggle competitions submit -c playground-series-s5e10 -f sub_TabM_param_3.csv -m "TabM_param_3"

