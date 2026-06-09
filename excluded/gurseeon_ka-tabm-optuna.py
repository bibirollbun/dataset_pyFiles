!pip install  scikit-learn==1.6.1  optuna
!pip install -qq pytabkit
#numpy==2.0.2 matplotlib==3.10.0 seaborn==0.13.2 pandas==2.2.2 xgboost==3.0.5 lightgbm==4.6.0 catboost==1.2.8 optuna==4.5.0


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os,shutil,optuna,sklearn, warnings
import pytabkit as tab
#import sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pytabkit import TabM_D_Regressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import sys
from optuna.storages import RDBStorage
from contextlib import contextmanager
from scipy.stats import norm
from copy import deepcopy
warnings.simplefilter("ignore",category=FutureWarning)



print(f"{os.getcwd()}")
if os.getcwd() != "/kaggle/working":
  path=path2='/content/drive/MyDrive/Kaggle/PG-S/5e10/data/'
  path3='/content/'
  out_path='/'
  print('>>> Colab DIR')

else:
  path='/kaggle/input/playground-series-s5e10/'
  path2='/kaggle/input/simulated-roads-accident-data/'
  path3='/kaggle/input/ka-tabm-optuna/'
  out_path=''
  print('--+++--++ Kaggle DIR')



train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
sub=pd.read_csv(path+'sample_submission.csv')
#train.shape


orig=[]
for k in [2,10,100]:
  df=pd.read_csv(f'{path2}synthetic_road_accidents_{k}k.csv')
  orig.append(df)

orig=pd.concat(orig,axis=0)
orig.shape



target='accident_risk'
both_data=False
base=[col for col in test.columns if col not in ['id']]
cats=train.select_dtypes(include=[np.object_,np.bool_]).columns.to_list()
print(f'base: {len(base)} {len(cats)} ')


ORIG=[]
for col in base:
  tmp=orig.groupby(col)[target].mean()
  new_col_name=f'orig_{col}'
  tmp.name=new_col_name
  train=train.merge(tmp,on=col,how='left')
  test=test.merge(tmp,on=col,how='left')
  orig=orig.merge(tmp,on=col,how='left')
  ORIG.append(new_col_name)
print(f'ORIG: {len(ORIG)} \n')
#train.shape



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
print(f'features: {len(features)} ')


if both_data:
  X=pd.concat([train[features],orig[features]],axis=0)
  y=pd.concat([train[target]-train['y'],orig[target]-orig['y']],axis=0)
  X_test=test[features]

else:
  X=train[features]
  y=train[target]-train['y']
  X_test=test[features]

print(f'{both_data=}')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


@contextmanager
def suppress_stdout():
  with open(os.devnull,'w') as devnull:
    old_stfout=sys.stdout
    sys.stdout=devnull
    try:
      yield
    finally:
      sys.stdout=old_stfout




param_1={'batch_size':'auto',
               'patience':16,
               'allow_amp':True,
               'arch_type':'tabm-mini',
               'tabm_k':32,
               'gradient_clipping_norm':1.0,
               'share_training_batches':False,
               'lr':0.0029993695720154537,
               'weight_decay':0.023742083301699905,
               'n_blocks':3,
               'd_block':448,
               'dropout':0.0,
               'num_emb_type':'pwl',
               'd_embedding':32,
               'num_emb_n_bins':119
              }

param_2={'patience': 18,"allow_amp":True,'arch_type':'tabm-mini',
            'tabm_k': 125,"share_training_batches": False,
            'gradient_clipping_norm': 0.055795068607357085,
            'lr': 0.0006139606665573677,"batch_size":'auto' ,
            'weight_decay': 0.008993214640470116,
            'n_blocks': 5, 'd_block': 503,"num_emb_type": 'pwl',
            'num_emb_n_bins': 119,"device":'cuda',
            'dropout': 0.015170216720376377, 'd_embedding': 16
         }  # RMSE value: 0.05605149096498026 and 0.056035622571093374 in v9
param_2_1=deepcopy(param_2)
param_2_1.update({'patience': 14, 'tabm_k': 90,
                  'gradient_clipping_norm': 0.8301321817909664,
                  'lr': 0.00015662898808778292,
                  'weight_decay': 0.002536089631584974,  'n_blocks': 3,
                  'd_block': 326, 'dropout': 0.05679846432052377,
                  'd_embedding': 12}) #0.05603729889691921


param_2_2=deepcopy(param_2)

param_2_2.update({'patience': 14, 'tabm_k': 54,
                'gradient_clipping_norm': 0.42520050162003964,
                'lr': 0.00046318583248775225, 'weight_decay': 0.005279674345228175,
                'n_blocks': 4, 'd_block': 161, 'dropout': 0.13650844749411428,
                'd_embedding': 23})# Best is trial 4 with value: 0.05605021387151689.


resume_study=False
study_name='TabM_study_6_try'


def objective(trial,X_train,y_train,X_val,y_val):
  params={
      "batch_size":'auto',
      "patience":trial.suggest_int("patience",10,20),
      "allow_amp":True,
      "arch_type":"tabm-mini",
      "tabm_k":trial.suggest_int("tabm_k",16,128),
      "gradient_clipping_norm":trial.suggest_float("gradient_clipping_norm",0.0,1.0),
      "share_training_batches":False,
      "lr":trial.suggest_float("lr",1e-5,1e-3),
      "weight_decay":trial.suggest_float("weight_decay",0.001,0.01),
      "n_blocks":trial.suggest_int("n_blocks",1,5),
      "d_block":trial.suggest_int("d_block",32,512),
      "dropout":trial.suggest_float("dropout",0.0,0.5),
      "num_emb_type":"pwl",
      "device":'cuda',
      "d_embedding":trial.suggest_int("d_embedding",8,32),
      'num_emb_n_bins':119
  }

  with suppress_stdout():
    model=TabM_D_Regressor(**params)
    model.fit(X_train,y_train,X_val,y_val,cat_col_names=cats)

  y_true=y_val+X_val['y']
  y_pred=model.predict(X_val)+X_val['y']
  return root_mean_squared_error(y_true,y_pred)

if resume_study:
  study=optuna.create_study(study_name='TabM_study_5',direction='minimize',
                storage='sqlite:////kaggle/input/ka-tabm-optuna/TAbM_5.db', load_if_exists=True)

else:
  study=optuna.create_study(study_name=study_name,direction='minimize',
                storage=RDBStorage(url=f"sqlite:///{study_name}.db"))
  #study.enqueue_trial(param_2)
  #study.enqueue_trial(param_2_1)
  study.enqueue_trial(param_2_2)

study.optimize(lambda trial: objective(trial,X_train,y_train,X_val,y_val),
               n_trials=1,timeout=41710)#,show_progress_bar=True)

study.best_params


df_trial=study.trials_dataframe()
df_trial.to_csv(f'{study_name}_df.csv',index=False)
df_trial


study.best_params


study.best_value


study.best_trial


study.best_trials


study.get_trials()


#os.listdir("/content/drive/MyDrive/Kaggle/PG-S/5e10/data")

