pip install autoviml shap


import pandas as pd
import numpy as np


df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
og=pd.read_csv('/kaggle/input/original-123324weax/bank-full.csv',delimiter=';')
te=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df.drop(columns='id',inplace=True)

og['y']=og['y'].map({'yes':'1','no':'0'})

df['y']=df['y'].astype(str)

df=pd.concat([df,og],ignore_index=True)


train=df
test=te
target='y'


from autoviml.Auto_ViML import Auto_ViML

model,features,trainm,testm=Auto_ViML(
    train,
    target,
    test,
    Add_Poly=3,
    Imbalanced_Flag=True,
    Binning_Flag=False,
    KMeans_Featurizer=False,
    feature_reduction=True,
    Boosting_Flag=False,
    Stacking_Flag=False,
    hyper_param='GS',
    scoring_parameter='roc_auc',
    sample_submission='',
    verbose=2

)


!nvidia-smi


testm.head()


last_col=testm.columns[-1]
submission=pd.DataFrame(
    {
        'id':test['id'],
        'y':testm[last_col]
    }
)
submission.to_csv('submission.csv',index=False)

