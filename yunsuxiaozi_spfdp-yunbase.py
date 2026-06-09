source_file_path = '/kaggle/input/yunbase/Yunbase/baseline.py'
target_file_path = '/kaggle/working/baseline.py'
with open(source_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
with open(target_file_path, 'w', encoding='utf-8') as file:
    file.write(content)


!pip install -q --requirement /kaggle/input/yunbase/Yunbase/requirements.txt  \
--no-index --find-links file:/kaggle/input/yunbase/


from baseline import Yunbase
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)


train=pd.read_csv("/kaggle/input/stock-pledge-defaults-prediction/train.csv").fillna(-1)
test=pd.read_csv("/kaggle/input/stock-pledge-defaults-prediction/test.csv").fillna(-1)
train.head()


#nunique==2
NUNIQUE=['Downgrade or negative', 'Company nature (state owned assets 0, others 1)',
          'Whether there are four major audits', 
          'Two positions in one (1 for the same, 0 for the different)']
def FE(df):
    for i in range(len(NUNIQUE)):
        for j in range(i+1,len(NUNIQUE)):
            df[NUNIQUE[i]+"_"+NUNIQUE[j]]=df[NUNIQUE[i]].astype(str)+df[NUNIQUE[j]].astype(str)
    return df

COM=[]
for i in range(len(NUNIQUE)):
    for j in range(i+1,len(NUNIQUE)):
        COM.append(NUNIQUE[i]+"_"+NUNIQUE[j])


total=COM+NUNIQUE
target_stat=[]
for c in total:
    target_stat.append( (c,'IsDefault',['mean','count'] ) )


drop_cols=[
           #1
           'Share pledge ratio of controlling shareholders','Enterprise age',
           'Debt financing costs','Financial cycle m2/gdp','Tobin Q',
           'Pledge ratio of limited sale shares',
           #2
           'Current ratio','Cash income ratio','Asset quality index',
           'SG&A Expense','Ratio of other receivables to total assets',
           'Average cash income ratio in recent three years',
           #3
           'Goodwill impairment ratio','Current liabilities/total liabilities',
           'Monetary capital/short-term debt',"Minority shareholders' equity/owners' equity",
            'P/B ratio',
           #4
           'Number of key audit matters','Equity concentration (the first largest shareholder)',
           'changes in operating income','Total asset turnover rate (Times)',
           'Asset liability ratio (excluding advance receipts)',
           #5
           'Equity checks and balances (2-5 large/1 large)', 'Inventory turnover rate (Times)',
           'Annual turnover rate','Stock price rise and fall in the last year',
           #6
           'changes in net assets','Cash ratio','Z-SCORE','Ratio of construction in progress to total assets',
          ]


lgb_params={"boosting_type": "gbdt","metric": 'auc',
            'random_state': 2025, "n_estimators": 300,
            'importance_type': 'gain',#better than 'split'
            }

yunbase=Yunbase(
    num_folds=10,
    objective='binary',
    metric='f1_score',
    drop_cols=['Stock code']+drop_cols,
    FE=FE,
    models=[(LGBMClassifier(**lgb_params),'lgb')],
    early_stop=512,
    num_classes=2,
    target_col='IsDefault',
    use_high_corr_feat=False,
    plot_feature_importance=True,
)
yunbase.fit(train)
test_preds=yunbase.predict(test)


yunbase.adversarial_validation(train.drop(drop_cols,axis=1),
                               test.drop(drop_cols,axis=1),
                               target_col='IsDefault'
                              )


oof=np.load("/kaggle/working/Yunbase_info/lgb_seed2025_repeat0_fold10_IsDefault.npy")[:,1]
best_margin,best_score,margin=0,0,1000
for m in range(margin):
    temp=(oof>=m/margin).astype(np.int8)
    score=yunbase.Metric(yunbase.target,temp)
    if score>best_score:
        best_score=score
        best_margin=m/margin
print(f"best_score:{best_score},best_margin:{best_margin}")
test_preds=np.load("/kaggle/working/Yunbase_info/IsDefault_test_preds.npy")[0][:,1]
test_preds=(test_preds>=best_margin).astype(np.int8)
print(f"train_set IsDefault mean:{yunbase.train['IsDefault'].mean()},test_set IsDefault mean:{test_preds.mean()}")


submission=pd.read_csv("/kaggle/input/stock-pledge-defaults-prediction/test.csv")[['Stock code']]
yunbase.submit(submission,save_name='yunbase',test_preds=test_preds)
pd.read_csv("yunbase.csv").head()

