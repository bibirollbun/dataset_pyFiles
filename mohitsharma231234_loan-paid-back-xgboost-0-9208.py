# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_d = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


import warnings
warnings.simplefilter(action = 'ignore',category = FutureWarning)


# importing all libraries
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score
from xgboost.callback import EarlyStopping
from sklearn.linear_model import LogisticRegressionCV


train_d.head()


train_d.shape


train_d.describe()


train_d.info()


# let's check for null values
train_d.isnull().sum()



train_d.nunique()


print("printing the unique values of the columns")


for col in train_d:
    unq = train_d[col].unique()
    print("-"*60)
    if len(unq) < 15:
        print(f"{col} : {unq}")
    else:
        print(f"{unq[:5]} \n There are So much Unique Vlaues")


sns.set_style("darkgrid")
num_cols = train_d.select_dtypes(include = ['int64','float64']).columns
plt.figure(figsize = (14,len(num_cols)*3))

for idx,col in enumerate(num_cols,1):
    plt.subplot(len(num_cols),2,idx)
    sns.histplot(train_d[col],kde = True,bins = 20)
    plt.title(f"Histplot of {col}")

plt.tight_layout()
plt.show()


# barplot for categorical columns 
object_cols = train_d.select_dtypes(include = ['object']).columns
plt.figure(figsize= (14,len(object_cols)*3))

for i, col in enumerate(object_cols,1):
    plt.subplot(len(object_cols),2,i)
    sns.countplot(x = col,data = train_d)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation = 90)

plt.tight_layout()
plt.show()


# barplot of target varivable 
counts = train_d['loan_paid_back'].value_counts()
plt.figure(figsize = (8,6))

plt.bar(counts.index,counts)
plt.title('Barplot of Target Columns')
plt.xlabel('Loan_Paid_Back')
plt.ylabel('counts')
plt.show()


object_cols


# applying Label Encoder 
ln = LabelEncoder()

for col in object_cols:
    train_d[col] = ln.fit_transform(train_d[col])
    test_d[col] = ln.transform(test_d[col])


std = StandardScaler()

# numerical columns
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
             'loan_amount', 'interest_rate']

train_d[num_cols] = std.fit_transform(train_d[num_cols])
test_d[num_cols] = std.transform(test_d[num_cols])
    


x = train_d.drop('loan_paid_back',axis = 1)
y = train_d['loan_paid_back']
test = test_d.copy()


params = {
    'n_estimators' : 10000,
    'n_jobs' :-1,
    'eval_metric': 'auc',
    'max_depth': 5,
    'learning_rate':0.01,
    'enable_categorical' : True,
    'subsample':0.85,
    'colsample_bytree' : 0.8,
    'lambda': 2.0,
    'alpha':1.0,
    'tree_method':'hist',
    'objective' : 'binary:logistic',
    'device' : 'cuda',
    'min_child_weight' : 3,
    'gamma' : 0.05
    
}


seeds = [42,128,256]

oof_preds_full = []
xgb_preds_full = []

for seed in seeds:
    print(f"The Current Seed Is {seed}\n")


    oof_preds = np.zeros(x.shape[0])
    xgb_preds = np.zeros(test.shape[0])

    n_splits = 5

    kf = StratifiedKFold(n_splits = n_splits,random_state = seed,shuffle = True)

    for fold,(train_idx,val_idx) in enumerate(kf.split(x,y)):
        print(f"========{fold+1}/{n_splits}======")
        x_train,y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val ,y_val = x.iloc[val_idx], y.iloc[val_idx]
        #early_stop = EarlyStopping(rounds = 300,metric_name = 'auc',data_name = 'validation_0')
        
        local_params = params.copy()
        local_params['seed'] = seed


        model = XGBClassifier(**local_params,early_stopping_rounds = 300)

        model.fit(x_train,
                  y_train,
                  eval_set = [(x_val,y_val)],
                  verbose = 500
        )
        oof_preds[val_idx] = model.predict_proba(x_val)[:,1]
        test_preds = model.predict_proba(test)[:,1]
        xgb_preds += test_preds/n_splits

        auc_score = roc_auc_score(y_val,oof_preds[val_idx])
        print(f"CV ROC AUC: {auc_score:.4f}\n")

    oof_preds_full.append(oof_preds)
    xgb_preds_full.append(xgb_preds)

ovr_mean = np.mean(oof_preds_full,axis = 0)
overall = roc_auc_score(y,ovr_mean)
print("="*8)
print(f"Overall OOF AUC {overall:.4f}")
print("="*8)


cols = [f"Seed {seed}" for seed in seeds]
oof_preds_full_array = np.array(oof_preds_full)
xgb_preds_full_array = np.array(xgb_preds_full)

oof_preds_full_df = pd.DataFrame(oof_preds_full_array.T,columns = cols,index = x.index)
xgb_preds_full_df = pd.DataFrame(xgb_preds_full_array.T,columns = cols,index = test.index)



meta_model = LogisticRegressionCV(
    Cs = 10,
    cv = 5,
    scoring = 'roc_auc',
    max_iter = 1000
)
meta_model.fit(oof_preds_full_df,y)


# Prediction 
final_preds = meta_model.predict_proba(xgb_preds_full_df)[:,1]


#Submission 
submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
submission['loan_paid_back'] = final_preds



submission.head()


submission.to_csv('submission.csv',index = False)
print("Submission Of Dataset Is Done")




