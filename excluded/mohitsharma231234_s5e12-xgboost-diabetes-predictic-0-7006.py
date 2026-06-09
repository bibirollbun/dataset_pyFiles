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


# Importing All Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder,StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


train_d = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
org_df = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")


test_d.head()


train_d.isnull().sum()


train_d.describe().style.set_properties(**{
    "background-color": '#f2f2f2',
    'color':"black"
})


train_d.shape


test_d.shape


# 1ï¸�âƒ£ Column list from train
columns = train_d.columns.tolist()

# 2ï¸�âƒ£ Take only those columns from org_df
new_org = org_df.reindex(columns=columns)



train_d = pd.concat(
    [train_d, new_org],
    ignore_index=True
)


train_d.duplicated().sum()


train_d.shape


print("#"*20,"List Of Categorical Columns","#"*20,"\n")
cat_cols = [c for c in train_d.columns if train_d[c].dtype == "object"]

for i, cat in enumerate(cat_cols):
    print(i,cat)
print("\n")
print("#"*20,"List Of Numerical Columns","#"*20,"\n")
num_cols = [c for c in train_d.columns if train_d[c].dtype not in ("object","category")]

for i,num in enumerate(num_cols):
    print(i,num)



# Unique Values 
print("Printing Unique Values Of All Columns")
for col in train_d:
    print("-"*60)
    unq = train_d[col].unique()
    if len(unq) < 15:
        print(f"{col} : {unq}\n")
    else:
        print(f"{col} : {unq[:5]} .... {col} Have So much Unique Values \n")


# Target Columns distribution
count = train_d.diagnosed_diabetes.value_counts()

# Ploting Plot For Target Variable
plt.figure(figsize= (8,6))
plt.bar(count.index.astype(str),count,color = ['#FF6F61','#955251'])
plt.title("Distribution of Target Columns")
plt.xlabel("Diabetes 1.0 = YES   0.0 = NO")
plt.ylabel("Count")
plt.show()


sns.set_style("darkgrid")
plt.figure(figsize = (14,len(num_cols)*3))

for idx,col in enumerate(num_cols,1):
    plt.subplot(len(num_cols),2,2*idx-1)
    sns.histplot(train_d[col],kde= True,bins = 20,color = 'orange',edgecolor = 'black')
    plt.title(f"Distribution of {col}",fontweight = 'bold',fontsize = 12)

    plt.subplot(len(num_cols),2,2*idx)
    sns.boxplot(x=  train_d[col],color = 'lightgreen')
    plt.title(f"Boxplot of {col}",fontweight = 'bold',fontsize = 12)

plt.tight_layout()
plt.show()


# Plot for Categorical Columns
plt.figure(figsize = (14,len(cat_cols)*3))
for idx,col in enumerate(cat_cols,1):
    plt.subplot(len(cat_cols),2,idx)
    sns.countplot(x = col,data = train_d,palette = 'husl')
    plt.title(f"Countplot of {col}",fontweight = 'bold',fontsize = 12)
    plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()


# Handelling missig values 
cols = ['physical_activity_minutes_per_week','diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi','waist_to_hip_ratio','systolic_bp',
       'diastolic_bp','heart_rate','cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides']

for col in cols:
    q1 = train_d[col].quantile(0.25)
    q2 = train_d[col].quantile(0.75)
    iqr = q2-q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q2 + 1.5 *iqr
    train_d[col] = train_d[col].clip(lower = lower_bound,upper = upper_bound)
    test_d[col] = test_d[col].clip(lower = lower_bound,upper = upper_bound)


train_d = train_d.drop('id',axis = 1)
test_d = test_d.drop('id',axis = 1)


# Applying Label Encoder
le = LabelEncoder()
for col in cat_cols:
    train_d[col]= le.fit_transform(train_d[col])
    test_d[col] = le.transform(test_d[col])


x,y  = train_d.drop('diagnosed_diabetes',axis = 1),train_d['diagnosed_diabetes']


# xgboost Parameters
params = {
    'n_estimators' : 10000,
    'n_jobs' :-1,
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


# Applying CV 

fold_auc,roc_curves = [],[]
xgb_oof = np.zeros(x.shape[0])
xgb_preds = np.zeros(test_d.shape[0])

n_splits = 5
sk = StratifiedKFold(n_splits =n_splits,shuffle = True,random_state = 42)

for fold,(train_idx,val_idx) in enumerate(sk.split(x,y)):
    x_train,y_train = x.iloc[train_idx] ,y.iloc[train_idx]
    x_val,y_val = x.iloc[val_idx],y.iloc[val_idx]

    model = XGBClassifier(**params,early_stopping_rounds = 300)

    model.fit(x_train,
             y_train,
             eval_set = [(x_val,y_val)],
             eval_metric = 'auc',
             verbose = 500)

    val_pred = model.predict_proba(x_val)[:,1]
    xgb_oof[val_idx] = val_pred
    xgb_preds += model.predict_proba(test_d)[:,1]/n_splits

    auc = roc_auc_score(y_val,val_pred)
    fold_auc.append(auc)
    print(f"Fold : {fold} AUC : {auc:.4f}")

overall_auc = roc_auc_score(y,xgb_oof)
print("Fold AUCs:", [round(s,4) for s in fold_auc])
print(f"Overall OOF AUC: {overall_auc:.5f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
submission['diagnosed_diabetes'] = xgb_preds


submission.to_csv("submission.csv",index = False)
print("Submission File Is Submitted Successfully")


submission.head()


np.save("xgb_oof.npy", xgb_oof)
np.save("xgb_preds.npy", xgb_preds)





