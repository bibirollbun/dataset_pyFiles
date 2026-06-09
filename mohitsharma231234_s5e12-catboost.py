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
from sklearn.model_selection import StratifiedKFold,KFold
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder,StandardScaler
from catboost import CatBoostClassifier as cgb
from sklearn.metrics import roc_auc_score

import optuna
import warnings
warnings.filterwarnings("ignore")


train_d = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


test_d.head()


train_d.isnull().sum()


train_d.describe().style.background_gradient(cmap="rainbow")


train_d.shape


test_d.shape


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


train_d = train_d.drop('id',axis = 1)
test_d = test_d.drop('id',axis = 1)


train_d.info()


# Applying target encoding 
def target_encoding(train, predict, n_splits=8):
    train = train.copy()
    predict = predict.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = {}
    mean_features_test = {}

    # Compute global mean once
    target_global = train[target].mean()

    for col in cols:

        # K-FOLD TARGET ENCODING
        oof = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]

            # Mean for each category in this fold
            fold_map = tr_fold.groupby(col)[target].mean()

            # Map on validation fold
            oof[val_idx] = train[col].iloc[val_idx].map(fold_map).fillna(target_global)

        mean_features_train[f"mean_{col}"] = oof

        # Apply encoding to the prediction set
        global_map = train.groupby(col)[target].mean()
        mean_features_test[f"mean_{col}"] = (
            predict[col].map(global_map).fillna(target_global)
        )


    train = pd.concat([train, pd.DataFrame(mean_features_train)], axis=1)
    predict = pd.concat([predict, pd.DataFrame(mean_features_test)], axis=1)

    return train, predict


target = 'diagnosed_diabetes'
cols = (
    train_d.drop(columns= ['id',target],errors = 'ignore')
        .columns
        .tolist()
)

cat = [c for c in cols if train_d[c].dtype in ("object","category")]

train_d,test_d = target_encoding(train_d,test_d,n_splits = 10)



# Handelling Outliers  using log
binning_cols = ['physical_activity_minutes_per_week','diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi','waist_to_hip_ratio','systolic_bp',
       'diastolic_bp','heart_rate','cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides']

for col in binning_cols:
    train_d[col + '_log'] = np.log1p(train_d[col])
    test_d[col + '_log'] = np.log1p(test_d[col])


# Now Applying Binning 
def apply_binning(train,test,cols,n_bins = 10):
    train[cols] = train[cols].replace([np.inf,-np.inf],np.nan)
    test[cols] = test[cols].replace([np.inf,-np.inf],np.nan)

    bins = pd.qcut(
        train[cols],
        q = n_bins,
        retbins = True,
        duplicates = 'drop'
    )[1]

    train[cols + '_bin'] = pd.cut(
        train[cols],
        bins = bins,
        labels = False,
        include_lowest = True
    )


    test[cols + '_bin'] = pd.cut(
        test[cols],
        bins = bins,
        labels = False,
        include_lowest = True
    )

    return train,test
    


for col in binning_cols:
    train_d, test_d = apply_binning(train_d, test_d, col + '_log', n_bins=10)



cat_features = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status']


x,y  = train_d.drop('diagnosed_diabetes',axis = 1),train_d['diagnosed_diabetes']


cat_params = {
    'iterations' : 1017,
    'learning_rate': 0.06904636051297385,
    'depth': 6,
    'l2_leaf_reg': 5.91801438105106,
    'random_strength': 0.7511110930048408,
    'bagging_temperature': 0.46442433080118917,
    'loss_function' : 'Logloss',
    'eval_metric' : 'AUC',
    'verbose' : 0,
    'random_seed' : 42,
    'allow_writing_files' : False,
    'task_type' : 'GPU',
    'devices' :'0',
    'bootstrap_type': 'Bayesian'

    
}


# Applying CV 

fold_auc,roc_curves = [],[]
cat_oof = np.zeros(x.shape[0])
cat_preds = np.zeros(test_d.shape[0])

n_splits = 5
sk = StratifiedKFold(n_splits =n_splits,shuffle = True,random_state = 42)

for fold,(train_idx,val_idx) in enumerate(sk.split(x,y)):
    x_train,y_train = x.iloc[train_idx] ,y.iloc[train_idx]
    x_val,y_val = x.iloc[val_idx],y.iloc[val_idx]

    model = cgb(**cat_params,early_stopping_rounds = 300)

    model.fit(x_train,
             y_train,
             eval_set = [(x_val,y_val)],
             cat_features = cat_features,
             verbose = 500)

    val_pred = model.predict_proba(x_val)[:,1]
    cat_oof[val_idx] = val_pred
    cat_preds += model.predict_proba(test_d)[:,1]/n_splits

    auc = roc_auc_score(y_val,val_pred)
    fold_auc.append(auc)
    print(f"Fold : {fold} AUC : {auc:.4f}")

overall_auc = roc_auc_score(y,cat_oof)
print("Fold AUCs:", [round(s,4) for s in fold_auc])
print(f"Overall OOF AUC: {overall_auc:.5f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
submission['diagnosed_diabetes'] = cat_preds


submission.to_csv("submission.csv",index = False)
print("Submission File Is Submitted Successfully")


submission.head()


np.save("cat_oof.npy", cat_oof)
np.save("cat_preds.npy", cat_preds)



train_d.head()


test_d.isnull().sum()




