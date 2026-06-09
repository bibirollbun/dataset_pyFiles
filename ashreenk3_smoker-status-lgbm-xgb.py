import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import os
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import Pool, CatBoostClassifier, cv


train_data = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv",index_col="id")
test_data = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv",index_col="id")


train_data.head()


test_data.head()


train_data.describe()


mask = np.triu(np.ones_like(train_data.corr()))
plt.figure(figsize=(20,12))
sns.heatmap(train_data.corr(), cmap="BuGn", annot=True, mask=mask,vmin=-1,vmax=1);


seed = np.random.seed(6)

X = train_data.drop(["smoking"],axis=1)
y = train_data["smoking"]


lgbmmodel = LGBMClassifier(random_state=seed, device="gpu")
print("CV score of LGBM is ",cross_val_score(lgbmmodel,X,y,cv=4, scoring = 'roc_auc').mean())


xgbmodel = XGBClassifier(random_state=seed, tree_method= 'gpu_hist')
print("CV score of XGB is ",cross_val_score(xgbmodel,X,y,cv=4, scoring = 'roc_auc').mean())


%%capture
lgbmmodel.fit(X,y)
xgbmodel.fit(X,y)


def plotImportance(modelNames,models):
    plt.subplots(len(modelNames),1,figsize=(14,5*len(modelNames)),dpi=300)
    for ind,modelName in enumerate(modelNames):
        history = pd.DataFrame()
        history["cols"] = test_data.columns
        if modelNames[ind] == "CatBoost":
            history["imp"] = models[ind].get_feature_importance()
        else:  
            history["imp"] = models[ind].feature_importances_
        history.sort_values("imp",inplace=True,ascending=False)
        history.reset_index(drop=True)
        plt.subplot(len(modelNames),1,ind+1)
        sns.barplot(x=history["imp"],y=history["cols"],palette="rocket");
        plt.title("Feature Imporance of "+modelName)



plotImportance(["LGBM","XGB"],[lgbmmodel,xgbmodel])


submission = pd.DataFrame()
submission["id"] = test_data.index


submission["smoking"] = lgbmmodel.predict_proba(test_data)[:,1] + xgbmodel.predict_proba(test_data)[:,1]
submission["smoking"] = submission["smoking"]/2
submission.head(10)


submission[["id","smoking"]].to_csv("submission.csv",header=True,index=False)

