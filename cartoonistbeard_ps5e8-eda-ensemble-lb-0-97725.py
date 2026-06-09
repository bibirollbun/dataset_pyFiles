import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno #type: ignore

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class config:
    V = 1
    train_path = "/kaggle/input/playground-series-s5e8/train.csv"
    test_path = "/kaggle/input/playground-series-s5e8/train.csv"
    sub_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"
    ensemble_dir = "/kaggle/input/ps5e8-experiments"
    target = 'y'
    seed = 42
    test_size=0.2
cfg = config()  


class DataIngestion:
    def __init__(self):
        self.train = pd.read_csv(cfg.train_path,index_col='id')
        self.test = pd.read_csv(cfg.test_path,index_col='id')
    def fetchData(self):
        print(f"Shape of Train: {self.train.shape}")
        display(self.train.head())
        print(f"Shape of Test: {self.test.shape}")
        display(self.test.head())
        print("#"*5,"Missing Values in Train",'#'*5)
        msno.matrix(self.train)
        print("#"*5,"Missing Values in Test",'#'*5)
        msno.matrix(self.test)
        return self.train, self.test
    
Ingester = DataIngestion()
train,test = Ingester.fetchData()


class UnivariateAnalysis:
    def __init__(self,df):
        self.data = df.copy()
        self.data = self.data[:10000]
        self.num_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64','float64'] and col != cfg.target]
        self.cat_cols = [col for col in self.data.columns if self.data[col].dtype in ['O'] and col != cfg.target]
        print(f"There are {len(self.num_cols)} numerical columns and {len(self.cat_cols)} categorical columns")
    def barplots(self):
        plt.figure(figsize=(18,10))
        for i,col in enumerate(self.num_cols,1):
            plt.subplot(2,4,i)
            sns.histplot(data=self.data,x=col,hue=cfg.target,kde=True,bins=3,common_norm=False,element='step',stat='density')
            plt.title(f"Distribution of {col}",pad=10,weight='bold')
        plt.tight_layout()
        plt.show()
    def boxplots(self):
        plt.figure(figsize=(18,10))
        for i,col in enumerate(self.num_cols,1):
            plt.subplot(2,4,i)
            sns.boxplot(data=self.data,y=col,x=cfg.target)
            plt.title(f"{col} by Target",pad=10,weight='bold')
        plt.tight_layout()
        plt.show()
    def countplots(self):
        plt.figure(figsize=(18,10))
        for i,col in enumerate(self.cat_cols,1):
            plt.subplot(2,5,i)
            sns.countplot(data=self.data,x=col,hue=cfg.target)
            plt.title(f"Frequency of {col}")
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    def targetDistribution(self):
        plt.figure(figsize=(18,10))
        plt.subplot(1,2,1)
        ax = sns.countplot(x=cfg.target,data=self.data)
        plt.title("Target Distribution",pad=15)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2., p.get_height()), #type: ignore
            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.subplot(1,2,2)
        self.data[cfg.target].value_counts().plot(kind='pie',autopct='%1.1f%%',explode=[0.05,0],startangle=90)
        plt.title("Target Propotion",pad=15)
        plt.tight_layout()
        plt.show()
    def UAautomation(self):
        print("#"*5,"Analysis of Numerical Columns",'#'*5)
        self.barplots()
        self.boxplots()
        print("#"*5,"Analysis of categorical Columns",'#'*5)
        self.countplots()
        self.targetDistribution()

UA = UnivariateAnalysis(train)
UA.UAautomation()


class BivariateAnalysis:
    def __init__(self,df):
        self.data = df.copy()
        self.num_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64','float64'] and col != cfg.target]
        self.cat_cols = [col for col in self.data.columns if self.data[col].dtype in ['O'] and col != cfg.target]
    def correlationPlot(self):
        sample_data = self.data[:1000]
        for col in self.cat_cols:
            sample_data[col],_ =pd.factorize(sample_data[col])
        corr_mat = sample_data.corr()
        mask = np.triu(np.ones_like(corr_mat,dtype=bool))
        plt.figure(figsize=(18,10))
        sns.heatmap(corr_mat,mask=mask,fmt='.2f',cmap='winter',annot=True)
        plt.title("Feature Correlation Matrix",pad=10)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.show()
    def featureInteraction(self):
        sample_df = self.data[:1000]
        plt.figure(figsize=(18,10))
        plt.subplot(2,2,1)
        sns.violinplot(x='education',y='age',data=sample_df,inner='quartile')
        plt.title("Age Distribution by Education",pad=15)
        plt.subplot(2,2,2)
        sns.scatterplot(data=sample_df,x='job',y='balance',hue=cfg.target,alpha=0.7)
        plt.title("Distribution of balance by jobs",pad=10)
        plt.xticks(rotation=90)
        plt.subplot(2,2,3)
        sns.boxplot(data=sample_df,x='marital',y='duration',hue=cfg.target)
        plt.title("Distribution of duration by marital status")
        plt.subplot(2,2,4)
        sns.kdeplot(data=sample_df,x='day',hue=cfg.target,fill=True,common_norm=False)
        plt.title("Day Distribution",pad=10)
    def feature_importance(self):
        sample_data = self.data[:1000]
        for col in self.cat_cols:
            sample_data[col],_ =pd.factorize(sample_data[col])
        X = sample_data.copy()
        y = X.pop(cfg.target)
        X_train,X_valid,Y_train,Y_valid = train_test_split(X,y,test_size=cfg.test_size,random_state=cfg.seed)
        rf = RandomForestClassifier(n_estimators=100,random_state=cfg.seed)
        rf.fit(X_train,Y_train)
        feature_imp = pd.Series(rf.feature_importances_,index=X.columns).sort_values()
        plt.figure(figsize=(10,6))
        sns.barplot(x=feature_imp,y=feature_imp.index)
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.show()
    def automation(self):
        self.correlationPlot()
        self.featureInteraction()
        self.feature_importance()

BA = BivariateAnalysis(train)
BA.automation()


# from hillclimbers import climb_hill, partial
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import os
import glob
import pickle

import warnings
warnings.filterwarnings("ignore")


def create_ensemble():
    data = {}

    print(f"Looking for .npy files in: {cfg.ensemble_dir}\n")

    # Load all .npy files
    for filename in os.listdir(cfg.ensemble_dir):
        if filename.endswith(".npy"):
            path = os.path.join(cfg.ensemble_dir, filename)
            arr = np.load(path)
            data[filename] = arr
            print(f"Loaded: {filename} | Shape: {arr.shape}")

    if not data:
        print("⚠ No .npy files found in the directory.")
        return pd.DataFrame(), pd.DataFrame()

    # Helper: get correct column
    def get_scores(arr):
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            return arr[:, 1]
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

    # Build OOF and prediction data
    oof_data = {
        "_".join(k.split("_")[:-1]): get_scores(v)
        for k, v in data.items() if "oof" in k.lower()
    }
    preds_data = {
        "_".join(k.split("_")[:-1]): get_scores(v)
        for k, v in data.items() if "preds" in k.lower()
    }
    print(f"\nOOF keys found: {list(oof_data.keys())}")
    print(f"Preds keys found: {list(preds_data.keys())}")
    oof_data = pd.DataFrame(oof_data)
    preds_data = pd.DataFrame(preds_data)
    cols = sorted(oof_data.columns.to_list())
    oof_data = oof_data[cols]
    preds_data = preds_data[cols]
    return oof_data,preds_data 


# Usage
oof, preds = create_ensemble()
display(oof.head())
display(preds.head())


# hc_test, hc_oof = climb_hill(train=train, target=config.target, objective='maximize', 
#                              eval_metric=partial(roc_auc_score),oof_pred_df= oof, 
#                              test_pred_df= preds,plot_hill=True,plot_hist=False, 
#                              precision=0.001,negative_weights=True,return_oof_preds=True)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def kfold_meta_learner(oof_df, preds_df, target, n_splits=5, random_state=42):
    """
    Train a meta learner using K-Fold CV on OOF predictions.
    
    Parameters:
        oof_df: DataFrame of OOF predictions from base models
        preds_df: DataFrame of test predictions from base models
        target: array-like, true labels for the training data
        n_splits: number of folds
        random_state: random seed for reproducibility
    Returns:
        meta_oof: array of OOF predictions from meta learner
        meta_preds: averaged predictions on test set
    """
    meta_oof = np.zeros(len(oof_df))
    meta_preds = np.zeros(len(preds_df))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(oof_df)):
        X_train, X_val = oof_df.iloc[train_idx], oof_df.iloc[val_idx]
        y_train, y_val = target[train_idx], target[val_idx]
        
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_train, y_train)
        
        meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]
        meta_preds += meta_model.predict_proba(preds_df)[:, 1] / n_splits
        
        print(f"Fold {fold+1} done")

    return meta_oof, meta_preds

train = pd.read_csv(cfg.train_path,index_col='id')
meta_oof, meta_preds = kfold_meta_learner(oof, preds, train[cfg.target])

print("Meta-learner training complete.")


sub1 = pd.read_csv(cfg.sub_path)
sub1[cfg.target] = meta_preds
display(sub1.head())


sub2 = pd.read_csv('/kaggle/input/no-blending-bank-classification-xgb-lgbm-cat-ydf/submission.csv')

rsub1 = sub1[cfg.target].rank(method='average')/(len(sub1)+1)
rsub2 = sub2[cfg.target].rank(method='average')/(len(sub2)+1)


best_w = None
best_score = -1

for w in np.linspace(0,1,21):
    blend = w*rsub1 + (1-w)*rsub2
    score = np.var(blend)
    if score>best_score:
        best_score = score
        best_w = w

best_w = 0.3
sub1[cfg.target] = best_w*rsub1 + (1-best_w)*rsub2
sub1.to_csv(f'submission.csv',index=False)


display(sub1.head())

