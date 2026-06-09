import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import optuna
import shutil
import glob
import json
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.base import clone
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
# from hillclimbers import climb_hill, partial
from tqdm import tqdm 
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


class config:
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"
    sub_path = "/kaggle/input/playground-series-s5e6/sample_submission.csv"
    org_path = "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
    target = "Fertilizer Name"
    oof_path = "/kaggle/input/xgboost-repeatedstratifiedkfold/oof.npy"
    oof_2 = "/kaggle/input/ps5e6-dataset/oof1.npy"
    stack_oof = "/kaggle/input/ps5e6-dataset/stacking_oof.npy"
    xgb_repeat_train_oof = "/kaggle/input/ps5e6-dataset/xgb_repeat_train_oof.npy"
    xgb_repeat_test_oof = "/kaggle/input/ps5e6-dataset/xgb_repeat_test_oof.npy"
    stack_pred = "/kaggle/input/ps5e6-dataset/stacking_test.npy"
    pred_path = "/kaggle/input/xgboost-repeatedstratifiedkfold/pred.npy"
    pred_2 = "/kaggle/input/ps5e6-dataset/preds1.npy"
    oof = "/kaggle/input/ps5e6-dataset/oof.npy"
    pred = "/kaggle/input/ps5e6-dataset/pred.npy"
    random_state = 42
    folds = 5
    V = 5
    augment = 6
    
cfg = config()


class DataIngestion:
    def __init__(self):
        self.train = pd.read_csv(cfg.train_path,index_col="id")
        self.test = pd.read_csv(cfg.test_path,index_col="id")
        self.org = pd.read_csv(cfg.org_path)
        display(self.train.head())
        display(self.test.head())
        display(self.org.head())
    def get_data(self):
        self.org.drop
        return self.train, self.test, self.org

class DataEDA:
    def __init__(self,train,test,org):
        self.train = train
        self.test = test
        self.org = org
        self.cols = [col for col in self.train if col != cfg.target]
    def nan_track(self):
        self.train.isna().sum().plot(kind="bar")
        plt.title("Train")
        plt.show()
        self.test.isna().sum().plot(kind="bar")
        plt.title("Test")
        plt.show()
        self.org.isna().sum().plot(kind="bar")
        plt.title("Test")
        plt.show()
    def val_cnt(self):
        for col in self.cols:
            if self.train[col].dtype == "O":
                self.train[col].value_counts().plot(kind="bar")
                plt.title(f"train_{col}")
                plt.show()
                self.test[col].value_counts().plot(kind="bar")
                plt.title(f"test_{col}")
                plt.show()
    def feature_range(self):
        for col in self.cols:
            if self.train[col].dtype in ["float64", "int64"]:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=self.train, x=col, bins=30, kde=True)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.show()
    def initiate_eda(self):
        self.nan_track()
        self.val_cnt()
        self.feature_range()
                
di = DataIngestion()
train,test,org = di.get_data()
# eda = DataEDA(train,test,org)
# eda.initiate_eda()


class Preprocessor:
    def __init__(self,train,test,org):
        self.train = train.copy()
        self.test = test.copy()
        self.org = org
        self.obj_col = [col for col in self.train.columns if self.train[col].dtype == "O" and col != cfg.target]
        
    def label_encode(self):
        for col in self.obj_col:
            encoder = LabelEncoder()
            self.train[col] = encoder.fit_transform(self.train[col])
            self.test[col] = encoder.transform(self.test[col])
            self.org[col] = encoder.transform(self.org[col])
            
    def category_encode(self):
        for col in self.obj_col:
            self.train[col] = self.train[col].astype("category")
            self.test[col] = self.test[col].astype("category")
            self.org[col] = self.org[col].astype("category")
            
    def encode_target(self):
        self.train[cfg.target].value_counts().plot(kind="bar")
        plt.title("Train Target Distribution")
        plt.show()
        self.org[cfg.target].value_counts().plot(kind="bar")
        plt.title("Original Target Distribution")
        plt.show()
        print(f"Shape before dropping : {self.train.shape} and {self.org.shape}")
        self.train.dropna(inplace=True)
        self.org.dropna(inplace=True)
        print(f"Shape after dropping : {self.train.shape} and {self.org.shape}")
        
    def initiate_preprocessing(self):
        self.category_encode()
        self.encode_target()
        display(self.train.head())
        display(self.test.head())
        display(self.org.head())
        return self.train, self.test, self.org

proc = Preprocessor(train,test,org)
train_p, test_p, org_p = proc.initiate_preprocessing()


def map3(y_true, y_pred_probs):
    y_true = [[x] for x in y_true]
    y_pred_probs = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1].tolist()
    
    def ap3(y_true, y_pred_probs):
        y_pred_probs = y_pred_probs[:3]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(y_pred_probs):
            if p in y_true and p not in y_pred_probs[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not y_true:
            return 0.0

        return score
    
    return np.mean([ap3(a,p) for a,p in zip(y_true, y_pred_probs)])


class ModelTrainer:
    
    def __init__(self,train,test,org):
        self.train = train.copy()
        self.test = test.copy()
        self.org = org.copy()
        self.X = self.train.drop(columns=[cfg.target])
        self.Y = self.train[cfg.target]
        self.X_o = self.org.drop(columns=[cfg.target]) 
        self.Y_o = self.org[cfg.target]
        
    def train_model(self,model_name,model_params):
        skf = StratifiedKFold(n_splits=cfg.folds,shuffle=True,random_state=cfg.random_state)
        preds = np.zeros((len(self.test),7))
        oof = np.zeros((len(self.X),7))

        for fold, (train_indx,test_indx) in tqdm(enumerate(skf.split(self.X,self.Y), 1), total=5, desc="Model Training"):
        # for fold,(train_indx,test_indx) in enumerate(skf.split(self.X,self.Y)):
        
            print("#"*25,f"FOLD : {fold}", "#"*25)
            X_train, X_test = self.X.iloc[train_indx],self.X.iloc[test_indx]
            y_train, y_test = self.Y[train_indx],self.Y[test_indx]

            X_train = pd.concat([X_train] + [self.X_o] * cfg.augment)
            y_train = pd.concat([y_train] + [self.Y_o] * cfg.augment)
            
            X_train, y_train = shuffle(X_train, y_train, random_state=cfg.random_state)
            dtrain = xgb.DMatrix(X_train,label=y_train,enable_categorical=True)
            dval   = xgb.DMatrix(X_test,label=y_test,enable_categorical=True)
            dtest  = xgb.DMatrix(self.test,enable_categorical=True)
            if model_name == "XGB":
                # model = XGBClassifier(**model_params)
                # model.fit(
                #     X_train,y_train,
                #     eval_set = [(X_test,y_test)],
                #     verbose = 300
                # )
                model = xgb.train(
                    model_params,
                    dtrain,
                    num_boost_round = 10000,
                    evals = [(dtrain,"train"),(dval,"validation")],
                    early_stopping_rounds = 100,
                    verbose_eval = 200
                )
                # oof[test_indx] = model.predict_proba(X_test)
                # preds += model.predict_proba(self.test)
                oof[test_indx] = model.predict(dval)
                preds += model.predict(dtest)
            print(f"Fold MAP Score = {map3(y_test,oof[test_indx])}")
        print(f"Overall MAP Score = {map3(self.Y,oof)}")
        return oof,preds/cfg.folds


xgb_params = {
    "device": "gpu",
    "max_depth": 12,
    "num_class":7,
    "colsample_bytree": 0.467,
    "subsample": 0.86,
    "n_estimators": 10000,
    "learning_rate": 0.03,
    "gamma": 0.26,
    "max_delta_step": 4,
    "reg_alpha": 2.7,
    "reg_lambda": 1.4,
    "early_stopping_rounds": 100,
    "objective": 'multi:softprob',
    "random_state": 13,
    "enable_categorical": True,
}


target_encoder = LabelEncoder()
train_p[cfg.target] = target_encoder.fit_transform(train_p[cfg.target])
org_p[cfg.target] = target_encoder.transform(org_p[cfg.target])

display(train_p.head())
display(org_p.head())


%%time

mt = ModelTrainer(train_p,test_p,org_p)
oof,preds = mt.train_model("XGB",xgb_params)


np.save(f"oof{cfg.V}.npy",oof)
np.save(f"preds{cfg.V}.npy",preds)


class Ensemble:
    
    def __init__(self):
        self.oof = oof
        self.preds = preds
        self.final_test = np.zeros((test.shape[0], 7))  # initialize
        self.meta_model = LGBMClassifier(
            objective='multiclass',
            num_class=7,
            learning_rate=0.01,
            n_estimators=10000,
            random_state=42,
            verbose=-1,
            device='gpu',
        )

    def fetchExternalOof(self):
        self.oof_e = np.load(cfg.oof_path)[:, :7]
        self.oof_e2 = np.load(cfg.oof_2)[:,:7]
        self.pred_e = np.load(cfg.pred_path)[:, :7]
        self.pred_e2 = np.load(cfg.pred_2)[:,:7]
        self.oof_stack = np.load(cfg.stack_oof)[:,:7]
        self.pred_stack = np.load(cfg.stack_pred)[:,:7]
        self.oof = np.load(cfg.oof)[:,:7]
        self.pred = np.load(cfg.pred)[:,:7]
        self.xgb_repeat_test_oof = np.load(cfg.xgb_repeat_test_oof)[:,:7]
        self.xgb_repeat_train_oof = np.load(cfg.xgb_repeat_train_oof)[:,:7]

    def createEnsembleDf(self):
        # assert self.oof.shape[0] == self.oof_e.shape[0], "Mismatch in oof samples"
        # assert self.preds.shape[0] == self.pred_e.shape[0], "Mismatch in pred samples"
        self.X = np.hstack([self.oof_e, self.oof_e2,self.oof_stack,self.oof,self.xgb_repeat_train_oof,self.oof])
        self.X_test = np.hstack([self.pred_e, self.pred_e2,self.pred_stack,self.pred,self.xgb_repeat_test_oof,self.preds])
        return self.X, self.X_test

    def fitMetaModel(self, y):
        """Fits meta model for 7-class classification using StratifiedKFold."""
        self.model = []
        self.ensemble_oof = np.zeros((self.X.shape[0], 7))
        self.ensemble_preds = np.zeros((self.X_test.shape[0], 7))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, valid_idx) in tqdm(enumerate(skf.split(self.X, y), 1), total=5, desc="Meta-model Training"):
            print("\n", "#" * 25, f"Fold :{fold}", "#" * 25)
            x_train, x_valid = self.X[train_idx], self.X[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            self.meta_model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )
            
            self.ensemble_oof[valid_idx] = self.meta_model.predict_proba(x_valid)
            self.final_test += self.meta_model.predict_proba(self.X_test)
            
            print(f"Fold MAP Score = {map3(y_valid, self.ensemble_oof[valid_idx])}")
        
        print(f"\nOverall MAP Score = {map3(y, self.ensemble_oof)}")

    def initiateEnsemble(self, y_true):
        print("#" * 25, "Fetching External Data", "#" * 25)
        self.fetchExternalOof()
        print("#" * 25, "Concatenating the external and internal files", "#" * 25)
        self.createEnsembleDf()
        print("#" * 25, "Fitting Meta Model", "#" * 25)
        self.fitMetaModel(y_true)
        return self.final_test / 5


# Usage
ensemble = Ensemble()
preds = ensemble.initiateEnsemble(train_p[cfg.target])


# np.argsort(preds)[:, -3:][:, ::-1]
target_encoder.inverse_transform([0])[0]


final_predictions = []
for i in np.argsort(preds)[:, -3:][:, ::-1]:
    prediction = target_encoder.inverse_transform(i)
    final_predictions.append(" ".join(prediction))

sub_df = pd.read_csv(cfg.sub_path)
sub_df[cfg.target] = final_predictions
display(sub_df.head())


sub_df.to_csv(f"Ensemble{cfg.V}.csv",index=False)

