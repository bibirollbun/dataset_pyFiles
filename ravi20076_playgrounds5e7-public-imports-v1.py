


%%writefile req_kaggle.txt

scikit-learn==1.7.0
xgboost==3.0.2
lightgbm==4.6.0
numpy==1.26.4
scipy==1.14.1
polars==1.31.0
pytorch_tabnet
tabpfn


%%writefile -a myimports.py

print(f"\n---> Commencing imports-part1")

from gc import collect
from warnings import filterwarnings
filterwarnings('ignore')
from IPython.display import display_html, clear_output
clear_output()
import os, sys, logging, re, joblib, ctypes, shutil, random, torch
from copy import deepcopy

import xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk, pandas as pd, polars as pl
collect()

from warnings import filterwarnings
filterwarnings('ignore')
from gc import collect

from os import path, walk, getpid
from psutil import Process
import re
from collections import Counter
from itertools import product, combinations

import ctypes
libc = ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import pandas as pd, numpy as np
from scipy.stats import pearsonr
import polars as pl
import polars.selectors as cs

from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore, Style, init
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm.notebook import tqdm

print(f"---> Imports- part 1 done")
print(f"---> Sklearn = {sk.__version__} | Pandas = {pd.__version__} | Polars = {pl.__version__}")


%%writefile -a myimports.py

# Pipeline specifics:-
from sklearn.preprocessing import *

from sklearn.impute import SimpleImputer
from sklearn.model_selection import *
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector

# ML Model training:-
from sklearn.metrics import *

from xgboost import QuantileDMatrix, XGBClassifier as XGBC, XGBRegressor as XGBR
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC, LGBMRegressor as LGBMR
from catboost import CatBoostClassifier as CBC, Pool, CatBoostRegressor as CBR
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, RandomForestClassifier as RFC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, RandomForestRegressor as RFR
from sklearn.ensemble import VotingRegressor as VR, VotingClassifier as VC
from sklearn.linear_model import LogisticRegression as LRC, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier as KNNC, KNeighborsRegressor as KNNR

# TabNet models
from pytorch_tabnet.tab_model import (TabNetRegressor as TNR, TabNetClassifier as TNC)

# TabPFN models
from tabpfn import TabPFNClassifier as TPFNC

# Ensemble and tuning:-
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

# Setting rc parameters in seaborn for plots and graphs-
sns.set({"axes.facecolor"       : "white",
         "figure.facecolor"     : "#ffffff",
         "axes.edgecolor"       : "black",
         "grid.color"           : '#b0b0b0',
         "font.family"          : ['Cambria'],
         "axes.labelcolor"      : "#000000",
         "xtick.color"          : "#000000",
         "ytick.color"          : "#000000",
         "grid.linewidth"       : 0.50,
         "grid.linestyle"       : "--",
         "axes.titlecolor"      : 'maroon',
         'axes.titlesize'       : 9,
         'axes.labelweight'     : "bold",
         'legend.fontsize'      : 7.0,
         'legend.title_fontsize': 7.0,
         'font.size'            : 7.5,
         'xtick.labelsize'      : 12.5,
         'ytick.labelsize'      : 9.0,
        }
       )

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)



%%writefile -a myimports.py

print(f"---> Commencing imports-part2")
optuna.logging.set_verbosity = optuna.logging.ERROR
optuna.logging.disable_default_handler()
print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__}")

##################################################################
# Customizing logging for LGBM
class MyLogger:
    """
    This class helps to suppress logs in lightgbm and Optuna
    Source - https://github.com/microsoft/LightGBM/issues/6014
    """

    def init(self, logging_lbl: str):
        self.logger = logging.getLogger(logging_lbl)
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        self.logger.error(message)

l = MyLogger()
l.init(logging_lbl = "lightgbm_custom")
lgb.register_logger(l)

##################################################################
# Customizing logging for XGBoost
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f'xgb_optimize.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

class XGBLogging(xgb.callback.TrainingCallback):
    """log train logs to file"""

    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(self, model, epoch:int,
                        evals_log:xgb.callback.TrainingCallback.EvalsLog
                        ):

        if self.epoch_log_interval <= 0:
            pass

        elif (epoch %  self.epoch_log_interval == 0):
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                    logger.info(f"XGBLogging epoch {epoch} dataset {data} {metric_name} {score}")

        return False

# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 200)
print(f"---> Imports- part 2 done")



%%writefile -a myimports.py

print(f"---> Seeding everything")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2025)
print(f"\n---> Imports done")


%%writefile -a myutils.py

class Utils:
    """
    This class creates and uses several utility methods to be used across the code
    """

    def __init__(self):
        pass

    def ScoreMetric(
        self, 
        ytrue, 
        ypreds, 
        cutoff = 0.50, 
        **params
    )-> np.float32:
        "Competition score metric for the assignment"
        score = accuracy_score(
            np.uint8( np.round(ytrue, 0)  ) , 
            np.uint8( np.where(ypreds >= cutoff, 1, 0) ) 
        )
        return score
        
    def pp_preds(self, ypreds : np.ndarray)-> np.ndarray :
        "Post-processes the predictions using min-max values from the training data"
        return np.clip( ypreds , a_min = 0, a_max = 1 )

    def CleanMemory(self):
        "This method cleans the memory off unused objects and displays the cleaned state RAM usage"

        collect();
        libc.malloc_trim(0)
        pid        = getpid()
        py         = Process(pid)
        memory_use = py.memory_info()[0] / 2. ** 30
        return f"\nRAM usage = {memory_use :.4} GB"

utils = Utils()
collect()
print()


%%writefile -a myutils.py

def reduce_mem_usage(
    dataframe, dataset: str
):  
    """
    Reducing memory for the dataset based on datatypes
    Source - https://www.kaggle.com/competitions/drw-crypto-market-prediction/discussion/580485

    Inputs - 
    dataframe - pd.DataFrame
    dataset   - str : dataset label

    Returns - 
    dataframe - reduced memory dataset
    """
    
    print(f'---> Reducing memory usage for: {dataset}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2

    for col in tqdm( dataframe.columns ):
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            try:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)
            except:
                pass

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('---> Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('---> Memory usage after: {:.2f} MB'.format(final_mem_usage))

    dec = float(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage)
    print(f'---> Decreased memory usage by {round(dec, 4)} percent \n')
    return dataframe



%%writefile -a myutils.py

class AdversarialCVMaker:
    """
    This class assists in adversarial CV between the train and test data with the below steps-

    1. Consider any classifier as a base model, I prefer any boosted tree model as I don't have to focus too much on preprocessing
    2. Load the train and test set features
    3. Make a new target column with 1 for test set occurrances and 0 for train-set
    4. Classify to predict the test set instances with the features and new target from the step above
    
    If the AUC score hovers around 50% (random model), then we can be sure that the train and test set have similar distributions 
    Else, if our model is able to differentiate between the train and test data, then our model is unlikely to generalize as-is.
    In this case, further adjustments may be necesary 
    """

    def __init__(self, n_splits: int = 5) :
        self.model = \
        LGBMC(
            n_estimators     = 200,
            learning_rate    = 0.02,
            max_depth        = 3, 
            colsample_bytree = 0.50,
            objective        = "binary",
            metric           = "auc",
            random_state     = 42,
            device           = "gpu" if torch.cuda.is_available() else "cpu",
        )

        self.n_splits = n_splits

    @staticmethod
    def scorer(ytrue, ypreds):
        return roc_auc_score( ytrue, ypreds )

    def make_cv(
        self, Xtrain, Xtest, **fit_params,
        ):
        "Fits the model with the auxilary target and calculates the AUC score for the CV"

        df = \
        pd.concat(
            [Xtrain.assign(**{"target" : 0}), 
             Xtest.assign(**{"target" : 1}),
            ], 
            axis=0, ignore_index = True
        )

        cv     = StratifiedKFold(n_splits = self.n_splits, random_state = 42, shuffle = True)
        scores = 0
        
        for train_idx, dev_idx in cv.split(df, df["target"]) :
            Xtr  = df.loc[train_idx].drop("target", axis=1)
            Xdev = df.loc[dev_idx].drop("target", axis=1)
            ytr  = df.loc[train_idx, "target"]
            ydev = df.loc[dev_idx, "target"]

            cat_cols = list( Xdev.select_dtypes(include = ["string", "category", "object"]).columns )

            if len(cat_cols) > 0 :
                Xtr[cat_cols]  = Xtr[cat_cols].astype("category")
                Xdev[cat_cols] = Xdev[cat_cols].astype("category")
            else:
                pass
                
            model = clone(self.model)
            model.fit(Xtr, ytr)
            dev_preds = model.predict_proba(Xdev)[:,1]
            score = self.scorer(ydev, dev_preds)
            scores += score

        score = scores / self.n_splits

        PrintColor(
            f"\n---> Overall adversarial CV score = {score :,.4f}"
        )

        if score > 0.60 :
            PrintColor(
                f"---> Check for test-train distribution shift\n", color = Fore.RED
            )
        else:
            PrintColor(
                f"---> Train-test distributions are similar\n", color = Fore.GREEN
            )



%%writefile -a training.py

def MakePermImp(
        method, mdl, X, y, ygrp,
        myscorer, 
        n_repeats = 2,
        state = 42,
        ntop: int = 15,
        **params,
):
    """
    This function makes the permutation importance for the provided model and returns the importance scores for all features
    
    Note-
    myscorer - scikit-learn -> metrics -> make_scorer object with the corresponding eval metric and relevant details
    """

    cv        = PredefinedSplit(ygrp)
    n_splits  = ygrp.nunique()
    drop_cols = ["Source", "id", "Id", "Label", "fold_nb"]

    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
        Xtr  = X.iloc[train_idx].drop(drop_cols, axis=1, errors = "ignore")
        Xdev = X.iloc[dev_idx].drop(drop_cols, axis=1, errors = "ignore")
        ytr  = y.loc[Xtr.index]
        ydev = y.loc[Xdev.index]

        model = clone(mdl)
        sel_cols = list(Xdev.columns)
        model.fit(Xtr, ytr)

        imp_ = permutation_importance(model,
                                      Xdev, ydev,
                                      scoring = myscorer,
                                      n_repeats = n_repeats,
                                      random_state = state,
                                      )["importances_mean"]
        imp_ = pd.Series(index = sel_cols, data = imp_)

        display(
            imp_.\
            sort_values(ascending = False).\
            head(ntop).\
            to_frame().\
            transpose().\
            style.\
            format(formatter = '{:,.3f}').\
            background_gradient("icefire", axis=1).\
            set_caption(f"Top {ntop} features")
            )

        return imp_


%%writefile -a training.py

class ModelTrainer:
    "Offline and online training and prediction collation for multiclass problem"

    def __init__(
        self,
        drop_cols      : list,
        len_train      : int  = 750_000,
        test_preds_req : bool = True,
        problem_type   : str  = "binary",
    ):

        self.drop_cols      = drop_cols
        self.len_train      = len_train
        self.test_preds_req = test_preds_req
        self.problem_type   = problem_type

    def _pp_data(self, df, method, cat_cols) :
        "Preprocesses the data for the model"

        if cat_cols is not None :
            if "CB" in method:
                df[cat_cols]  = df[cat_cols].astype("string")
            else:
                df[cat_cols]  = df[cat_cols].astype("category")

        return df.drop(self.drop_cols, axis=1, errors = "ignore")

    def _augment_data(self, Xtr, ytr, extra, **params) : 
        "Augments the training data if requested"
         
        if extra is not None :
            Xtr = pd.concat([Xtr, extra[0][Xtr.columns]], axis = 0, ignore_index = True)
            ytr = pd.concat([ytr, extra[1]], axis = 0, ignore_index = True)
            
        return (Xtr, ytr)
        
    def _compmetric(self, ytrue, ypreds):
        "Competition proxy metric for the assignment"
        score = log_loss(ytrue, ypreds)
        return score

    def predict(self, X, model) :
        "Predicts based on problem type"

        if self.problem_type == "regression" :
            return model.predict(X)
        elif self.problem_type == "binary" :
            return model.predict_proba(X)[:, 1]
        elif self.problem_type == "multiclass" :
            return model.predict_proba(X)       

    def fit_predict(
        self,
        Xtrain,
        ytrain,
        Xtest  : pd.DataFrame | None,
        ygrp   : pd.Series | np.ndarray,
        extra  : list | None,
        method : str,
        mymodel,
        cat_cols : list | None,
        **fit_params,
    ):
        "Trains and predicts for the given multiclass model/ pipeline"

        oof_preds, mdl_preds, fitted_models = [], [], []
        cv     = PredefinedSplit(ygrp)
        scores = []

        PrintColor(f"\n {method} offline model training" )

        if extra is not None :
            print(f"---> NOTE:- Original data is added to each fold of the training process")

        print()
        for fold_nb, (train_idx, dev_idx) in tqdm(enumerate( cv.split(Xtrain, ytrain), start = 1 ), method) :
            Xtr  = Xtrain.iloc[train_idx]
            Xdev = Xtrain.iloc[dev_idx]
            ytr  = ytrain.iloc[train_idx]
            ydev = ytrain.iloc[dev_idx]

            Xtr, ytr  = self._augment_data(Xtr, ytr, extra)
            Xtr, Xdev = self._pp_data(Xtr, method, cat_cols), self._pp_data(Xdev, method, cat_cols)

            if self.test_preds_req :
                Xt = self._pp_data(Xtest, method, cat_cols)

            model = clone(mymodel)
            try:
                model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], **fit_params)
            except:
                model.fit(Xtr, ytr, **fit_params)

            dev_preds  = pd.DataFrame(
                self.predict(Xdev, model),
                index = dev_idx,
                dtype = np.float32
            )
            oof_preds.append(dev_preds)

            score = self._compmetric(ydev,dev_preds)
            print(f"---> OOF score = {score:,.8f} | Fold {fold_nb }")
            scores.append(score)

            if self.test_preds_req :
                test_preds = pd.DataFrame(
                    self.predict(Xt, model),
                    index = Xt.index.values,
                    dtype = np.float32
                )
                mdl_preds.append(test_preds)

        oof_preds = pd.concat(
            oof_preds,
            axis=0,
            ignore_index = False
        ).sort_index(ascending = True).to_numpy()

        if self.test_preds_req :
            mdl_preds = pd.concat(
                mdl_preds,
                axis=0,
                ignore_index = False
            ).groupby(level = 0).mean().to_numpy()
        else:
            print(f"---> Test predictions are not needed")

        PrintColor(
            f"\n---> Overall score = {np.mean(scores):,.8f} +- {np.std(scores):,.8f}",
            color = Fore.RED
        )

        return (fitted_models, oof_preds, mdl_preds)

    def refit_full(
        self,
        X : pd.DataFrame,
        y : pd.Series | np.ndarray,
        Xtest: pd.DataFrame,
        method : str,
        mymodel,
        cat_cols : list | None,
        **fit_params
    ):
        "Refits the model on the full dataset, mostly after CV scheme and returns the test-predictions"

        model = clone(mymodel)
        try:
            model.early_stopping_rounds = None
        except:
            pass

        Xtr  = self._pp_data(X, method, cat_cols)
        Xt   = self._pp_data(Xtest, method, cat_cols)

        model.fit(Xtr, y, **fit_params)
        test_preds = self.predict(Xt, model)
        oof_preds  = self.predict(Xtr, model)

        PrintColor(
            f"---> Refitted {method} model succesfully", color = Fore.BLACK
        )
        return (model, oof_preds, test_preds)


%%writefile -a mypp.py

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. In this case, we need to process the original data target column to be compatible with the competition dataset
    3. Check information and description
    4. Check unique values and nulls
    5. Collate starting features 
    """
    
    def __init__(self):

        self.train             = pd.read_csv(os.path.join(CFG.ip_path,"train.csv"), index_col = 'id') 
        self.test              = pd.read_csv(os.path.join(CFG.ip_path ,"test.csv"), index_col = 'id')
        self.target            = CFG.target 
        
        self.conjoin_orig_data = True if CFG.nb_orig > 0 else False
        self.dtl_preproc_req   = CFG.dtl_preproc_req
        self.test_req          = CFG.test_req
        self.cv                = cv_selector[CFG.mdlcv_mthd]
         
        self.original            = pd.read_csv(CFG.orig_path).drop_duplicates()
        self.original.index      = range(len(self.original))
        self.original.index.name = "id"    
        self.original            = self.original[self.train.columns]

        self.sub_fl = pd.read_csv(os.path.join(CFG.ip_path, "sample_submission.csv"))
        PrintColor(
            f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}"
        )
        
        for tbl in [self.train, self.original, self.test,]:
            obj_cols      = tbl.select_dtypes(include = ["object", "category"]).columns
            tbl.columns   = tbl.columns.str.replace(r"\(|\)|\.|\?|/|\s+","", regex = True)
            
    def _VisualizeDF(self):
        "This method visualizes the heads for the train, test and original data"
        
        PrintColor(f"\nTrain set head", color = Fore.CYAN)
        display(self.train.head(5).style.format(precision = 3))
        
        PrintColor(f"\nTest set head", color = Fore.CYAN)
        display(self.test.head(5).style.format(precision = 3))
        
        PrintColor(f"\nOriginal set head", color = Fore.CYAN)
        display(self.original.head(5).style.format(precision = 3))
              
    def _AddSourceCol(self):
        self.train['Source']    = "Competition"
        self.test['Source']     = "Competition"
        self.original['Source'] = 'Original'
        
        self.strt_ftre = self.test.columns
        return self
          
    def _CollateInfoDesc(self):
        if self.dtl_preproc_req :
            PrintColor(f"\n{'-' * 20} Information and description {'-' * 20}\n", color = Fore.MAGENTA);

            # Creating dataset information and description:
            for lbl, df in {'Train': self.train, 'Test': self.test, 'Original': self.original}.items():
                PrintColor(f"\n{lbl} description\n");
                display(df.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\
                        transpose().\
                        drop(columns = ['count'], errors = 'ignore').\
                        drop([self.target], axis=0, errors = 'ignore').\
                        style.format(formatter = '{:,.2f}').\
                        background_gradient(cmap = 'Blues')
                       );

                PrintColor(f"\n{lbl} information\n");
                display(df.info());
                collect();
        return self;
    
    def _CollateUnqNull(self):
        
        if self.dtl_preproc_req :
            PrintColor(f"\nUnique and null values\n")
            _ = pd.concat([self.train[self.strt_ftre].nunique(), 
                           self.test[self.strt_ftre].nunique(), 
                           self.original[self.strt_ftre].nunique(),
                           self.train[self.strt_ftre].isna().sum(axis=0),
                           self.test[self.strt_ftre].isna().sum(axis=0),
                           self.original[self.strt_ftre].isna().sum(axis=0)
                          ], 
                          axis=1)
            _.columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq', 
                         'Train_Nulls', 'Test_Nulls', 'Original_Nulls'
                        ]
            display(_.T.style.background_gradient(cmap = 'Blues', axis=1).\
                    format(formatter = '{:,.0f}')
                   )
            
        return self
       
    def _ConjoinTrainOrig(self):
        if self.conjoin_orig_data :
            PrintColor(f"\n\nTrain shape before conjoining with original = {self.train.shape}")
            train = pd.concat([self.train] + [self.original] * CFG.nb_orig, 
                              axis=0, 
                              ignore_index = True
                             )
            PrintColor(f"Train shape after conjoining with original= {train.shape}")

            train.index = range(len(train))
            train.index.name = 'id'

        else:
            PrintColor(f"\nWe are using the competition training data only")
            train = self.train
        return train
       
    def DoPreprocessing(self):
        self._VisualizeDF()
        self._AddSourceCol()
        self._CollateInfoDesc()
        self._CollateUnqNull()
        self.train = self._ConjoinTrainOrig()

        self.train = self.train.dropna(subset = [self.target])
        self.train.index = range(len(self.train))
        
        self.cat_cols  = \
        list(
            self.test.drop("Source", axis=1).select_dtypes(["object", "string", "category"]).columns
        )
        self.cont_cols = \
        [c for c in self.strt_ftre if c not in self.cat_cols + ['Source']]
        
        return self ;
            

