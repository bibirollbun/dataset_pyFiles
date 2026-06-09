


%%writefile -a req_kaggle.txt

scikit-learn==1.6.0
xgboost==2.1.3
numpy==1.26.4
scipy==1.14.1
polars==1.21.0
pytorch_tabnet


%%writefile -a req_colab.txt

xgboost==2.1.3
catboost==1.2.7
numpy==1.26.4
scipy==1.14.1
polars==1.21.0
colorama
cloudpickle
optuna
pytorch_tabnet


%%writefile req_ag.txt

polars==1.21.0
colorama
optuna
catboost==1.2.7
autogluon.tabular
ray==2.10.0
dask


%%writefile -a requirements_lama.txt

lightautoml
colorama
polars==1.21.0


%%writefile -a myimports.py

print(f"\n---> Commencing imports- part1")

from gc import collect
from warnings import filterwarnings
filterwarnings('ignore')
from IPython.display import display_html, clear_output
import os, sys, logging, re, joblib, ctypes, shutil, random, torch
from copy import deepcopy
collect()

# General library imports:-
from os import path, walk, getpid
from psutil import Process
import re
from collections import Counter
from itertools import product, combinations

import ctypes
libc = ctypes.CDLL("libc.so.6")
from pprint import pprint
from functools import partial
from copy import deepcopy
import pandas as pd, numpy as np, os
import polars as pl
import polars.selectors as cs

import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore, Style, init
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm.notebook import tqdm


%%writefile -a myimports.py

print(f"\n---> Commencing imports- part2")

# Pipeline specifics:-
from sklearn.preprocessing import (RobustScaler,
                                   MinMaxScaler,
                                   StandardScaler,
                                   FunctionTransformer,
                                   PowerTransformer,
                                   OrdinalEncoder,
                                   OneHotEncoder, 
                                  )

try:
    from sklearn.preprocessing import TargetEncoder
except:
    print(f"---> Target encoder is not available in the present sklearn version")

from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF,
                                     StratifiedKFold as SKF,
                                     StratifiedGroupKFold as SGKF,
                                     KFold,
                                     GroupKFold as GKF,
                                     RepeatedKFold as RKF,
                                     PredefinedSplit as PDS,
                                     cross_val_score,
                                     cross_val_predict,
                                    )
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector

# ML Model training:-
from sklearn.metrics import *

import xgboost as xgb, lightgbm as lgb, catboost as cb
from xgboost import QuantileDMatrix, XGBClassifier as XGBC, XGBRegressor as XGBR
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC, LGBMRegressor as LGBMR
from catboost import CatBoostClassifier as CBC, Pool, CatBoostRegressor as CBR
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, RandomForestClassifier as RFC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, RandomForestRegressor as RFR
from sklearn.linear_model import LogisticRegression as LRC, Ridge, Lasso

# TabNet models
from pytorch_tabnet.tab_model import (TabNetRegressor as TNR, TabNetClassifier as TNC)

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

print(f"---> Commencing imports - part3")
optuna.logging.set_verbosity = optuna.logging.ERROR
optuna.logging.disable_default_handler()

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

seed_everything(2024)
print(f"\n---> Imports done")


%%writefile -a myutils.py

class Utils:
    """
    This class creates and uses several utility methods to be used across the code
    """;

    def __init__(self):
        pass

    def ScoreMetric(self, ytrue, ypred)-> float:
        """
        This method calculates the metric for the competition
        Inputs- ytrue, ypred:- input truth and predictions
        Output- float:- competition metric
        """

        try:
            score = root_mean_squared_error(ytrue, ypred)
        except:
            score = mean_squared_error(ytrue, ypred, squared = False)
        return score

    def CleanMemory(self):
        "This method cleans the memory off unused objects and displays the cleaned state RAM usage"

        collect()
        libc.malloc_trim(0)
        pid        = getpid()
        py         = Process(pid)
        memory_use = py.memory_info()[0] / 2. ** 30
        return f"\nRAM usage = {memory_use :.4} GB"

    def DisplayAdjTbl(self, *args):
        """
        This function displays pandas tables in an adjacent manner, sourced from the below link-
        https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
        """

        html_str = ''
        for df in args:
            html_str += df.to_html()
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
        collect()

    def DisplayScores(
        self, Scores: pd.DataFrame, TrainScores: pd.DataFrame, methods: list
    ):
        "This method displays the scores and their means"

        args = \
        [Scores.style.format(precision = 5).\
         background_gradient(cmap = "Blues", subset = methods + ["Ensemble"]).\
         set_caption(f"\nOOF scores across methods and folds\n"),

         TrainScores.style.format(precision = 5).\
         background_gradient(cmap = "Pastel2", subset = methods).\
         set_caption(f"\nTrain scores across methods and folds\n")
        ];

        PrintColor(f"\n\n\n---> OOF score across all methods and folds\n",
                   color = Fore.LIGHTMAGENTA_EX
                   )
        self.DisplayAdjTbl(*args)

        print('\n')
        display(Scores.mean().to_frame().\
                transpose().\
                style.format(precision = 5).\
                background_gradient(cmap = "mako", axis=1,
                                    subset = Scores.columns
                                   ).\
                set_caption(f"\nOOF mean scores across methods and folds\n")
               )


utils = Utils()
collect()
print()


%%writefile -a mytrainer.py

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

    cv        = PDS(ygrp)
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


%%writefile -a mytrainer.py

class ModelTrainer:
    "This class trains the provided model on the train-test data and returns the predictions and fitted models"

    def __init__(
        self,
        problem_type   : str   = "regression", 
        es             : int   = 100,
        target         : str   = "",
        metric_lbl     : str   = "rmse",
        orig_req       : bool  = False,
        orig_all_folds : bool  = False,
        drop_cols      : list  = ["Source", "id", "Id", "Label", "fold_nb"],
        pp_preds       : bool  = False,
    ):
        """
        Key parameters-
        es_iter  - early stopping rounds for boosted trees
        pp_preds - do you want to post-process predictions (true/ false boolean)
        """

        self.problem_type   = problem_type
        self.es_iter        = es
        self.target         = target
        self.drop_cols      = drop_cols + [self.target]
        self.metric_lbl     = metric_lbl
        self.orig_req       = orig_req
        self.orig_all_folds = orig_all_folds
        self.pp_preds       = pp_preds
        
    def ScoreMetric(self, ytrue, ypreds):
        "Returns the score metric based on the metric label specified"

        if self.metric_lbl.lower() == "mae" :
            return mean_absolute_error(ytrue, ypreds)
            
        elif self.metric_lbl.lower() == "rmse" :
            try:
                return root_mean_squared_error(ytrue, ypreds) 
            except:
                return np.sqrt(mean_squared_error(ytrue, ypreds))

    def PlotFtreImp(
        self, 
        ftreimp: pd.Series, 
        method: str,
        ntop: int = 50,
        title_specs: dict = {'fontsize': 12,'fontweight' : 'bold','color': '#992600'},
        **params,
    ):
        "This function plots the feature importances for the model provided"

        print()
        
        with sns.axes_style("white"):
            fig, ax = plt.subplots(1, 1, figsize = (25, 7.5))
    
            ftreimp.sort_values(ascending = False).\
            head(ntop).\
            plot.bar(ax = ax, color = "#1285c7")
            ax.set_title(
                f"Feature Importances - {method}", 
                **title_specs
            )
    
            plt.tight_layout()
            plt.show()
        print()

    def PostProcessPreds(self, ypred):
        "This method post-processes predictions optionally"
        if self.pp_preds :
            return np.clip(ypred, a_min = 15.0, a_max = 150.0)
        else:
            return ypred

    def LoadData(
        self, X, y, Xtest,
        train_idx : list = [],
        dev_idx   : list = [],
    ):
        "This method loads the train and test data for the model fold using/ not using the original data"

        try:
            mycol = X['Source']
        except KeyError:
            X["Source"], Xtest["Source"] = "Competition", "Competition"

        if self.orig_req == False:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        elif self.orig_req == True and self.orig_all_folds == True:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

            orig_x = X.query("Source == 'Original'")[Xtr.columns]
            orig_y = y.iloc[orig_x.index]

            Xtr = pd.concat([Xtr, orig_x], axis = 0, ignore_index = True)
            ytr = pd.concat([ytr, orig_y], axis = 0, ignore_index = True)

        elif self.orig_req == True and self.orig_all_folds == False:
            Xtr  = X.iloc[train_idx].drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        Xt = Xtest[Xdev.columns]

        print(f"\n---> Shapes = {Xtr.shape} {ytr.shape} -- {Xdev.shape} {ydev.shape} -- {Xt.shape}")
        return (Xtr, ytr, Xdev, ydev, Xt)
    
    def MakePreds(self, X, fitted_model):
        "This method creates the model predictions based on the model provided, with optional post-processing"

        if self.problem_type == "regression":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict(X.to_numpy()).flatten())
            else:
                return self.PostProcessPreds(fitted_model.predict(X))
                
        elif self.problem_type == "binary":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict_proba(X.to_numpy()[:,1]).flatten())
            else:
                return self.PostProcessPreds(fitted_model.predict_proba(X)[:, 1])
                
        elif self.problem_type == "multiclass":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict_proba(X.to_numpy()))
            else:
                return self.PostProcessPreds(fitted_model.predict_proba(X))

    def MakeOrigPreds(
            self, orig: pd.DataFrame, fitted_models: list, n_splits : int, ygrp: pd.Series,
            ):
        "This method creates the original data predictions separately only if required"

        if self.orig_req == False:
            orig_preds = 0

        elif self.orig_req == True and self.orig_all_folds == True:
            orig_preds = 0
            df = orig.drop(self.drop_cols, axis = 1, errors = "ignore")

            for fitted_model in fitted_models:
                orig_preds = orig_preds + (self.MakePreds(df, fitted_model) / n_splits)

        elif self.orig_req == True and self.orig_all_folds == False:
            len_orig   = orig.shape[0]
            orig.index = range(len_orig)
            orig_ygrp  = ygrp[-1 * len_orig:]
            orig_ygrp.index = range(len_orig)
            
            orig_preds = np.zeros(len_orig)
            for fold_nb, fitted_model in enumerate(fitted_models):
                df = \
                orig.iloc[orig_ygrp.loc[orig_ygrp == fold_nb].index].\
                drop(self.drop_cols, axis=1, errors = "ignore")
                
                orig_preds[df.index] = self.MakePreds(df, fitted_model)
                del df
        return orig_preds

    def MakeOfflineModel(
        self, X, y, ygrp, Xtest, mdl, method,
        test_preds_req   : bool = True,
        ftreimp_plot_req : bool = True,
        ntop             : int  = 50,
        **params,
    ):
        """
        This function trains the provided model on the dataset and cross-validates appropriately

        Inputs-
        X, y, ygrp       - training data components (Xtrain, ytrain, fold_nb)
        Xtest            - test data (optional)
        model            - model object for training
        method           - model method label
        test_preds_req   - boolean flag to extract test set predictions
        ftreimp_plot_req - boolean flag to plot tree feature importances
        ntop             - top n features for feature importances plot

        Returns-
        oof_preds, test_preds - prediction arrays
        fitted_models         - fitted model list for test set
        ftreimp               - feature importances across selected features
        mdl_best_iter         - model average best iteration across folds
        """

        oof_preds     = np.zeros(len(X.loc[X.Source == "Competition"]))
        orig_preds    = np.zeros(len(X.loc[X.Source == "Original"]))
        test_preds    = []
        mdl_best_iter = []
        ftreimp       = 0

        scores, tr_scores, fitted_models = [], [], []

        if self.orig_req == True:
            cv = PDS(ygrp)
        elif self.orig_req == False:
            X  = X.loc[X.Source == "Competition"]
            y  = y.iloc[X.index]
            cv = PDS(ygrp.iloc[0 : len(X)])

        n_splits = ygrp.nunique()

        for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
            Xtr, ytr, Xdev, ydev, Xt = \
            self.LoadData(X, y, Xtest, train_idx, dev_idx)

            model = clone(mdl)

            if "CB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose = 0,
                          early_stopping_rounds = self.es_iter,
                          )
                best_iter = model.get_best_iteration()

            elif "LGB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          callbacks = [log_evaluation(0),
                                       early_stopping(stopping_rounds = self.es_iter, verbose = False,),
                                       ],
                          eval_metric = mymetric,
                          )
                best_iter = model.best_iteration_

            elif "XGB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose  = 0,
                          )
                best_iter = model.best_iteration

            elif "TN" in method :
                model.fit(
                    Xtr.to_numpy(), ytr.to_numpy().reshape(-1,1),
                    eval_set    = [(Xdev.to_numpy(), ydev.to_numpy().reshape(-1,1))],
                    eval_name   = ["dev"],
                    eval_metric = ['rmse'],
                    max_epochs  = 100,
                    patience    = 6,
                    batch_size  = 128,
                    virtual_batch_size = 64,
                )

            else:
                model.fit(Xtr, ytr)
                best_iter = -1

            fitted_models.append(model)

            try:
                ftreimp += model.feature_importances_
            except:
                try:
                    ftreimp += model.coef_.flatten()
                except:
                    pass

            try:
                ftreimp += model["M"].feature_importances_
            except:
                try:
                    ftreimp += model["M"].coef_.flatten()
                except:
                    pass
                         
            dev_preds = self.MakePreds(Xdev, model)
            oof_preds[Xdev.index] = dev_preds

            train_preds  = self.MakePreds(Xtr, model)
            tr_score     = self.ScoreMetric(ytr.values.flatten(), train_preds)
            score        = self.ScoreMetric(ydev.values.flatten(), dev_preds)

            scores.append(score)
            tr_scores.append(tr_score)

            nspace = 15 - len(method) - 2 if fold_nb <= 9 else 15 - len(method) - 1

            if best_iter > 0 :
                PrintColor(f"{method} Fold{fold_nb} {' ' * nspace} OOF = {score:.6f} | Train = {tr_score:.6f} | Iter = {best_iter:,.0f} ")
            else:
                PrintColor(f"{method} Fold{fold_nb} {' ' * nspace} OOF = {score:.6f} | Train = {tr_score:.6f} ")
                
            mdl_best_iter.append(best_iter)

            if test_preds_req:
                test_preds.append(self.MakePreds(Xt, model))
            else:
                pass

        test_preds    = np.mean(np.stack(test_preds, axis = 1), axis=1)
        ftreimp       = pd.Series(ftreimp, index = Xdev.columns)
        mdl_best_iter = np.uint16(np.amax(mdl_best_iter))

        if ftreimp_plot_req :
            print()
            self.PlotFtreImp(ftreimp, method = method, ntop = ntop,)
        else:
            pass

        PrintColor(f"\n---> {np.mean(scores):.6f} +- {np.std(scores):.6f} | OOF", color = Fore.RED)
        PrintColor(f"---> {np.mean(tr_scores):.6f} +- {np.std(tr_scores):.6f} | Train", color = Fore.RED)

        if best_iter > 0:
            pass
        else:
            PrintColor(
                f"---> Max best iteration = {mdl_best_iter :,.0f}",
                color = Fore.RED
            )

        if self.orig_req:
            print(f"---> Collecting original predictions")
            orig_preds = self.MakeOrigPreds(X.loc[X.Source == "Original"],
                                            fitted_models,
                                            n_splits,
                                            ygrp,
                                            )
            oof_preds = np.concatenate([oof_preds, orig_preds], axis= 0)
        else:
            pass
        return (fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter)

    def MakeOnlineModel(
        self, X, y, Xtest, 
        model, method,
        test_preds_req : bool = False,
    ):
        "This method refits the model on the complete train data and returns the model fitted object and predictions"

        try:
            model.early_stopping_rounds = None
        except:
            pass

        if "TN" in method:
            model.fit(
                X.to_numpy(), y.to_numpy().reshape(-1,1),
                max_epochs  = 100,
                batch_size  = 128,
                virtual_batch_size = 64,
                )
        else:
            try:
                model.fit(X, y, verbose = 0)
            except:
                model.fit(X, y,)

        oof_preds  = self.MakePreds(X, model)
        if test_preds_req:
            test_preds = self.MakePreds(Xtest[X.columns], model)
        else:
            test_preds = 0
            
        return (model, oof_preds, test_preds)


%%writefile -a myensembler.py

class HillClimber:
    "This class develops the Hill Climber algorithm for the provided datasets"

    def __init__(self):
        self.ScoreMetric = utils.ScoreMetric

    def DoHillClimb(
        target:str,
        direction:str,
        cutoff:float,
        neg_wgt:str,
        OOF_Preds: pd.DataFrame,
        Mdl_Preds: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ):
        """
        This method performs hill-climbing on the OOF and Test predictions dataset and returns the below-
        1. OOF ensemble predictions
        2. Test set predictions
        3. Score dataframe (with scores in sort-order)
        """

        oof_df     = OOF_Preds
        test_preds = Mdl_Preds
    
        # Scoring the individual models:-
        Scores = pd.DataFrame(index = oof_df.columns, columns = ['Score'])
    
        for col in oof_df.columns:
            Scores.at[col, 'Score'] = self.ScoreMetric(y, oof_df[col].values.flatten())
    
        # Sorting scores
        Scores.sort_values(
            by= 'Score',
            ascending = [True if direction == 'minimize' else False],
            inplace = True,
        )
    
        PrintColor(f"\n----- Data preparation: ------ \n");
        display(
            Scores.
            transpose().
            style.
            format(precision = 5)
            )
    
        PrintColor(f"\n ----- Initiating hill-climb ----- \n");
        STOP = False
        current_best_ensemble   = oof_df.iloc[:,0]
        current_best_test_preds = test_preds.iloc[:,0]
        MODELS                  = oof_df.iloc[:,1:]
    
        if neg_wgt == "Y":
            weight_range = np.arange(-0.5,0.51,0.01);
        else:
            weight_range = np.arange(0.01,0.51,0.01);
    
        history = [self.ScoreMetric(y, current_best_ensemble)]
    
        i=0
    
        # Hill climbing algorithm:-
        while not STOP:
            i+=1
    
            potential_new_best_cv_score = self.ScoreMetric(y, current_best_ensemble)
            k_best, wgt_best = None, None
    
            for k in MODELS:
                for wgt in weight_range:
                    potential_ensemble = (1- wgt) * current_best_ensemble + wgt * MODELS[k]
                    cv_score = self.ScoreMetric(y, potential_ensemble)
    
                    if direction == 'minimize':
                        if cv_score < potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
                    if direction == 'maximize':
                        if cv_score > potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
            if k_best is not None:
                current_best_ensemble   = (1- wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
                current_best_test_preds = (1- wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
                MODELS.drop(k_best, axis=1, inplace=True)
    
                if MODELS.shape[1]==0:  STOP = True
    
                num_space = 50 - len(k_best) if i <= 9 else 49 - len(k_best)
                PrintColor(f" {i}.{k_best} {' ' * num_space} Weight = {wgt_best: .4f} {' ' * 5} Score = {potential_new_best_cv_score:.6f}",
                           color = Fore.CYAN
                          )
                del num_space
    
                history.append(potential_new_best_cv_score)
    
            else:
                STOP = True
    
        return (current_best_ensemble, current_best_test_preds, Scores)



%%writefile -a myimports_lama.py

from warnings import filterwarnings
filterwarnings('ignore')

import os
from os import path, walk, getpid
from psutil import Process
import re
from collections import Counter
from itertools import product
from gc import collect

import ctypes
libc = ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from tqdm.notebook import tqdm
import tempfile

# Essential DS libraries
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.metrics import *

from sklearn.model_selection import PredefinedSplit as PDS
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

print(f"---> CUDA available = {torch.cuda.is_available()}\n\n")


%%writefile -a myimports_ag.py

import numpy as np, pandas as pd
import polars as pl
import polars.selectors as cs
import re, os, joblib, logging
from gc import collect

from IPython.display import display_html, clear_output
from pprint import pprint
from tqdm.notebook import tqdm
from colorama import Fore, Back, Style
from os import path, walk, getpid
from psutil import Process
import ctypes
libc = ctypes.CDLL("libc.so.6")

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import (StratifiedKFold as SKF, GroupKFold as GKF, KFold as KF, PredefinedSplit as PDS)
from sklearn.metrics import *
from autogluon.tabular import TabularPredictor, TabularDataset

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)


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

        if CFG.fulltrain_req == True : 
            self.train = \
            pd.concat(
                [pd.read_csv(os.path.join(CFG.ip_path,"train.csv"), index_col = 'id'),
                 pd.read_csv(os.path.join(CFG.ip_path,"training_extra.csv"), index_col = 'id'),   
                ], 
                axis=0, ignore_index = True,
            )
            PrintColor(f"---> We use only the extra training data and the initial training data")
            
        elif CFG.fulltrain_req == False :
            self.train = pd.read_csv(os.path.join(CFG.ip_path,"training_extra.csv"), index_col = 'id') 
            PrintColor(f"---> We use only the extra training data only")

        self.test              = pd.read_csv(os.path.join(CFG.ip_path ,"test.csv"), index_col = 'id')
        self.target            = CFG.target 
        self.conjoin_orig_data = True if CFG.nb_orig > 0 else False
        self.dtl_preproc_req   = CFG.dtl_preproc_req
        self.test_req          = CFG.test_req
        self.cv                = cv_selector[CFG.mdlcv_mthd]
         
        self.original = pd.read_csv(CFG.orig_path)
        self.original.index = range(len(self.original))
        self.original.index.name = "id"    
        self.original = self.original[self.train.columns]

        self.sub_fl = pd.read_csv(os.path.join(CFG.ip_path, "sample_submission.csv"))
        PrintColor(f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}")
        
        for tbl in [self.train, self.original, self.test]:
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
        if self.dtl_preproc_req == "Y":
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
        
        if self.dtl_preproc_req == "Y":
            # Dislaying the unique values across train-test-original:-
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
            
        return self;
       
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
        
        return self 
            
collect()
print()


%%writefile -a mypp.py

class FeaturePlotter:
    """
    This class develops plots for the targets, continuous and category features
    """;
    
    def __init__(
        self, target: str, ftre_plots_req: bool, title_specs : dict, grid_specs: dict,
    ):
        self.target         = target
        self.ftre_plots_req = ftre_plots_req
        self.title_specs    = title_specs
        self.grid_specs     = grid_specs
        
    def MakeTgtPlot(
        self, train, original
    ):
        "This method returns the target plots"
        
        if self.ftre_plots_req == True: 
            with sns.axes_style("white") :
                fig, axes = \
                plt.subplots(
                    1,2, figsize = (25, 6), gridspec_kw = {'wspace': 0.35}
                )
                
                for i, df in tqdm(enumerate([train, original]), f"Target plot- {self.target} ---> "):
                    ax= axes[i]

                    df[self.target] = np.int32(np.round(df[self.target].values, 0))
                    a = df[self.target].value_counts(normalize = True)
                    a.sort_index().plot.bar(color = 'tab:blue', ax = ax)
                    
                    df_name = 'Train' if i == 0 else "Original"
                    _ = ax.set_title(f"\n{df_name} data- {self.target}\n", **self.title_specs)
    
                plt.tight_layout()
                plt.show()
                
    def MakeCatFtrePlots(
        self, cat_cols, train, test, original
    ):
        "This method returns the category feature plots";
        
        if cat_cols != [] and self.ftre_plots_req == True:
            fig, axes = \
            plt.subplots(len(cat_cols), 3, 
                         figsize = (25, len(cat_cols)* 4.5), 
                         gridspec_kw = {'wspace': 0.45, 'hspace': 0.40},
                        );

            for i, col in enumerate(cat_cols):
                ax = axes[i, 0] if len(cat_cols) > 1 else axes[0]
                a = train[col].value_counts(normalize = True)
                a.sort_index().plot.barh(ax = ax, color = '#007399')
                ax.set_title(f"{col}_Train", **self.title_specs)
                ax.set_xticks(np.arange(0.0, 1.01, 0.05), 
                              labels = np.round(np.arange(0.0, 1.01, 0.05),2), 
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '')
                del a;

                ax = axes[i, 1] if len(cat_cols) > 1 else axes[1];
                a = test[col].value_counts(normalize = True);
                a.sort_index().plot.barh(ax = ax, color = '#0088cc');
                ax.set_title(f"{col}_Test", **self.title_specs);
                ax.set_xticks(np.arange(0.0, 1.01, 0.05), 
                              labels = np.round(np.arange(0.0, 1.01, 0.05),2), 
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '');
                del a;

                ax = axes[i, 2] if len(cat_cols) > 1 else axes[2];
                a = original[col].value_counts(normalize = True);
                a.sort_index().plot.barh(ax = ax, color = '#0047b3');
                ax.set_title(f"{col}_Original", **self.title_specs);
                ax.set_xticks(np.arange(0.0, 1.01, 0.05), 
                              labels = np.round(np.arange(0.0, 1.01, 0.05), 2), 
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '');
                del a;       

            plt.suptitle(
                f"Category column plots", **self.title_specs, y= 0.92
            )
            plt.tight_layout();
            plt.show();
            
    def MakeContColPlots(
        self, cont_cols, train, test, original, 
    ):
        "This method returns the continuous feature plots"
        
        if self.ftre_plots_req == True:
            df = pd.concat([train[cont_cols].assign(Source = 'Train'), 
                            test[cont_cols].assign(Source = 'Test'),
                            original[cont_cols].assign(Source = "Original")
                           ], 
                           axis=0, ignore_index = True
                          );

            fig, axes = plt.subplots(len(cont_cols), 4 ,figsize = (16, len(cont_cols) * 4.2), 
                                     gridspec_kw = {'hspace': 0.35, 
                                                    'wspace': 0.3, 
                                                    'width_ratios': [0.80, 0.20, 0.20, 0.20]
                                                   }
                                    );

            for i,col in enumerate(cont_cols):
                ax = axes[i,0];
                sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', 
                            palette = ['#0039e6', '#ff5500', '#00b300'], 
                            ax = ax, linewidth = 2.1
                           )
                ax.set_title(f"\n{col}", **self.title_specs)
                ax.grid(**self.grid_specs);
                ax.set(xlabel = '', ylabel = '');

                ax = axes[i,1]
                sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,
                            color = '#33ccff', saturation = 0.90, linewidth = 0.90, 
                            fliersize= 2.25,
                            ax = ax
                           )
                ax.set(xlabel = '', ylabel = '');
                ax.set_title(f"Train", **self.title_specs);

                ax = axes[i,2]
                sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                            color = '#80ffff', saturation = 0.6, linewidth = 0.90, 
                            ax = ax); 
                ax.set(xlabel = '', ylabel = '')
                ax.set_title(f"Test", **self.title_specs)

                ax = axes[i,3]
                sns.boxplot(data = df.loc[df.Source == 'Original', [col]], y = col, width = 0.25, fliersize= 2.25,
                            color = '#99ddff', saturation = 0.6, linewidth = 0.90, 
                            ax = ax); 
                ax.set(xlabel = '', ylabel = '')
                ax.set_title(f"Original", **self.title_specs)

            plt.suptitle(f"\nDistribution analysis- continuous columns\n", **CFG.title_specs, 
                         y = 0.95, x = 0.50
                        )
            plt.tight_layout()
            plt.show()
     
    def CalcSkew(self, cont_cols, train, test, original):
        "This method calculates the skewness across columns"
        
        if self.ftre_plots_req == True: 
            skew_df = pd.DataFrame(index = cont_cols);
            for col, df in {"Train"   : train[cont_cols], 
                            "Test"    : test[cont_cols], 
                            "Original": original[cont_cols]
                           }.items():   
                skew_df = \
                pd.concat([skew_df, 
                           df.drop(columns =  [self.target, "Source", "id"], errors = "ignore").skew()],
                           axis=1).rename({0: col}, axis=1);

            PrintColor(
                f"\nSkewness across independent features\n"
            )
            display(
                skew_df.transpose().
                style.format(precision = 2).
                background_gradient("PuBuGn")
            )

print()
collect()

