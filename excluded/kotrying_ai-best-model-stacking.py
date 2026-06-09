%%capture
!pip install fontstyle
!pip install omegaconf


import os
import numpy as np
import pandas as pd
import random
import torch
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
%matplotlib inline
import fontstyle
from tqdm.rich import tqdm

import glob
from dataclasses import dataclass, asdict
import yaml
from omegaconf import OmegaConf
import pickle
from time import time

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from catboost import Pool

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize


class CFG:
    TRAIN = "/kaggle/input/narou/train.csv"
    TEST = "/kaggle/input/narou/test.csv"
    SUB = "/kaggle/input/narou/sample_submission.csv"
    EXP = "/kaggle/input/owner-s-best-model-ai-syosetu-ai/output/exp001"


train = pd.read_csv(CFG.TRAIN)
test = pd.read_csv(CFG.TEST)
sub = pd.read_csv(CFG.SUB)

display(train)
display(test)
display(sub)


oofs_path_list = glob.glob(CFG.EXP + "/*/preds/oof_pred.npy")
oofs_fav_path_list = sorted([l for l in oofs_path_list if 'fav_novel_cnt' in l])
oofs_all_path_list = sorted([l for l in oofs_path_list if 'all_point' in l])
oofs_global_path_list = sorted([l for l in oofs_path_list if 'global_point' in l])

display(oofs_fav_path_list)
display(oofs_all_path_list)
display(oofs_global_path_list)


model_names = ["lightgbm", "xgboost", "catboost", "nn"]

oof_fav_df = pd.DataFrame()
for model_name in model_names:
    oof_fav_df[model_name] = np.load([path for path in oofs_fav_path_list if model_name in path][0])
display(oof_fav_df)

oof_all_df = pd.DataFrame()
for model_name in model_names:
    oof_all_df[model_name] = np.load([path for path in oofs_all_path_list if model_name in path][0])
display(oof_all_df)

oof_global_df = pd.DataFrame()
for model_name in model_names:
    oof_global_df[model_name] = np.load([path for path in oofs_global_path_list if model_name in path][0])
display(oof_global_df)

oof_df = pd.DataFrame()
oof_df[[f"{model_name}_fav" for model_name in model_names]] = oof_fav_df
oof_df[[f"{model_name}_all" for model_name in model_names]] = oof_all_df
oof_df[[f"{model_name}_global" for model_name in model_names]] = oof_global_df
display(oof_df)


preds_path_list = glob.glob(CFG.EXP + "/*/preds/*.csv")
preds_fav_path_list = [l for l in preds_path_list if 'fav_novel_cnt' in l]
preds_all_path_list = [l for l in preds_path_list if 'all_point' in l]
preds_global_path_list = [l for l in preds_path_list if 'global_point' in l]
display(preds_fav_path_list)
display(preds_all_path_list)
display(preds_global_path_list)


model_names = ["lightgbm", "xgboost", "catboost", "nn"]

pred_fav_df = pd.DataFrame()
for model_name in model_names:
    pred_fav_df[model_name] = pd.read_csv([path for path in preds_fav_path_list if model_name in path][0], usecols=["fav_novel_cnt"])
display(pred_fav_df)

pred_all_df = pd.DataFrame()
for model_name in model_names:
    pred_all_df[model_name] = pd.read_csv([path for path in preds_all_path_list if model_name in path][0], usecols=["all_point"])
display(pred_all_df)

pred_global_df = pd.DataFrame()
for model_name in model_names:
    pred_global_df[model_name] = pd.read_csv([path for path in preds_global_path_list if model_name in path][0], usecols=["global_point"])
display(pred_global_df)

pred_df = pd.DataFrame()
pred_df[[f"{model_name}_fav" for model_name in model_names]] = pred_fav_df
pred_df[[f"{model_name}_all" for model_name in model_names]] = pred_all_df
pred_df[[f"{model_name}_global" for model_name in model_names]] = pred_global_df
display(pred_df)


train_stack = pd.concat([oof_df, train[['novel_id', 'global_point', 'fav_novel_cnt', 'all_point']]], axis=1)
display(train_stack)


test_stack = pd.concat([pred_df, test[['novel_id']]], axis=1)
display(test_stack)


class Timer:
    '''
    タイマークラス

    引用: https://www.guruguru.science/competitions/21/discussions/ab028e86-d011-485e-8844-45d15717fec4/

    Args:
        logger (Optional[logging.Logger]): ロガーオブジェクト。デフォルトはNone。
        format_str (str): 出力フォーマットの文字列。デフォルトは"{:.3f}[s]"。
        prefix (Optional[str]): 出力文字列の先頭に追加するプレフィックス。デフォルトはNone。
        suffix (Optional[str]): 出力文字列の末尾に追加するサフィックス。デフォルトはNone。
        sep (str): プレフィックス/サフィックスとフォーマット文字列の間の区切り文字。デフォルトは" "。

    Attributes:
        format_str (str): 出力フォーマットの文字列。
        logger (Optional[logging.Logger]): ロガーオブジェクト。
        start (float): タイマーの開始時間。
        end (float): タイマーの終了時間。

    '''
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        '''
        タイマーの経過時間を返します。

        Returns:
            float: タイマーの経過時間（秒）。
        '''
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        '''
        コンテキストマネージャの開始時に呼び出されるメソッドです。
        タイマーの開始時間を記録します。
        '''
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''
        コンテキストマネージャの終了時に呼び出されるメソッドです。
        タイマーの終了時間を記録し、経過時間を出力します。

        Args:
            exc_type (Type[BaseException]): 例外の型。
            exc_val (BaseException): 例外オブジェクト。
            exc_tb (TracebackType): トレースバックオブジェクト。
        '''
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def set_seed(seed=1234):
    '''
    乱数シードを設定する関数

    Args:
        seed: 乱数シードの値。

    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def save_results(save_cfg, output_path: str):
    '''
    結果を保存する関数

    Args:
        save_cfg: 保存する結果の設定情報。
        output_path (str): 結果の出力パス。

    '''
    try:
        results_df = pd.read_csv(output_path)
    except:
        results_df = pd.DataFrame()
    results_df = results_df.append(asdict(save_cfg), ignore_index=True)
    results_df.to_csv(output_path, index=False)

def MSLE(y_true: np.array, y_pred: np.array) -> np.float64:
    """
    The Mean Squared Log Error (MSLE) metric

    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Our predictions
    :return: The MSLE score
    """
    return mean_squared_log_error(y_true, y_pred, squared=True)

def process_target(y):
    '''
    目的変数を処理する関数
    MSLEでは負の値を処理できないため、負の値は0に変換する

    Args:
        y: 目的変数
        
    Returns:
        numpy.array: 目的変数の自然対数
    '''
    return np.where(y < 0, 0, y)

def preprocess_target(y):
    '''
    目的変数を前処理する関数

    Args:
        y: 目的変数
        
    Returns:
        numpy.array: 目的変数の自然対数
    '''
    return np.log1p(y)

def postprocess_target(y):
    '''
    目的変数を後処理する関数

    Args:
        y: 目的変数
        
    Returns:
        numpy.array: ネイピア数eの目的変数分の累乗から1を引いたもの
    '''
    return np.expm1(y)

# LightGBM
def fit_lightgbm(cfg, X, y,
             folds,
             params: dict=None,
             early_stopping_rounds: int = 50,
             verbose: int = 100,
             suffix: str=''):
    if params is None:
        params = {}
#     params['metrics'] = str(loss_fnc.__name__)
    if not cfg.tuning:
        print(params)

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records, ), dtype=np.float32)

    set_seed(cfg.seed)

    for fold in sorted(folds.unique()):
        if fold == -1: continue
        idx_train = (folds!=fold)
        idx_valid = (folds==fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        _y_train = preprocess_target(y_train) # for MSLE
        _y_valid = preprocess_target(y_valid) # for MSLE

        model = lgb.LGBMRegressor(**params)

        with Timer(prefix="fit fold={} ".format(fold)):
            model.fit(x_train, _y_train,
                      eval_set=[(x_valid, _y_valid)],
                      callbacks=[
                                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True), # 早期停止回数
                                lgb.log_evaluation(verbose) # 表示頻度
                            ],
                     )
            
        model_path = os.path.join(cfg.OUTPUT_MODEL, f'lgb_fold{fold}{suffix}.pkl')
        pickle.dump(model, open(model_path, 'wb'))
        # model = pickle.load(open(model_path, 'rb'))

        pred_i = model.predict(x_valid)
        pred_i = postprocess_target(pred_i) # for MSLE
        pred_i = process_target(pred_i) # for MSLE
        oof_pred[idx_valid] = pred_i
        models.append(model)
        score =  cfg.loss_fnc(y_valid, pred_i)
        if not cfg.tuning:
            print(fontstyle.apply(f' - fold{fold + 1} - {score:.4f}', 'BLACK/BOLD'))

    np.save(os.path.join(cfg.OUTPUT_PREDS, f'oof_pred{suffix}'), oof_pred)
    cv_score =  cfg.loss_fnc(y, oof_pred)

    if not cfg.tuning:
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))
        print(fontstyle.apply(f'FINISH: CV Score: {cv_score:.7f}', 'BLACK/BOLD'))
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))

    return oof_pred, models, cv_score

# XGBoost
def fit_xgboost(cfg, X, y,
                folds,
                params: dict=None,
                early_stopping_rounds: int = 50,
                verbose: int = 100,
                suffix: str=''):
    if params is None:
        params = {}
    # params['metrics'] = str(loss_fnc.__name__)
    if not cfg.tuning:
        print(params)

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records,), dtype=np.float32)

    set_seed(cfg.seed)

    for fold in sorted(folds.unique()):
        if fold == -1: continue
        idx_train = (folds != fold)
        idx_valid = (folds == fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        _y_train = preprocess_target(y_train)  # for MSLE
        _y_valid = preprocess_target(y_valid)  # for MSLE

        xgb_train = xgb.DMatrix(x_train, _y_train, enable_categorical=True)
        xgb_valid = xgb.DMatrix(x_valid, _y_valid, enable_categorical=True)

        with Timer(prefix="fit fold={} ".format(fold)):
            model = xgb.train(
                params,
                dtrain=xgb_train,
                evals=[(xgb_train, 'train'), (xgb_valid, 'eval')],
                num_boost_round=cfg.xgb_param.n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose
            )

        model_path = os.path.join(cfg.OUTPUT_MODEL, f'xgb_fold{fold}{suffix}.pkl')
        pickle.dump(model, open(model_path, 'wb'))

        pred_i = model.predict(xgb_valid)
        pred_i = postprocess_target(pred_i)  # for MSLE
        pred_i = process_target(pred_i)  # for MSLE
        oof_pred[idx_valid] = pred_i
        models.append(model)
        score = cfg.loss_fnc(y_valid, pred_i)
        if not cfg.tuning:
            print(fontstyle.apply(f' - fold{fold + 1} - {score:.4f}', 'BLACK/BOLD'))

    np.save(os.path.join(cfg.OUTPUT_PREDS, f'oof_pred{suffix}'), oof_pred)
    cv_score = cfg.loss_fnc(y, oof_pred)

    if not cfg.tuning:
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))
        print(fontstyle.apply(f'FINISH: CV Score: {cv_score:.7f}', 'BLACK/BOLD'))
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))

    return oof_pred, models, cv_score

# CatBoost
def fit_catboost(cfg, X, y,
                 folds,
                 params: dict=None,
                 early_stopping_rounds: int = 50,
                 verbose: int = 100,
                 suffix: str='',
                 categorical_features: list=None):
    if params is None:
        params = {}
    # params['metrics'] = str(loss_fnc.__name__)
    if not cfg.tuning:
        print(params)

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records,), dtype=np.float32)

    set_seed(cfg.seed)

    for fold in sorted(folds.unique()):
        if fold == -1: continue
        idx_train = (folds != fold)
        idx_valid = (folds == fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        _y_train = preprocess_target(y_train)  # for MSLE
        _y_valid = preprocess_target(y_valid)  # for MSLE

        cat_train = Pool(data=x_train, label=_y_train, cat_features=categorical_features)
        cat_valid = Pool(data=x_valid, label=_y_valid, cat_features=categorical_features)

        with Timer(prefix="fit fold={} ".format(fold)):
            model = CatBoostRegressor(**params)
            model.fit(
                cat_train,
                eval_set=[cat_valid],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                use_best_model=True,
            )

        model_path = os.path.join(cfg.OUTPUT_MODEL, f'catboost_fold{fold}{suffix}.pkl')
        pickle.dump(model, open(model_path, 'wb'))

        pred_i = model.predict(x_valid)
        pred_i = postprocess_target(pred_i)  # for MSLE
        pred_i = process_target(pred_i)  # for MSLE
        oof_pred[idx_valid] = pred_i
        models.append(model)
        score = cfg.loss_fnc(y_valid, pred_i)
        if not cfg.tuning:
            print(fontstyle.apply(f' - fold{fold + 1} - {score:.4f}', 'BLACK/BOLD'))

    np.save(os.path.join(cfg.OUTPUT_PREDS, f'oof_pred{suffix}'), oof_pred)
    cv_score = cfg.loss_fnc(y, oof_pred)

    if not cfg.tuning:
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))
        print(fontstyle.apply(f'FINISH: CV Score: {cv_score:.7f}', 'BLACK/BOLD'))
        print(fontstyle.apply('=' * 50, 'BLACK/BOLD'))

    return oof_pred, models, cv_score

def setup(cfg):

    # dir path
#     cfg.INPUT = os.path.join(cfg.COMPETITION, "input")
    cfg.INPUT = cfg.COMPETITION # kaggle ver
#     cfg.EXP = os.path.join(cfg.COMPETITION, "output", cfg.exp_name)
    cfg.EXP = os.path.join("/kaggle/working/", "output", cfg.exp_name) # kaggle ver

    # file path
    cfg.TRAIN_FILE = os.path.join(cfg.INPUT, "train.csv")
    cfg.TEST_FILE = os.path.join(cfg.INPUT, "test.csv")
    cfg.SUB_FILE = os.path.join(cfg.INPUT, "sample_submission.csv")
    cfg.RESULTS_FILE = os.path.join(cfg.COMPETITION, "output", "results.csv")

    cfg.TRAIN_FEAT_FILE = os.path.join(cfg.EXP, "train_feat")
    cfg.TEST_FEAT_FILE = os.path.join(cfg.EXP, "test_feat")

    cfg.FOLDS_FILE = "/kaggle/input/owner-s-best-model-ai-syosetu-ai/output/folds/folds_skf.npy" # kaggle ver

    # make dirs
    for d in [cfg.EXP]:
        os.makedirs(d, exist_ok=True)

    # set seed
    set_seed(cfg.seed)

    # set plot style
    pd.set_option('display.max_columns', 200)
    plt.rcParams['axes.facecolor'] = 'EEFFFE'

    return cfg

# Make output directories for run
def make_output_dir_run(cfg):
    cfg.OUTPUT = os.path.join(cfg.EXP, cfg.run_name)
    cfg.OUTPUT_MODEL = os.path.join(cfg.OUTPUT, 'model')
    cfg.OUTPUT_FIG = os.path.join(cfg.OUTPUT, 'fig')
    cfg.OUTPUT_PREDS = os.path.join(cfg.OUTPUT, 'preds')
    # make dirs
    for d in [cfg.OUTPUT, cfg.OUTPUT_MODEL, cfg.OUTPUT_FIG, cfg.OUTPUT_PREDS]:
        os.makedirs(d, exist_ok=True)
    return cfg

# Run Training
def run(cfg):
    if not cfg.tuning:
        print(fontstyle.apply(f'\n{cfg.run_name}', 'BLUE/BOLD/UNDERLINE'))
    if cfg.debug:
        print(fontstyle.apply('debug mode', 'BLUE/BOLD/UNDERLINE'))
    
    # load data
    df = pd.read_parquet(cfg.TRAIN_FEAT_FILE)
    test_df = pd.read_parquet(cfg.TEST_FEAT_FILE)
    folds = pd.Series(np.load(cfg.FOLDS_FILE))
    sub_df = pd.read_csv(cfg.SUB_FILE)

    if cfg.debug:
        df = df.sample(frac=0.01, random_state=cfg.seed).reset_index(drop=True)
        test_df = test_df.sample(frac=0.01, random_state=cfg.seed).reset_index(drop=True)

    # split feat / target
    base_cols = ['novel_id']
    useless_cols = []
    target_cols = ['global_point', 'fav_novel_cnt', 'all_point']
    not_use_cols_test = base_cols + useless_cols
    not_use_cols_train = not_use_cols_test + target_cols

    train_X = df.drop(not_use_cols_train, axis=1).values
    train_y = df[cfg.target_name].values
    test_X = test_df.drop(not_use_cols_test, axis=1).values

    if not cfg.tuning:
        print(f'train_X: {train_X.shape}\n',
              f'train_y: {train_y.shape}\n',
              f'test_X: {test_X.shape}')

    # training & inference
#     print(fontstyle.apply(f'======= {cv} =======', 'BLACK/BOLD'))

    # train & inference
    if cfg.model_name == 'lightgbm':
        oof, models, cv_score = fit_lightgbm(cfg, train_X, train_y, folds,
                              params=asdict(cfg.lgbm_param),
                              )
        # Predict Test
        pred_test = np.array([model.predict(test_X) for model in models])
        pred_test = np.mean(pred_test, axis=0)

    elif cfg.model_name == 'xgboost':
        oof, models, cv_score = fit_xgboost(cfg, train_X, train_y, folds,
                              params=asdict(cfg.xgb_param),
                              )
        # Predict Test
        xgb_test = xgb.DMatrix(test_X)
        pred_test = np.array([model.predict(xgb_test) for model in models])
        pred_test = np.mean(pred_test, axis=0)
    
    elif cfg.model_name == 'catboost':
        oof, models, cv_score = fit_catboost(cfg, train_X, train_y, folds,
                              params=asdict(cfg.catboost_param),
                              )
        # Predict Test
        pred_test = np.array([model.predict(test_X) for model in models])
        pred_test = np.mean(pred_test, axis=0)

    # post-processing
    pred_test = postprocess_target(pred_test) # for MSLE
    pred_test = process_target(pred_test) # for MSLE
    cfg.cv_score = cv_score

    # save results
    @dataclass
    class Save_Config:
        # name
        exp_name: str = cfg.exp_name
        run_name: str = cfg.run_name
        model_name: str = cfg.model_name
        # score
        cv_score: float = cfg.cv_score
        # params
        seed: int = cfg.seed
        num_fold: int = cfg.num_fold
#         learning_rate: float = cfg.lgbm_param.learning_rate
        # memo
        memo: str = ''

    if not cfg.debug:
#         save_cfg = Save_Config()
#         save_results(save_cfg, cfg.RESULTS_FILE)
        submit(cfg, sub_df, pred_test)

    # save params
    with open(os.path.join(cfg.OUTPUT, 'config.yml'), 'w') as f:
        yaml.dump(asdict(cfg), f)

# Submit
def submit(cfg, sub_df, pred_test):
    sub_path = os.path.join(cfg.OUTPUT_PREDS,
                            f"{cfg.user_name}_{cfg.exp_name}_{cfg.run_name}_cv{cfg.cv_score:.7f}.csv")
    sub_df[cfg.target_name] = pred_test
    sub_df.to_csv(sub_path, index=False)


@dataclass
class LGBM_Param:
    objective: str = 'regression'
    metrics: str = 'rmse'
    n_estimators: int = 100000
    learning_rate: float = .01
    importance_type: str = 'gain'
    random_state: int = 42

@dataclass
class XGB_Param:
    objective: str = 'reg:squarederror' 
    metrics: str = 'rmse' 
    n_estimators: int = 100000
    learning_rate: float = .01
    seed: int = 42

@dataclass
class CatBoost_Param:
    objective: str = 'RMSE'
    eval_metric: str = 'RMSE'
    learning_rate: float = .01
    iterations: int = 100000
    random_state: int = 42

@dataclass
class Config:
    # name
    user_name: str = 'kotrying'
    exp_name: str = 'exp001_stacking001'
    run_name: str = ''
    model_name: str = 'ligthtgbm'
    target_name: str = 'fav_novel_cnt' # 'global_point', 'fav_novel_cnt', 'all_point'
    group_name: str = 'novel_id'

    # mode
    debug: bool = False
    verbose: bool = True
    visualize: bool = True
    tuning: bool = False
    submit: bool = False

    # path
    COMPETITION: str = "/kaggle/input/narou"
    INPUT: str = ''
    EXP: str = ''
    OUTPUT: str = ''
    OUTPUT_MODEL: str = ''
    OUTPUT_FIG: str = ''
    OUTPUT_PREDS: str = ''
    TRAIN_FILE: str = ''
    TEST_FILE: str = ''
    SUB_FILE: str = ''
    RESULTS_FILE: str = ''

    TRAIN_FEAT_SKF_FILE: str = ''
    TEST_FEAT_SKF_FILE: str = ''

    FOLDS_SKF_FILE: str = ''

    # params
    seed: int = 42
    num_fold: int = 5
    loss_fnc: str = MSLE

    lgbm_param: LGBM_Param = LGBM_Param()
    xgb_param: XGB_Param = XGB_Param()
    catboost_param: CatBoost_Param = CatBoost_Param()

    # score
    cv_score: float = None


# setup exp config
cfg = Config()
cfg = setup(cfg)
# save dataframe
train_stack.to_parquet(cfg.TRAIN_FEAT_FILE)
test_stack.to_parquet(cfg.TEST_FEAT_FILE)


for target_col in ['fav_novel_cnt', 'all_point', 'global_point']:
    # setup exp config
    cfg = Config()
    cfg = setup(cfg)

    try:
        # setup run config
        model_name = "lightgbm"
        cfg.run_name = f'{model_name}_target_{target_col}'
        cfg = make_output_dir_run(cfg)

        # update run params
        cfg.model_name = model_name
        cfg.target_name = target_col
        
        print(cfg)

        # run
        %time run(cfg)

    except Exception as e:
        print(e)

########################################
# fav_novel_cnt: 2.0148861
# all_point: 2.3961705
# global_point: 2.4179221
########################################


for target_col in ['fav_novel_cnt', 'all_point', 'global_point']:
    # setup exp config
    cfg = Config()
    cfg = setup(cfg)

    try:
        # setup run config
        model_name = "xgboost"
        cfg.run_name = f'{model_name}_target_{target_col}'
        cfg = make_output_dir_run(cfg)

        # update run params
        cfg.model_name = model_name
        cfg.target_name = target_col
        
        print(cfg)

        # run
        %time run(cfg)

    except Exception as e:
        print(e)

########################################
# fav_novel_cnt: 2.0187207
# all_point: 2.3992401
# global_point: 2.4227381
########################################


for target_col in ['fav_novel_cnt', 'all_point', 'global_point']:
    # setup exp config
    cfg = Config()
    cfg = setup(cfg)

    try:
        # setup run config
        model_name = "catboost"
        cfg.run_name = f'{model_name}_target_{target_col}'
        cfg = make_output_dir_run(cfg)

        # update run params
        cfg.model_name = model_name
        cfg.target_name = target_col
        
        print(cfg)

        # run
        %time run(cfg)

    except Exception as e:
        print(e)

########################################
# fav_novel_cnt: 2.0131855
# all_point: 2.3900216
# global_point: 2.4157724
########################################


oofs_path_list = glob.glob(cfg.EXP + "/*/preds/oof_pred.npy")
oofs_fav_path_list = sorted([l for l in oofs_path_list if 'fav_novel_cnt' in l])
oofs_all_path_list = sorted([l for l in oofs_path_list if 'all_point' in l])
oofs_global_path_list = sorted([l for l in oofs_path_list if 'global_point' in l])

display(oofs_fav_path_list)
display(oofs_all_path_list)
display(oofs_global_path_list)


model_names = ["lightgbm", "xgboost", "catboost"]

oof_fav_df = pd.DataFrame()
for model_name in model_names:
    oof_fav_df[model_name] = np.load([path for path in oofs_fav_path_list if model_name in path][0])
display(oof_fav_df)

oof_all_df = pd.DataFrame()
for model_name in model_names:
    oof_all_df[model_name] = np.load([path for path in oofs_all_path_list if model_name in path][0])
display(oof_all_df)

oof_global_df = pd.DataFrame()
for model_name in model_names:
    oof_global_df[model_name] = np.load([path for path in oofs_global_path_list if model_name in path][0])
display(oof_global_df)

oof_df = pd.DataFrame()
oof_df[[f"{model_name}_fav" for model_name in model_names]] = oof_fav_df
oof_df[[f"{model_name}_all" for model_name in model_names]] = oof_all_df
oof_df[[f"{model_name}_global" for model_name in model_names]] = oof_global_df
display(oof_df)


preds_path_list = glob.glob(cfg.EXP + "/*/preds/*.csv")
preds_fav_path_list = [l for l in preds_path_list if 'fav_novel_cnt' in l]
preds_all_path_list = [l for l in preds_path_list if 'all_point' in l]
preds_global_path_list = [l for l in preds_path_list if 'global_point' in l]
display(preds_fav_path_list)
display(preds_all_path_list)
display(preds_global_path_list)


model_names = ["lightgbm", "xgboost", "catboost"]

pred_fav_df = pd.DataFrame()
for model_name in model_names:
    pred_fav_df[model_name] = pd.read_csv([path for path in preds_fav_path_list if model_name in path][0], usecols=["fav_novel_cnt"])
display(pred_fav_df)

pred_all_df = pd.DataFrame()
for model_name in model_names:
    pred_all_df[model_name] = pd.read_csv([path for path in preds_all_path_list if model_name in path][0], usecols=["all_point"])
display(pred_all_df)

pred_global_df = pd.DataFrame()
for model_name in model_names:
    pred_global_df[model_name] = pd.read_csv([path for path in preds_global_path_list if model_name in path][0], usecols=["global_point"])
display(pred_global_df)

pred_df = pd.DataFrame()
pred_df[[f"{model_name}_fav" for model_name in model_names]] = pred_fav_df
pred_df[[f"{model_name}_all" for model_name in model_names]] = pred_all_df
pred_df[[f"{model_name}_global" for model_name in model_names]] = pred_global_df
display(pred_df)


### 
train_act = train["global_point"].values
### 

# weight optimization
solution_objective = []
solution_weights = []

def func_for_minimize(weights):
    blend_list = []
    for col, w in zip(oof_global_df.columns, weights):
        blend_list.append(oof_global_df[col]*w)
    blend_pred = np.array(blend_list).sum(axis=0)
    return MSLE(train_act, blend_pred)

# optimize oof preds
set_seed(cfg.seed)
for i in tqdm(range(10)):
    starting_values = np.random.uniform(size=oof_global_df.shape[1])
    bounds = [(0,3)]*oof_global_df.shape[1]
    
    res = minimize(func_for_minimize, starting_values, method='Nelder-Mead', 
                bounds=bounds, options={'disp': False, 'maxiter': 10000})
    solution_objective.append(res['fun'])
    solution_weights.append(res['x'])

best_objective = np.min(solution_objective)
best_weight = solution_weights[np.argmin(solution_objective)]

print('\n Ensemble Score: {best_score:.7f}'.format(best_score=best_objective))
print('\n Best Weights: {weights:}'.format(weights=best_weight))


# optimized pred
blend_list = []
for col, w in zip(pred_global_df.columns, best_weight):
    blend_list.append(pred_global_df[col]*w)
blend_pred = np.array(blend_list).sum(axis=0)
blend_pred


# blend sub
sub_df = pd.DataFrame()
df_tmp = pd.read_csv(preds_path_list[0])
sub_df["novel_id"] = df_tmp["novel_id"]
sub_df["global_point"] = blend_pred
sub_df.to_csv(f'blend_preds_score{best_objective:.7f}.csv', index=False)
sub_df


# weight optimization
solution_objective = []
solution_weights = []

def func_for_minimize(weights):
    blend_list = []
    for col, w in zip(oof_df.columns, weights):
        blend_list.append(oof_df[col]*w)
    blend_pred = np.array(blend_list).sum(axis=0)
    return MSLE(train_act, blend_pred)

# optimize oof preds
set_seed(cfg.seed)
for i in tqdm(range(10)):
    starting_values = np.random.uniform(size=oof_df.shape[1])
    bounds = [(0,3)]*oof_df.shape[1]
    
    res = minimize(func_for_minimize, starting_values, method='Nelder-Mead', 
                bounds=bounds, options={'disp': False, 'maxiter': 10000})
    solution_objective.append(res['fun'])
    solution_weights.append(res['x'])

best_objective = np.min(solution_objective)
best_weight = solution_weights[np.argmin(solution_objective)]

print('\n Ensemble Score: {best_score:.7f}'.format(best_score=best_objective))
print('\n Best Weights: {weights:}'.format(weights=best_weight))


# optimized pred
blend_list = []
for col, w in zip(pred_df.columns, best_weight):
    blend_list.append(pred_df[col]*w)
blend_pred = np.array(blend_list).sum(axis=0)
blend_pred


# blend sub
sub_df = pd.DataFrame()
df_tmp = pd.read_csv(preds_path_list[0])
sub_df["novel_id"] = df_tmp["novel_id"]
sub_df["global_point"] = blend_pred
sub_df.to_csv(f'blend_preds_score{best_objective:.7f}.csv', index=False)
sub_df

