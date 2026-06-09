!pip install -qq pytabkit
!pip install -qq scikit-learn==1.5.2
!pip install autogluon.tabular


from autogluon.tabular import TabularDataset, TabularPredictor
from itertools import combinations
from lightgbm import LGBMClassifier
import numpy as np
import os
import optuna
import pickle
import pandas as pd
from pytabkit import TabM_D_Classifier
from pytabkit import RealMLP_HPO_Classifier, RealMLP_TD_Classifier as RealMLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import tqdm
from typing import Callable, Optional
import xgboost as xgb
import ydf


SEED = 2025

class CFG:
    train_csv = '/kaggle/input/playground-series-s5e12/train.csv'
    test_csv = '/kaggle/input/playground-series-s5e12/test.csv'
    orig_csv = '/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv'
    sample_submission_csv = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    target = 'diagnosed_diabetes'
    output = 'submission/%s.csv'
    ag_presets = 'best_quality'
    ag_time_limit = 4*3600

    kfold_n_splits = 5
    kfold_n_repeats = 2
    kfold_random_state = SEED

    optuna = False

    predict_with_original_dataset = False
    merge_original_dataset = False
    drop_inhomogeneous_features = False
    positive_weights = True

    limit1000 = False

    xgb_params7_optuna = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.008559367757686604,
        'max_depth': 5,
        'subsample': 0.93,
        'colsample_bytree': 0.19,
        'seed': SEED,
        'device': 'cuda',
        'grow_policy': 'lossguide',
        'alpha': 2.0,
        'lambda': 0.73,
        'min_child_weight': 5,
        'min_samples_split': 5,
        'max_bin': 512,
        'n_estimators': 20000,
        'early_stopping_rounds': 300
    }

    xgb_params2 = {
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'random_state': 42,
        'enable_categorical': True,
        'verbosity': 0,
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'n_jobs': -1,
        'learning_rate': 0.01,
        "device": "cuda",
        'lambda': 1.134330929497114,
        'alpha': 6.780537184218281,
        'colsample_bytree': 0.1007615968427798,
        'subsample': 0.7215917619002097,
        'max_depth': 4,
        'min_child_weight': 5,
        'n_estimators': 100_000,
        'early_stopping_rounds': 200,
    }

    lgb_params7_optuna = {
        'random_state': SEED, 
        'verbose': 1, 
        'n_estimators': 20_000, 
        'metric': 'AUC', 
        'objective': 'binary', 
        'learning_rate': 0.005230191195442491, 
        'max_depth': 3, 
        'min_child_samples': 128, 
        'subsample': 0.86, 
        'colsample_bytree': 0.48, 
        'num_leaves': 519, 
        'reg_alpha': 0.28, 
        'reg_lambda': 7.92, 
        'max_bin': 202, 
        'device': 'gpu'}

    ydf_params = {
        'random_seed': SEED,
        'max_depth': 5,
        'subsample': 0.93,
        'num_trees': 10_000,
        'l2_regularization': 0.2,
        'l1_regularization': 0.01,
        'early_stopping_num_trees_look_ahead': 300
    }

    tabm_params = {
        'device': 'cuda',
        'random_state': SEED,
        'verbosity': 2,
        'val_metric_name': '1-auc_ovr',
        'n_refit': 0,
        'n_epochs': 10,
        'batch_size': 'auto',
        'patience': 1,
        'allow_amp': False,
        'arch_type': 'tabm-mini',
        'tabm_k': 32,
        'gradient_clipping_norm': 1.0,
        'share_training_batches': False,
        'num_emb_type': 'pwl',
        'lr': 0.0007022171531739075,
        'weight_decay': 0.0,
        'n_blocks': 4,
        'd_block': 224,
        'dropout': 0.0,
        'd_embedding': 8,
        'num_emb_n_bins': 8,
    }

    realmlp_params = {
        'device': 'cuda:0', 
        'random_state': 0, 
        'n_cv': 1, 
        'n_refit': 0,
        'n_epochs': 128, 
        'batch_size': 4096, 
        'hidden_sizes': [256] * 3,
        'use_ls': False,  # for metrics like AUC / log-loss
        'lr': 0.04, 
        'verbosity': 2,
        
        #'n_cv': 8,
        #'hpo_space_name': 'tabarena-new', 
        #'use_caruana_ensembling': True, 
        #'n_hyperopt_steps': 50,
        'val_metric_name': '1-auc_ovr'
    }


class CV:
    def __init__(self,
                 k_fold_strategy,
                 df_train: pd.DataFrame,
                 target: str,
                 proba: bool=False,
                 fold_transform_train: Optional[Callable]=None,
                 fold_transform_test: Optional[Callable]=None
                 ) -> None:
        self.k_fold_strategy = k_fold_strategy
        self.df = df_train
        self.y = df_train[target]
        self.target = target
        self.proba = proba
        self.fold_transform_train = fold_transform_train if fold_transform_train else lambda inp, model:inp
        self.fold_transform_test = fold_transform_test if fold_transform_test else lambda inp, model:inp


    def fit(self, get_model: Callable, provide_val_data: bool=False) -> tuple[list, np.ndarray]:
        self.models = []
        oof_preds = np.zeros_like(self.y)
        for fold, (train_index, test_index) in enumerate(self.k_fold_strategy.split(self.y, self.y)):
            print("#" * 25)
            print(f"### Fold {fold + 1} ###")
            print("#" * 25)

            model = get_model()

            x_train = self.fold_transform_train(self.df.iloc[train_index].copy(), model)
            y_train = x_train.pop(self.target)

            x_test = self.fold_transform_test(self.df.iloc[test_index].copy(), model)
            y_test = x_test.pop(self.target)

            if provide_val_data:
                model.fit(x_train, y_train, x_test, y_test)
            else:
                model.fit(x_train, y_train)

            if self.proba:
                proba = model.predict_proba(x_test)
                if hasattr(proba, 'to_numpy'):
                    proba = proba.to_numpy()
                oof_preds[test_index] += proba[:, 1]
            else:
                oof_preds[test_index] += model.predict(x_test)
            self.models.append(model)
        return self.models, oof_preds


    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.target in df_test.columns:
            df_test = df_test.drop(self.target, axis=1)
        preds = 0
        for model in self.models:
            df_test_transformed = self.fold_transform_test(df_test.copy(), model)
            if self.proba:
                proba = model.predict_proba(df_test_transformed)
                if hasattr(proba, 'to_numpy'):
                    proba = proba.to_numpy()
                preds += proba[:, 1]
            else:
                preds += model.predict(df_test_transformed)
        return preds / len(self.models)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df
        self.target = target

    def columns_x_info(self) -> tuple[list, list]:
        """
        Print the dataframe columns info and split columns by their type
        Target y column is skipped, so only X columns are analysed
        """
        categoric = []
        numeric = []
        for col in self.df.columns:
            if col in [self.target, '_dataset']:
                continue

            if self.df[col].dtype == 'object':
                categoric.append(col)
                tp = 'CAT'
            else:
                numeric.append(col)
                tp = 'NUM'
            num_values = self.df[col].nunique()
            nans = self.df[col].isna().sum()
            print(f"[{tp}] {col} has {num_values} unique values and {nans} NANs")
        print("Categoric:", categoric )
        print("Numeric:", numeric )
        return categoric, numeric

    def label_encoder_categoric(self, categoric: list[str]) -> None:
        """
        Inplace transform all categoric columns to the columns with integer-only values
        """
        for col in categoric:
            self.df[col] = self.df[col].factorize()[0].astype('int32')

    def label_encoder_numeric(self, numeric: list[str], suffix: str = '_cat') -> list[str]:
        """
        Inplace duplicate numeric columns transforming them to categorical ones
        """
        new_cols = []
        for col in numeric:
            self.df[col + suffix] = self.df[col].factorize()[0].astype('int32')
            new_cols.append(col + suffix)
        return new_cols

    def create_categoric_pairs(self, categoric: list[str], inplace: bool=True) -> pd.DataFrame:
        """
        Combine previously label-encoded categorical columns into one new dataframe
        """
        pairs = list(combinations(categoric, 2))
        return self.create_pairs(pairs, inplace)

    def create_pairs(self, pairs: list[tuple[str,str]], inplace: bool=True, to_str: bool=False) -> pd.DataFrame:
        """
        Combine pairs of categorical columns
        """
        combined_df = {}
        for col1, col2 in pairs:
            combined = "_".join(sorted((col1, col2)))
            if to_str:
                combined_df[combined] = self.df[col1].astype(str) + '_' + self.df[col2].astype(str)
            else:
                combined_df[combined] = self.df[col1] * (1 + max(self.df[col2])) + self.df[col2]
        combined_df = pd.DataFrame(combined_df)

        if inplace:
            self.df = pd.concat([self.df, combined_df], axis=1)

        return combined_df

    def count_encoder_categoric(self, categoric: list[str], suffix: str='_cnt') -> list[str]:
        """
        Inplace addition of count-encoded categorical columns
        """
        new_cols = []
        for col in categoric:
            col_cnt = self.df.groupby(col)[self.target].count().astype('int32')
            col_cnt.name = col + suffix
            new_cols.append(col_cnt.name)
            self.df = self.df.merge(col_cnt, on=col, how='left')
        return new_cols

    def target_encoder_categoric(self, categoric: list[str], dataset: pd.DataFrame, suffix: str='_trg') -> list[str]:
        """
        Inplace addition of target-encoded categorical columns using the specified dataset for calculations
        """
        new_cols = []
        for col in categoric:
            col_trg = dataset.groupby(col)[self.target].mean()
            col_trg.name = col + suffix
            new_cols.append(col_trg.name)
            self.df = self.df.merge(col_trg, on=col, how='left')
        return new_cols
        
    def add_bin_features(self, numeric: list[str], q: int=5, suffix: str='_bin'):
        """
        Inplace addition of bins for the specified numeric features, thus transforming them to categoric
        """
        new_cols = []
        for col in numeric:
            self.df[col + suffix], _ = pd.qcut(self.df[col], q=q, labels=False, retbins=True, duplicates="drop")
            new_cols.append(col + suffix)
        return new_cols

    def add_digit_features(self, numeric: list[str], suffix: str='_dig'):
        """
        Inplace addition of digit features
        """
        new_cols = []
        for col in numeric:
            sp = self.df[col].astype(str).str.split('.', expand=True).fillna('')
            b = max(sp[0].astype(str).str.len())
            a = max(sp[1].astype(str).str.len()) if 1 in sp.columns else 0

            for k in range(1 - b, a + 1):
                new_col = f'{col}{suffix}{k}'
                self.df[new_col] = ((self.df[col] * 10**k) % 10).fillna(-1).astype("int8")
                new_cols.append(new_col)
        return new_cols

    def add_round_features(self, numeric: list[str], round: list=[-1, 0], suffix: str='_round'):
        """
        Inplace addition of rounded features
        """
        new_cols = []
        for col in numeric:
            for r in round:
                new_col = f'{col}{suffix}{r}'
                self.df[new_col] = self.df[col].round(r)
                new_cols.append(new_col)
        return new_cols

    def astype(self, columns, to_type):
        """
        Transform the columns' type
        """
        self.df[columns] = self.df[columns].astype(to_type)

    def drop(self, columns):
        """Drop the specified columns"""
        self.df = self.df.drop(columns, axis=1)


class XGBWrapper:
    def __init__(self, xgb_params: dict) -> None:
        self.model = xgb.XGBClassifier(**xgb_params)

    def fit(self, x, y, val_x, val_y):
        self.model.fit(x, y, eval_set=[(val_x, val_y)], verbose=300)

    def predict_proba(self, x):
        return self.model.predict_proba(x, iteration_range=(0, self.model.best_iteration + 1))


class LGBWrapper:
    def __init__(self, lgb_params: dict) -> None:
        self.model = LGBMClassifier(**lgb_params)

    def fit(self, x, y, val_x, val_y):
        self.model.fit(x, y, eval_set=[(val_x, val_y)])

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class YDFWrapper:
    def __init__(self, ydf_params: dict) -> None:
        self.base_model = ydf.GradientBoostedTreesLearner(label=CFG.target, **ydf_params)

    def fit(self, x, y, val_x, val_y):
        ds = x.copy()
        ds[CFG.target] = y.astype(int)
        valid = val_x.copy()
        valid[CFG.target] = val_y.astype(int)
        self.model = self.base_model.train(ds=ds, valid=valid)

    def predict_proba(self, x):
        p = self.model.predict(x).reshape((-1,1))
        y = np.hstack([1 - p, p])
        return y


class TABMWrapper:
    def __init__(self, tabm_params: dict) -> None:
        self.model = TabM_D_Classifier(**tabm_params)

    def fit(self, x, y, val_x, val_y):
        self.model.fit(x, y, val_x, val_y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class RealMLPWrapper:
    def __init__(self, mlp_params: dict) -> None:
        self.model = RealMLPClassifier(**mlp_params)

    def fit(self, x, y, val_x, val_y):
        try:
            self.model.fit(x, y, val_x, val_y)
        except:
            pass

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class AGWrapper:
    def __init__(self, ag_params: dict) -> None:
        self.model = TabularPredictor(label=CFG.target, problem_type='binary', eval_metric='roc_auc')

    def fit(self, x, y, val_x, val_y):
        train = x.copy()
        train[CFG.target] = y.astype(int)
        valid = val_x.copy()
        valid[CFG.target] = val_y.astype(int)
        self.model.fit(TabularDataset(train), presets='best_quality', time_limit=3000, dynamic_stacking=False, num_stack_levels=1)

    def predict_proba(self, x):
        return self.model.predict_proba(TabularDataset(x))


def load_combined_dataset(*datasets_csv, target: str) -> pd.DataFrame:
    dfs = []
    columns = None

    for n, csv in enumerate(datasets_csv):
        df = pd.read_csv(csv)
        if CFG.limit1000:
            df = df[:1000]
        if 'id' in df.columns:
            df = df.drop(columns='id')
        df['_dataset'] = n

        if columns is None:
            columns = set(df.columns)
        else:
            columns &= set(df.columns)

        dfs.append(df)

    for df in dfs:
        df.drop(list(set(df.columns) - columns - {target}), axis=1, inplace=True)

    return pd.concat(dfs, axis=0)


def split_combined_dataset(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [df[df['_dataset'] == n].drop('_dataset', axis=1) for n in sorted(df['_dataset'].unique())]


# Load the data
df = load_combined_dataset(CFG.train_csv, CFG.test_csv, CFG.orig_csv, target=CFG.target)
fe = FeatureEngineer(df, CFG.target)                                # Without FE: error
categoric, numeric = fe.columns_x_info()
fe.astype(numeric, 'int32')
fe.label_encoder_categoric(categoric)                               #

df_train, df_test, df_orig = split_combined_dataset(fe.df)


def adversarial_validation(df1, df2):
    # Below, we use simplified settings for speed since we are just testing the datasets here
    target = '_dataset'
    xgb_params = CFG.xgb_params7_optuna.copy()
    xgb_params['n_estimators'] = 1000
    k_fold_strategy = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=CFG.kfold_random_state)
    
    df1 = df1.drop(CFG.target, axis=1)
    df2 = df2.drop(CFG.target, axis=1)    
    df1[target] = 1.0
    df2[target] = 0.0
    df = pd.concat((df1, df2))
    cv = CV(k_fold_strategy,
            df,
            target=target,
            proba=True)
    _, oof_preds = cv.fit(get_model=lambda: XGBWrapper(xgb_params), provide_val_data=True)
    return roc_auc_score(df[target], oof_preds), cv

if True:
    # Just to save the time; uncomment for experiments!
    oof_score_train_test, cv_tt = adversarial_validation(df_train, df_test)
    oof_score_train_orig, cv_tro = adversarial_validation(df_train, df_orig)
    oof_score_test_orig, cv_teo = adversarial_validation(df_test, df_orig)
else:
    oof_score_train_test = 0.6195482986928571
    oof_score_train_orig = 0.9111857960214286
    oof_score_test_orig = 0.8700643765666667


print(f'TRAIN <--> TEST oof score = {oof_score_train_test}')
print(f'TRAIN <--> ORIG oof score = {oof_score_train_orig}')
print(f'TEST <--> ORIG oof score = {oof_score_test_orig}')


for name, model in [('train-test', cv_tt.models[0].model), ('train-orig', cv_tro.models[0].model), ('test-orig', cv_teo.models[0].model), ]:
    print(name)
    dd = pd.DataFrame((zip(df_train.columns, model.feature_importances_)), columns=['feature', 'importance'])
    display(dd.sort_values(by='importance', ascending=False)[:6])


if False:
    # Just to save the time; uncomment for experiments!
    df_train1 = df_train.drop(['physical_activity_minutes_per_week', 'triglycerides', 'cholesterol_total', 'alcohol_consumption_per_week', 'ldl_cholesterol'], axis=1)
    df_test1 = df_test.drop(['physical_activity_minutes_per_week', 'triglycerides', 'cholesterol_total', 'alcohol_consumption_per_week', 'ldl_cholesterol'], axis=1)
    df_orig1 = df_orig.drop(['physical_activity_minutes_per_week', 'triglycerides', 'cholesterol_total', 'alcohol_consumption_per_week', 'ldl_cholesterol'], axis=1)
    
    oof_score_train_test, cv_tt = adversarial_validation(df_train1, df_test1)
    oof_score_train_orig, cv_tro = adversarial_validation(df_train1, df_orig1)
    oof_score_test_orig, cv_teo = adversarial_validation(df_test1, df_orig1)
else:
    oof_score_train_test = 0.5457904335666667
    oof_score_train_orig = 0.8344544324357142
    oof_score_test_orig = 0.8130414084833333


print(f'TRAIN <--> TEST oof score = {oof_score_train_test}')
print(f'TRAIN <--> ORIG oof score = {oof_score_train_orig}')
print(f'TEST <--> ORIG oof score = {oof_score_test_orig}')


if CFG.predict_with_original_dataset:
    # Here, k-fold strategy is the same as for 'main' training below, because we need high accuracy
    k_fold_strategy = RepeatedStratifiedKFold(n_splits=CFG.kfold_n_splits, n_repeats=CFG.kfold_n_repeats, random_state=CFG.kfold_random_state)
    
    orig_based_cv = CV(k_fold_strategy,
                      df_orig,
                      target=CFG.target,
                      proba=True)
    _, orig_based_oof_preds = orig_based_cv.fit(get_model=lambda: XGBWrapper(CFG.xgb_params7_optuna), provide_val_data=True)



if CFG.predict_with_original_dataset:
    orig_oof_score = roc_auc_score(df_orig[CFG.target], orig_based_oof_preds)
    print(f'ORIG OOF score = {orig_oof_score}')
    
    df_train_orig_pred = orig_based_cv.predict(df_train)
    df_test_orig_pred = orig_based_cv.predict(df_test)


cat1, cat2 = [], []

# Load the data
df = load_combined_dataset(CFG.train_csv, CFG.test_csv, CFG.orig_csv, target=CFG.target)

# Now do different processing
fe = FeatureEngineer(df, CFG.target)                                # Without FE: error

if CFG.drop_inhomogeneous_features:
    fe.drop(['physical_activity_minutes_per_week', 'triglycerides', 'cholesterol_total'])

categoric, numeric = fe.columns_x_info()

categoric += fe.add_bin_features(numeric)
categoric += fe.add_digit_features(numeric)
fe.add_round_features(numeric)

fe.label_encoder_categoric(categoric)                               #

cat1 = fe.label_encoder_numeric(numeric)                            #
fe.astype(numeric, 'int32')
categoric += cat1

#combined_df = fe.create_categoric_pairs(categoric, inplace=True)    # Uncomment for experimenting
#cat2 = list(combined_df.columns)
#categoric += cat2

fe.count_encoder_categoric(categoric)                               #
df_train, df_test, df_orig = split_combined_dataset(fe.df)

if CFG.predict_with_original_dataset:
    # Insert new column based on the original dataset
    df_train['orig_based_target'] = df_train_orig_pred
    df_test['orig_based_target'] = df_test_orig_pred
    df_orig['orig_based_target'] = orig_based_oof_preds


k_fold_strategy = RepeatedStratifiedKFold(n_splits=CFG.kfold_n_splits, n_repeats=CFG.kfold_n_repeats, random_state=CFG.kfold_random_state)

def fold_transform_test(df, model) -> pd.DataFrame:
    for col in cat1 + cat2:
        df[col] = model.te[col].transform(df[col].to_frame()).astype('float32')
    return df


def fold_transform_train(df, model) -> pd.DataFrame:
    if CFG.merge_original_dataset:
        df = pd.concat((df, df_orig), axis=0, ignore_index=True)
    model.te = {}
    for col in cat1 + cat2:
        model.te[col] = TargetEncoder(target_type='continuous', smooth=1, cv=10, shuffle=True, random_state=42)
        #model.te[col] = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
        df[col] = model.te[col].fit_transform(df[col].to_frame(), df[CFG.target]).astype('float32')
    return df


def xgb_objective(trial):
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": trial.suggest_float('learning_rate', 1e-3, 1e-1, step=None, log=True),
        "max_depth": trial.suggest_int('max_depth', 0, 10, step=1, log=False),
        "subsample": trial.suggest_float('subsample', 0, 1, step=0.01, log=False),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0, 1, step=0.01, log=False),
        "seed": SEED,
        "device": "cuda",
        "grow_policy": "lossguide",
        "alpha": trial.suggest_float('alpha', 1, 10, step=0.1, log=False),
        "lambda": trial.suggest_float('lambda', 0, 1, step=0.01, log=False),
        "min_child_weight": trial.suggest_int('max_depth', 2, 10, step=1, log=False),
        "min_samples_split": trial.suggest_int('max_depth', 2, 10, step=1, log=False),
        "max_bin": 512,

        'n_estimators': 20_000,
        'early_stopping_rounds': 300
    }

    cv = CV(k_fold_strategy,
        df_train,
        target=CFG.target,
        proba=True,
        fold_transform_train=fold_transform_train,
        fold_transform_test=fold_transform_test)
    _, oof_preds = cv.fit(get_model=lambda: XGBWrapper(xgb_params), provide_val_data=True)
    oof_score = roc_auc_score(df_optuna[CFG.target], oof_preds)

    # Log optuna's results
    with open('optuna_xgb.txt', 'a') as f:
        f.write(f'{oof_score:.6f} => {xgb_params}\n')

    return oof_score


def lgb_objective(trial):
    lgb_params = {
        'random_state': CFG.lgb_params['random_state'],
        'verbose': -1,
        'n_estimators': 20000,#trial.suggest_int('iterations', 10, 10_000, log=True),
        'metric': 'AUC',
        'objective': 'binary',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1, step=None, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 1000, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1, step=0.01),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1, step=0.01),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1, step=0.01),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, step=0.01),
        'max_bin': trial.suggest_int('max_bin', 10, 255, log=True),
        'device': 'gpu',
    }

    cv = CV(k_fold_strategy,
        df_optuna,
        target=CFG.target,
        proba=True,
        fold_transform_train=fold_transform_train,
        fold_transform_test=fold_transform_test)
    _, oof_preds = cv.fit(get_model=lambda: LGBWrapper2(lgb_params), provide_val_data=True)
    oof_score = roc_auc_score(df_optuna[CFG.target], oof_preds)

    with open('optuna_lgb.txt', 'a') as f:
        f.write(f'{oof_score:.6f} => {lgb_params}\n')

    return oof_score
    

if CFG.optuna:
    study = optuna.create_study(direction='maximize')
    study.optimize(xgb_objective, n_trials=200)
    print(study.best_params)

    with open('best_params_xgb_optuna.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)
    xgb_params = study.best_params
else:
    xgb_params = CFG.xgb_params7_optuna
    lgb_params = CFG.lgb_params7_optuna    
    ydf_params = CFG.ydf_params
    tabm_params = CFG.tabm_params


def train_model(get_model: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv = CV(k_fold_strategy,
            df_train,
            target=CFG.target,
            proba=True,
            fold_transform_train=fold_transform_train,
            fold_transform_test=fold_transform_test)
    
    _, oof_preds = cv.fit(get_model=get_model, provide_val_data=True)
    oof_score = roc_auc_score(df_train[CFG.target], oof_preds)
    
    train_preds = cv.predict(df_train)
    train_score = roc_auc_score(df_train[CFG.target], train_preds)
    
    test_preds = cv.predict(df_test)
    
    # TRAIN score is an overfitted one, so OOF score is correct estimate, while TRAIN should be higher if everything is ok
    print(f'OOF score = {oof_score} / TRAIN score = {train_score}')

    return test_preds, train_preds, oof_preds, oof_score, train_score


predictors, test_preds, train_preds, oof_preds = [], [], [], []


# LGB seems worse, so don't include it
if False:
    test_preds_lgb, train_preds_lgb, oof_preds_lgb, oof_score_lgb, train_score_lgb = train_model(get_model=lambda: LGBWrapper(lgb_params))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_lgb.npy', test_preds_lgb)
    np.save('oof_preds_lgb.npy', oof_preds_lgb)
else:
    test_preds_lgb = np.load('/kaggle/input/pgs512-predictions/test_preds_lgb.npy')
    train_preds_lgb = np.zeros((len(df_train)))
    oof_preds_lgb = np.load('/kaggle/input/pgs512-predictions/oof_preds_lgb.npy')
    oof_score_lgb = roc_auc_score(df_train[CFG.target], oof_preds_lgb)
    train_score_lgb = 0
    
test_preds.append(test_preds_lgb)
train_preds.append(train_preds_lgb)
oof_preds.append(oof_preds_lgb)
predictors.append('LGB')


print(f'OOF score = {oof_score_lgb} / TRAIN score = {train_score_lgb}')


# XGB is default
if False:
    test_preds_xgb_drop_inh, train_preds_xgb_drop_inh, oof_preds_xgb_drop_inh, oof_score_xgb_drop_inh, train_score_xgb_drop_inh = train_model(get_model=lambda: XGBWrapper(xgb_params))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_xgb_drop_inh.npy', test_preds_xgb_drop_inh)
    np.save('oof_preds_xgb_drop_inh.npy', oof_preds_xgb_drop_inh)
elif False:
    test_preds_xgb_drop_inh = np.load('/kaggle/input/pgs512-predictions/test_preds_xgb_drop_inh.npy')
    train_preds_xgb_drop_inh = np.zeros((len(df_train)))
    oof_preds_xgb_drop_inh = np.load('/kaggle/input/pgs512-predictions/oof_preds_xgb_drop_inh.npy')
    oof_score_xgb_drop_inh = roc_auc_score(df_train[CFG.target], oof_preds_xgb_drop_inh)
    train_score_xgb_drop_inh = 0

test_preds_xgb_noorig = np.load('/kaggle/input/pgs512-predictions/test_preds_xgb_noorig.npy')
train_preds_xgb_noorig = np.zeros((len(df_train)))
oof_preds_xgb_noorig = np.load('/kaggle/input/pgs512-predictions/oof_preds_xgb_noorig.npy')
oof_score_xgb_noorig = roc_auc_score(df_train[CFG.target], oof_preds_xgb_noorig)
train_score_xgb_noorig = 0

test_preds_xgb_orig = np.load('/kaggle/input/pgs512-predictions/test_preds_xgb.npy')
train_preds_xgb_orig = np.zeros((len(df_train)))
oof_preds_xgb_orig = np.load('/kaggle/input/pgs512-predictions/oof_preds_xgb.npy')
oof_score_xgb_orig = roc_auc_score(df_train[CFG.target], oof_preds_xgb_orig)
train_score_xgb_orig = 0

test_preds.append(test_preds_xgb_orig)
train_preds.append(train_preds_xgb_orig)
oof_preds.append(oof_preds_xgb_orig)
predictors.append('XGB')

test_preds.append(test_preds_xgb_noorig)
train_preds.append(train_preds_xgb_noorig)
oof_preds.append(oof_preds_xgb_noorig)
predictors.append('XGB_noorig')

#test_preds.append(test_preds_xgb_drop_inh)
#train_preds.append(train_preds_xgb_drop_inh)
#oof_preds.append(oof_preds_xgb_drop_inh)
#predictors.append('XGB_drop_inh')


print(f'OOF score = {oof_score_xgb_orig} / TRAIN score = {train_score_xgb_orig}')
# OOF score = 0.7317553554654964

print(f'OOF score noorig = {oof_score_xgb_noorig} / TRAIN score = {train_score_xgb_noorig}')

#print(f'OOF score drop_inh = {oof_score_xgb_drop_inh} / TRAIN score = {train_score_xgb_drop_inh}')


if False:
    # XGB2
    if False:
        test_preds_xgb2, train_preds_xgb2, oof_preds_xgb2, oof_score_xgb2, train_score_xgb2 = train_model(get_model=lambda: XGBWrapper(CFG.xgb_params2))
        
        # Save individual predictions for future use in HC
        np.save('test_preds_xgb2.npy', test_preds_xgb2)
        np.save('oof_preds_xgb2.npy', oof_preds_xgb2)
    else:
        test_preds_xgb2 = np.load('/kaggle/input/pgs512-predictions/test_preds_xgb2.npy')
        train_preds_xgb2 = np.zeros((len(df_train)))
        oof_preds_xgb2 = np.load('/kaggle/input/pgs512-predictions/oof_preds_xgb2.npy')
        oof_score_xgb2 = roc_auc_score(df_train[CFG.target], oof_preds_xgb2)
        train_score_xgb2 = 0
    
    test_preds.append(test_preds_xgb2)
    train_preds.append(train_preds_xgb2)
    oof_preds.append(oof_preds_xgb2)
    predictors.append('XGB2')

    print(f'OOF score XGB2 = {oof_score_xgb2} / TRAIN score = {train_score_xgb2}')


# XGB2
class XGBWrapper2:
    def __init__(self, xgb_params: dict) -> None:
        self.base_model = xgb.XGBClassifier(**xgb_params)

    def fit(self, x, y, val_x, val_y):
        X_train = DMatrix(X_train, label=y_train, enable_categorical=True, weight=w_trn)
        X_val = DMatrix(X_val, label=y_val, enable_categorical=True, weight=w_val)
        
        self.model = xgb.train(
            params=self.base_model.get_params(),
            dtrain=X_train,
            evals=[(X_val, "valid")],
            num_boost_round=100_000,
            early_stopping_rounds=200,
            verbose_eval=False
        )
        #self.model.fit(x, y, eval_set=[(val_x, val_y)], verbose=300)

    def predict_proba(self, x):
        X_test = DMatrix(x, enable_categorical=True)
        #return self.model.predict_proba(x, iteration_range=(0, self.model.best_iteration + 1))

if False: 
    if True:
        test_preds_xgb3, train_preds_xgb3, oof_preds_xgb3, oof_score_xgb3, train_score_xgb3 = \
        train_model(get_model=lambda: XGBWrapper2(CFG.xgb_params2))
        
        # Save individual predictions for future use in HC
        np.save('test_preds_xgb3.npy', test_preds_xgb3)
        np.save('oof_preds_xgb3.npy', oof_preds_xgb3)
    else:
        test_preds_xgb3 = np.load('/kaggle/input/pgs512-predictions/test_preds_xgb3.npy')
        train_preds_xgb3 = np.zeros((len(df_train)))
        oof_preds_xgb3 = np.load('/kaggle/input/pgs512-predictions/oof_preds_xgb3.npy')
        oof_score_xgb3 = roc_auc_score(df_train[CFG.target], oof_preds_xgb3)
        train_score_xgb3 = 0
    
    test_preds.append(test_preds_xgb3)
    train_preds.append(train_preds_xgb3)
    oof_preds.append(oof_preds_xgb3)
    predictors.append('xgb3')


if False:
    test_preds_ydf, train_preds_ydf, oof_preds_ydf, oof_score_ydf, train_score_ydf = train_model(get_model=lambda: YDFWrapper(ydf_params))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_ydf.npy', test_preds_ydf)
    np.save('oof_preds_ydf.npy', oof_preds_ydf)
else:
    test_preds_ydf = np.load('/kaggle/input/pgs512-predictions/test_preds_ydf.npy')
    train_preds_ydf = np.zeros((len(df_train)))
    oof_preds_ydf = np.load('/kaggle/input/pgs512-predictions/oof_preds_ydf.npy')
    oof_score_ydf = roc_auc_score(df_train[CFG.target], oof_preds_ydf)
    train_score_ydf = 0
    
test_preds.append(test_preds_ydf)
train_preds.append(train_preds_ydf)
oof_preds.append(oof_preds_ydf)
predictors.append('YDF')


print(f'OOF score LGB = {oof_score_lgb} / TRAIN score = {train_score_lgb}')


if False:
    test_preds_tabm, train_preds_tabm, oof_preds_tabm, oof_score_tabm, train_score_tabm = train_model(get_model=lambda: TABMWrapper(tabm_params))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_tabm.npy', test_preds_tabm)
    np.save('oof_preds_tabm.npy', oof_preds_tabm)
else:
    test_preds_tabm = np.load('/kaggle/input/pgs512-predictions/test_preds_tabm.npy')
    train_preds_tabm = np.zeros((len(df_train)))
    oof_preds_tabm = np.load('/kaggle/input/pgs512-predictions/oof_preds_tabm.npy')
    oof_score_tabm = roc_auc_score(df_train[CFG.target], oof_preds_tabm)
    train_score_tabm = 0

test_preds.append(test_preds_tabm)
train_preds.append(train_preds_tabm)
oof_preds.append(oof_preds_tabm)
predictors.append('TABM')


print(f'OOF score TABM = {oof_score_tabm} / TRAIN score = {train_score_tabm}')


if False:
    test_preds_mlp, train_preds_mlp, oof_preds_mlp, oof_score_mlp, train_score_mlp = train_model(
        get_model=lambda: RealMLPWrapper(CFG.realmlp_params))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_mlp.npy', test_preds_mlp)
    np.save('oof_preds_mlp.npy', oof_preds_mlp)
else:
    test_preds_mlp = np.load('/kaggle/input/pgs512-predictions/test_preds_mlp2.npy')
    train_preds_mlp = np.zeros((len(df_train)))
    oof_preds_mlp = np.load('/kaggle/input/pgs512-predictions/oof_preds_mlp2.npy')
    oof_score_mlp = roc_auc_score(df_train[CFG.target], oof_preds_mlp)
    train_score_mlp = 0

test_preds.append(test_preds_mlp)
train_preds.append(train_preds_mlp)
oof_preds.append(oof_preds_mlp)
predictors.append('RealMLP')


print(f'OOF score RealMLP = {oof_score_mlp} / TRAIN score = {train_score_mlp}')


if False:
    test_preds_ag, train_preds_ag, oof_preds_ag, oof_score_ag, train_score_ag = train_model(
        get_model=lambda: AGWrapper(None))
    
    # Save individual predictions for future use in HC
    np.save('test_preds_ag3000.npy', test_preds_ag)
    np.save('oof_preds_ag3000.npy', oof_preds_ag)
else:
    test_preds_ag = np.load('/kaggle/input/pgs512-predictions/test_preds_ag.npy')
    train_preds_ag = np.zeros((len(df_train)))
    oof_preds_ag = np.load('/kaggle/input/pgs512-predictions/oof_preds_ag.npy')
    oof_score_ag = roc_auc_score(df_train[CFG.target], oof_preds_ag)
    train_score_ag = 0

    test_preds_ag3 = np.load('/kaggle/input/pgs512-predictions/test_preds_ag3000.npy')
    train_preds_ag3 = np.zeros((len(df_train)))
    oof_preds_ag3 = np.load('/kaggle/input/pgs512-predictions/oof_preds_ag3000.npy')
    oof_score_ag3 = roc_auc_score(df_train[CFG.target], oof_preds_ag3)
    train_score_ag3 = 0

test_preds.append(test_preds_ag)
train_preds.append(train_preds_ag)
oof_preds.append(oof_preds_ag)
predictors.append('AutoGluon1200')

test_preds.append(test_preds_ag3)
train_preds.append(train_preds_ag3)
oof_preds.append(oof_preds_ag3)
predictors.append('AutoGluon3000')


print(f'OOF score AG1200 = {oof_score_ag} / TRAIN score = {train_score_ag}')
print(f'OOF score AG3000 = {oof_score_ag3} / TRAIN score = {train_score_ag3}')


test_preds_xgb2 = pd.read_csv('/kaggle/input/pgs512-predictions/test_preds_xgb2.csv').to_numpy().reshape(-1)
#train_preds_xgb2 = np.zeros((len(df_train)))
#oof_preds_xgb2 = pd.read_csv('/kaggle/input/pgs512-predictions/oof_preds_xgb2.csv').to_numpy().reshape(-1)
#oof_score_xgb2 = roc_auc_score(df_train[CFG.target], oof_preds_xgb2)
#train_score_xgb2 = 0

#test_preds.append(test_preds_xgb2)
#train_preds.append(train_preds_xgb2)
#oof_preds.append(oof_preds_xgb2)
#predictors.append('XGB2')

#print(f'OOF score XGB2 = {oof_score_xgb2} / TRAIN score = {train_score_xgb2}')


test_preds_ens = sum(test_preds) / len(test_preds)
oof_preds_ens = sum(oof_preds) / len(oof_preds)
train_preds_ens = sum(train_preds) / len(train_preds)

oof_score = roc_auc_score(df_train[CFG.target], oof_preds_ens)
train_score = roc_auc_score(df_train[CFG.target], train_preds_ens)

# TRAIN score is an overfitted one, so OOF score is correct estimate, while TRAIN should be higher if everything is ok
print(f'AVG: OOF score = {oof_score} / TRAIN score = {train_score}')


def hc(oof_preds, positive_weights=True, weights_choice=(-0.5, 0.5, -0.01, 0.01), initial_weights=None, n_epochs=20):
    oof_preds_np = np.array(oof_preds).T
    
    weights = np.ones((len(test_preds), 1)) / len(test_preds) if initial_weights is None else initial_weights
    score = 0
    
    for _ in tqdm.tqdm(range(n_epochs)):
        for p, _ in enumerate(oof_preds):
            for w in weights_choice:
                # Create new weights
                test_w = weights.copy()
                test_w[p] += w
                if positive_weights:
                    test_w = np.clip(test_w, 0, 100)
    
                # Make a prediction on OOF data
                oof_test_pred = oof_preds_np @ test_w
                oof_test_score = roc_auc_score(df_train[CFG.target], oof_test_pred)
                
                if oof_test_score > score:
                    weights, score = test_w, oof_test_score
    return weights, score


weights1, score1 = hc(oof_preds, positive_weights=CFG.positive_weights, weights_choice=(-0.5, 0.5, -0.01, 0.01))
print(f'Step1: {score1}')

weights2, score2 = hc(oof_preds, positive_weights=CFG.positive_weights, weights_choice=(-0.5, 0.5, -0.001, 0.001), initial_weights=weights1)
print(f'Step2: {score2}')

weights = weights2

test_preds_hc = np.array(test_preds).T @ weights + test_preds_xgb2.reshape((-1, 1)) * 16.4


predictors


print('Resulting weights of different classifiers')
print(dict(zip(predictors, list(weights.reshape(-1)/sum(weights)))))


# Save the results
fname = 'submission.csv'
submission = pd.read_csv(CFG.sample_submission_csv)
submission[CFG.target] = test_preds_hc
#submission[CFG.target] = test_preds_ens
submission.to_csv(fname, index=False)
submission.head()

