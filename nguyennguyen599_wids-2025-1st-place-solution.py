!pip install fastparquet -q
!pip install feature-engine -q
!pip install umap -q
!pip install umap-learn -q
!pip install lightning -q
!pip install pytabkit -q
!pip install shap -q
# !pip install optuna -q
# !pip install openpyxl -q
# !pip install missingno -q
# !pip install torch==2.5.1 torchvision -q
# !pip install torch_geometric -q
# !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html -q


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

from copy import deepcopy

from matplotlib import pyplot as plt
import seaborn as sns
import time
import missingno as msno
import random
import multiprocessing
from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import f1_score as f1_score_calc, roc_curve, roc_auc_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix

from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor, RealMLP_TD_Classifier, TabM_D_Classifier, Ensemble_TD_Regressor
import optuna

import torch
import networkx as nx

pd.options.display.max_columns = 60

import shap

SEED = 42
torch.manual_seed(42)
random.seed(42)
np.random.seed(SEED)
shap.initjs()


train_q = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_c = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
test_q = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_c = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
sample = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
# Use parquet format for faster load data
train_fcm = pd.read_parquet("/kaggle/input/data-parquet/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.parquet", engine="fastparquet")
test_fcm = pd.read_parquet("/kaggle/input/data-parquet/TEST_FUNCTIONAL_CONNECTOME_MATRICES.parquet", engine="fastparquet")

df = pd.merge(train_q, train_c, on="participant_id", how="left")
test = pd.merge(test_q, test_c, on="participant_id", how="left")

labels = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")

df['weight'] = 1
test['weight'] = 1
df = pd.merge(df, labels, on="participant_id", how="left")

# weights equal 2 for ADHD_Outcome == 1 and Sex_F == 1
df.loc[(df['ADHD_Outcome']==1) & (df['Sex_F']==1), "weight"] = 2

# backup
df_ori = df.copy()
test_ori = test.copy()


df


test


# fast check target
sns.countplot(data=df, x='ADHD_Outcome', hue='Sex_F')
plt.title('ADHD Outcome Count by Sex')
plt.xlabel('ADHD Outcome (0=Other/None, 1=ADHD)')
plt.ylabel('Count')
plt.legend(title='Sex (0=Male, 1=Female)')
plt.show()



msno.matrix(df.drop(['ADHD_Outcome', 'Sex_F'], axis=1))


msno.matrix(test)


df = df_ori.copy()
test = test_ori.copy()

# Fill nan
imputer = IterativeImputer(estimator=LassoCV(random_state=42, n_jobs=8, cv=8), max_iter=5, random_state=42, verbose=2)

cols_to_impute = df.columns.drop(['participant_id', "ADHD_Outcome", 'Sex_F'])

combined = pd.concat([df,test],axis=0,ignore_index=True)

# Fit and transform the data
imputed_data = imputer.fit_transform(df.drop(['participant_id', "ADHD_Outcome", 'Sex_F'], axis=1))

print('START IMPUTER')
df[cols_to_impute] = imputed_data[:len(df), :]
test[cols_to_impute] = imputer.transform(test.drop(['participant_id'], axis=1))

df_c = df.copy()
test_c = test.copy()


train_fcm


# Function to convert a single row from the dataframe into a matrix
def reshape_to_matrix(row, participant_id=None):
    # return matrix
    # Drop participant_id if it exists
    if participant_id is not None and participant_id in row.index:
        row = row.drop(participant_id)
    
    # Extract all row and column indices from column names
    cols = row.index
    row_indices = set()
    col_indices = set()
    
    for col in cols:
        if '_' in col:
            try:
                parts = col.split('_')
                row_idx = int(parts[0].replace('throw', ''))
                col_idx = int(parts[1].replace('thcolumn', ''))
                row_indices.add(row_idx)
                col_indices.add(col_idx)
            except:
                continue
    
    max_row = max(row_indices) + 1 if row_indices else 0
    max_col = max(col_indices) + 1 if col_indices else 0
    
    # If dimensions don't match, make square by using the larger dimension
    matrix_size = max(max_row, max_col)
    matrix = np.zeros((matrix_size, matrix_size))

    # Fill the matrix with values
    for col in cols:
        if '_' in col:
            try:
                parts = col.split('_')
                row_idx = int(parts[0].replace('throw', ''))
                col_idx = int(parts[1].replace('thcolumn', ''))
                if row_idx < matrix_size and col_idx < matrix_size:
                    matrix[row_idx, col_idx] = row[col]
            except:
                continue
    
    return participant_id, matrix

def sym_matrix(corr):
    corr_full = corr + corr.T - np.diag(np.diag(corr))
    np.fill_diagonal(corr_full, 1)
    return corr_full

def fcm2ft(df_fcm, do_sym=True):
    results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(reshape_to_matrix)(row.copy(deep=False), row['participant_id']) 
            for pid, row in tqdm(df_fcm.iterrows(), total=len(df_fcm))
        )
    
    all_matrices_fixed = {}
    for participant, matrix in results:
        if do_sym:
            all_matrices_fixed[participant] = sym_matrix(matrix)
        else:
            all_matrices_fixed[participant] = matrix
    
    return all_matrices_fixed

all_matrices_train = fcm2ft(train_fcm)
all_matrices_test = fcm2ft(test_fcm)


def plot_cases(mat1, mat2, label1=None, label2=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    vmin = min(mat1.min(), mat2.min())
    vmax = max(mat1.max(), mat2.max())
    
    cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)
    sns.heatmap(mat1, ax=axes[0], annot=False, cmap=cmap, square=True, cbar=False, vmin=vmin, vmax=vmax)
    if label1!=None:
        axes[0].set_title(f'Case {label1} - ADHD: {df["ADHD_Outcome"][label1]} - Sex_F: {df["Sex_F"][label1]}')
    else:
        axes[0].set_title('Matrix1')
        
    sns.heatmap(mat2, ax=axes[1], annot=False, cmap=cmap, square=True, cbar=False, vmin=vmin, vmax=vmax)
    if label2!=None:
        axes[1].set_title(f'Case {label2} - ADHD: {df["ADHD_Outcome"][label2]} - Sex_F: {df["Sex_F"][label2]}')
    else:
        axes[1].set_title('Maxtrix test')
    
    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.85])
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    
train_idx = 0
test_idx = 0
plot_cases(list(all_matrices_train.values())[train_idx],
         list(all_matrices_test.values())[test_idx],
          label1=train_idx,
          label2=None
          )

print('Case ADHD vs non ADHD in Female')
train_idx = df[(df['Sex_F']==1) & (df['ADHD_Outcome']==1)].index[0]
test_idx = df[(df['Sex_F']==1) & (df['ADHD_Outcome']==0)].index[0]

plot_cases(list(all_matrices_train.values())[train_idx],
         list(all_matrices_train.values())[test_idx],
          label1=train_idx,
          label2=test_idx
          )


# merge with functional connectome matrices
df = df_c.copy()
test = test_c.copy()

df = pd.merge(df, train_fcm, on="participant_id", how="left")
test = pd.merge(test, test_fcm, on="participant_id", how="left")


# Function util
def save_submit(y_pred, save_to='/kaggle/working/submission.csv', col=''):
    try:
        sample = pd.read_csv('/kaggle/working/submission.csv')
    except:
        sample = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
    sample[col] = y_pred
    sample.to_csv(save_to,index=False)
    return sample

def ACC(y_true, y_pred, weight):
    return accuracy_score(y_true, y_pred)

def F1(y_true, y_pred, threshold=0.5, weight=None):
    x = f1_score_calc(y_true, (y_pred > threshold).astype(int), sample_weight=weight)
    # print(x)
    return x

def balanced_kfold_split(df_xx, df_x, n_splits=1, group_column='ADHD_Outcome', df_buff=None, seed=42):
    """
    Custom balanced k-fold split for a dataframe ensuring balanced distribution for each groups
    Args:
        df_xx: DataFrame to be split
        df_x: Origin DataFrame containing the target variable
        n_splits: Number of splits
        group_column: Column name to be used for grouping
        df_buff: Buffer DataFrame to be used in all folds
    """
    if n_splits==1:
        return [[df_xx.index, df_xx.index]]

    # Align group column from df_x to df
    df = df_xx.copy()
    df[group_column] = df_x[group_column].values  # Ensure correct order of target in df
    buffer_indices = None
    if df_buff is not None:
        buffer_indices = df_buff.index.tolist()

    # Initialize folds
    folds = [[] for _ in range(n_splits)]
    groups = df[group_column].unique()

    for group in groups:
        group_indices = df[df[group_column] == group].index.tolist()
        np.random.seed(seed)
        np.random.shuffle(group_indices)  # Shuffle indices for the group

        # Distribute the group indices across folds
        for i, idx in enumerate(group_indices):
            folds[i % n_splits].append(idx)

    # Create train and validation indices for each fold
    splits = []
    for i in range(n_splits):
        val_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        if buffer_indices:
            buffer_indices_tmp = [idx for idx in buffer_indices if idx not in train_indices]
            print('buff', len(buffer_indices_tmp))
            train_indices = np.array(list(train_indices)+buffer_indices_tmp)
        splits.append((train_indices, val_indices))
    np.random.seed(42)
    return splits


def data_process(df_x, target_cols=[]):
    print('Preparing data...')
    df = df_x.copy()

    ignore_columns = [
                    'participant_id',
                    'Basic_Demos_Enroll_Year',
                    'Basic_Demos_Study_Site',
                    'MRI_Track_Scan_Location',
                ]
    ignore_columns = [x for x in ignore_columns if x in df.columns]
    if ignore_columns:
        df = df.drop(ignore_columns, axis=1, errors="ignore")

    # Some feature engineering, main purpose is boost for ADHD_Outcome model,
    # Sex_F model mainly depends on functional connectome matrices
    df["edu_diff"] = df["Barratt_Barratt_P1_Edu"] - df["Barratt_Barratt_P2_Edu"]
    df["occ_diff"] = df["Barratt_Barratt_P1_Occ"] - df["Barratt_Barratt_P2_Occ"]

    sdq_cols = [c for c in df.columns if c.startswith("SDQ_SDQ_")]
    apq_cols = [c for c in df.columns if c.startswith("APQ_P_APQ_P")]

    # Normalize SDQ and APQ columns to [0, 1]
    # Tree base model not affect by scale, target only to calculate percent for each column across all columns in the group.
    scaler = MinMaxScaler()
    df.loc[:, sdq_cols] = scaler.fit_transform(df[sdq_cols])
    df.loc[:, apq_cols] = scaler.fit_transform(df[apq_cols])

    df['SDQ_SDQ_sum'] = df.filter(like='SDQ_SDQ').sum(axis=1)
    df['APQ_P_APQ_sum'] = df.filter(like='APQ_P_APQ').sum(axis=1)

    for col in apq_cols:
        df[f"{col}_percent"] = df[col]/df['APQ_P_APQ_sum']
    for col in sdq_cols:
        df[f"{col}_percent"] = df[col]/df['SDQ_SDQ_sum']

    TARGET_COLUMNS = target_cols
    if set(TARGET_COLUMNS).issubset(df.columns.tolist()):
        feature_train = df
        target_train = df[TARGET_COLUMNS].copy()**(1/1)
    else:
        feature_train = df

    # Process categorical columns
    categorical_columns=[
            "Basic_Demos_Study_Site",
            "MRI_Track_Scan_Location",
            "Basic_Demos_Enroll_Year",
            "PreInt_Demos_Fam_Child_Ethnicity",
            "PreInt_Demos_Fam_Child_Race",
            "Barratt_Barratt_P1_Occ",
            "Barratt_Barratt_P2_Occ",
            "Barratt_Barratt_P1_Edu",
            "Barratt_Barratt_P2_Edu",
            ]

    print('categorical_columns:', categorical_columns)

    for col in categorical_columns:
        if col in feature_train.columns:
            feature_train[col] = feature_train[col].round(0).astype(int) # better when round filled nan values with nearest int
            feature_train[col] = feature_train[col].astype('category')
            feature_train[col] = feature_train[col].cat.codes
            feature_train[col] = feature_train[col].astype('category')
        else:
            print('NO column', col)

    feature_train = feature_train.reset_index(drop=True)

    print('Done data process')
    return feature_train, target_train


class CONFIG:
    th = {"ADHD_Outcome": 0.8, "Sex_F": 0.14}
    target_cols = ['ADHD_Outcome']

CONFIG.target_cols = ["Sex_F"]
CONFIG.target_cols = ["ADHD_Outcome"]
CONFIG.target_cols = ["ADHD_Outcome", "Sex_F"]

def get_data(df, test, save_out=False):
    combined = pd.concat([df,test],axis=0,ignore_index=True)
    print(len(combined))

    feature_train, target_train = data_process(combined, target_cols=CONFIG.target_cols)
    print(len(feature_train))

    feature_test = feature_train.iloc[len(df):].reset_index(drop=True).copy()
    print(len(feature_test))

    feature_train = feature_train.iloc[:len(df)].copy()
    feature_train.drop([x for x in ["ADHD_Outcome", "Sex_F"] if x not in CONFIG.target_cols], axis=1, inplace=True)

    feature_test.drop(["ADHD_Outcome", "Sex_F"], axis=1, inplace=True)
    feature_test.drop(["weight"], axis=1, inplace=True)

    weights_train = feature_train[['weight']]

    feature_train.drop(["weight"], axis=1, inplace=True)
    target_train = target_train.iloc[:len(df)].copy()

    if save_out:
        feature_train.to_parquet("/kaggle/working/feature_train.parquet",
                                 compression=None,
                                 engine="fastparquet")
        feature_test.to_parquet("/kaggle/working/feature_test.parquet",
                                compression=None,
                                engine="fastparquet")
        weights_train.to_parquet("/kaggle/working/weights_train.parquet",
                                compression=None,
                                engine="fastparquet")
        df.to_parquet("/kaggle/working/df.parquet", compression=None)
    return feature_train, target_train, weights_train, feature_test

feature_train, target_train, weights_train, feature_test = get_data(df, test, save_out=True)
combined = pd.concat([feature_train,feature_test],axis=0,ignore_index=True)


feature_train


feature_test


target_train


# pack for MultiOutputRegressor, this allow diffrent hyperparameters for each model
# you can skip it
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from abc import ABCMeta, abstractmethod

from sklearn.base import (
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    _routing_enabled,
    process_routing,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel as sk_Parallel, delayed as sk_delayed
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    has_fit_parameter,
)


def _fit_estimator(estimator, X, y, sample_weight=None, set_params=None, **fit_params):
    estimator = clone(estimator)
    if set_params:
        if hasattr(estimator, "set_params"):
            estimator.set_params(**set_params)
        else:
            print('No set_params')
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


class CustomMultiOutputRegressor(MultiOutputRegressor):

    @_fit_context(
        # MultiOutput*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, custom_allow_cols=None, custom_set_params=[], **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if _routing_enabled():
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            routed_params = process_routing(
                self,
                "fit",
                **fit_params,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            fit_params_validated = _check_method_params(X, params=fit_params)
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            if sample_weight is not None:
                routed_params.estimator.fit["sample_weight"] = sample_weight
        X_new = []
        self.custom_allow_cols = []
        if custom_allow_cols:
            for list_cols in custom_allow_cols[:y.shape[1]]:
                self.custom_allow_cols.append(list_cols)
                X_new.append(X[list_cols])
            for _ in range(y.shape[1]-len(custom_allow_cols)):
                self.custom_allow_cols.append(list(X.columns))
                X_new.append(X)

        for _ in range(y.shape[1]-len(custom_set_params)):
            custom_set_params.append({})
        # print(routed_params.estimator.fit)
        if 'custom_allow_cols' in routed_params.estimator.fit.keys():
            routed_params.estimator.fit.pop('custom_allow_cols')
        if 'custom_set_params' in routed_params.estimator.fit.keys():
            routed_params.estimator.fit.pop('custom_set_params')
        if X_new:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X_new[i], y[:, i], set_params=custom_set_params[i], **routed_params.estimator.fit
                )
                for i in range(y.shape[1])
            )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], set_params=custom_set_params[i], **routed_params.estimator.fit
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")
        if not self.custom_allow_cols:
            y = sk_Parallel(n_jobs=self.n_jobs)(
                sk_delayed(e.predict)(X) for e in self.estimators_
            )
        else:
            y = sk_Parallel(n_jobs=self.n_jobs)(
                sk_delayed(e.predict)(X[self.custom_allow_cols[i]]) for i, e in enumerate(self.estimators_)
            )
        return np.asarray(y).T


# parallel

def train_fold(fold, n_jobs, train_index, val_index,
               feature_train, target_train, feature_test, weights_train, best_hypr, combined, model_type='lgb',
               custom_set_params=[], custom_allow_cols=[],
               skip_opt=False,
               focus_fold=None, save_end=True):
    """
    Train a single fold.
    """

    if focus_fold is not None:
        fold = focus_fold
    print(f"Training fold {fold + 1}...")

    # Split the data into training and validation sets
    X_train, X_val = feature_train.iloc[train_index].copy(), feature_train.iloc[val_index].copy()
    y_train, y_val = target_train.iloc[train_index].copy(), target_train.iloc[val_index].copy()
    weight_train, weight_val = weights_train.iloc[train_index].copy(), weights_train.iloc[val_index].copy()

    model_lgb_reg = LGBMRegressor(
                            objective='binary',
                            random_state=42,
                            device_type='cpu',
                          **best_hypr.copy(),
                          # is_unbalance=True,
                            extra_trees=True, # use for faster training with litte loss of accuracy
                            # class_weight={0:1,1:1.5},
                            verbose=-1,
                          n_jobs=multiprocessing.cpu_count()//n_jobs,
    )
    model_lgb_clf = LGBMClassifier(
                            objective='binary',
                            random_state=42,
                            device_type='cpu',
                          **best_hypr.copy(),
                          # is_unbalance=True,
                            extra_trees=True,
                            # class_weight={0:1,1:1.5},
                            verbose=-1,
                          n_jobs=multiprocessing.cpu_count()//n_jobs,
    )

    model_xgb_reg = XGBRegressor(objective='binary:logistic', **best_hypr.copy(),
                      device='cpu',
                      # weight_column='weight',
                      random_state=42,
                      enable_categorical=True,
                      nthread=multiprocessing.cpu_count()//n_jobs,
                    )
    model_xgb_clf = XGBClassifier(objective='binary:logistic', **best_hypr.copy(),
                      device='cpu',
                      random_state=42,
                      enable_categorical=True,
                      # nthread=multiprocessing.cpu_count()//n_jobs,
                      nthread=multiprocessing.cpu_count()//n_jobs,
                    )

    if 'reg_alpha' in list(best_hypr.keys()):
        best_hypr.pop('reg_alpha')
    if 'max_leaves' in list(best_hypr.keys()):
        best_hypr.pop('max_leaves')

    model_cat_clf = CatBoostClassifier(loss_function="CrossEntropy", **best_hypr.copy(), # MultiLogloss, MultiCrossEntropy
        verbose=False,
        random_seed=42,
        thread_count=multiprocessing.cpu_count()//n_jobs,
        grow_policy='Lossguide',
        task_type = 'CPU',
        cat_features=list(X_train.select_dtypes(include=['category']).columns),
    )


    if target_train.shape[-1]>1 or 1:
        if model_type == 'lgb_reg':
            model = CustomMultiOutputRegressor(model_lgb_reg)
        elif model_type == 'lgb_clf':
            model = CustomMultiOutputRegressor(model_lgb_clf)
        elif model_type == 'xgb_reg':
            model = CustomMultiOutputRegressor(model_xgb_reg)
        elif model_type == 'xgb_clf':
            model = CustomMultiOutputRegressor(model_xgb_clf)
        elif model_type == 'cat_clf':
            model = CustomMultiOutputRegressor(model_cat_clf)
        else:
            model = CustomMultiOutputRegressor(model)

    # Train the model
    if fold==0:
        print('START TRAIN')


    if model_type=='tabm':
        model.fit(X_train.drop(columns=CONFIG.target_cols), y_train.values,
                  X_val=X_val.drop(columns=CONFIG.target_cols), y_val=y_val.values[:,0], # pytabkit must have val to work
                 )
    else:
        model.fit(X_train.drop(columns=CONFIG.target_cols), y_train.values,
                  custom_allow_cols=custom_allow_cols, custom_set_params=custom_set_params,
             )

    if fold==0:
        print('END TRAIN')

    # Predict on the validation set
    y_pred_train = model.predict(X_train.drop(columns=CONFIG.target_cols))

    y_pred_val = model.predict(X_val.drop(columns=CONFIG.target_cols))

    # Calculate f1 for validation set
    F1_score_trains = []
    F1_score_vals = []


    for i, c in enumerate(CONFIG.target_cols):
        F1_score_trains.append(F1(y_train.values[:,i], y_pred_train[:,i], CONFIG.th[c], weight=None))
        F1_score_vals.append(F1(y_val.values[:,i], y_pred_val[:,i], CONFIG.th[c], weight=None))

    # Predict on the test set
    y_pred_test = model.predict(feature_test)

    if not skip_opt:
        sc = {}
        ts = []
        for idx, c in enumerate(CONFIG.target_cols):
            ts.append(f"{c} Mean: {y_pred_test.mean():4f}".ljust(10))

        print(f"Fold {fold} val F1: {' '.join([*map(str, F1_score_vals)])}".ljust(9), f"Mean: {(y_pred_test).mean():.2f}",
                          f"train F1: {' '.join([*map(str, F1_score_trains)])}".ljust(9), ' '.join(ts)
         )

    feature_importances = []
    lgbm_shaps = []
    
    for i in range(len(model.estimators_)):
        if 'LGBM' in str(model):
            lgbm_shaps.append(model.estimators_[i].predict(feature_test, pred_contrib=True))
            feature_importances.append(pd.Series(model.estimators_[i].feature_importances_,
                                                 index=X_train.drop(columns=CONFIG.target_cols).columns).sort_values(ascending=False))
        elif "XGB" in str(model):
            feature_importances.append(pd.Series(model.estimators_[i].get_booster().get_score(importance_type='gain'),
                                                 index=X_train.drop(columns=CONFIG.target_cols).columns).sort_values(ascending=False))

    os.makedirs('/kaggle/tmp/folds', exist_ok=True)
    os.makedirs('/kaggle/working/folds', exist_ok=True)
    if save_end:
        for i in range(len(model.estimators_)):
            model.estimators_[i].booster_.save_model(f'/kaggle/working/folds/model_{CONFIG.target_cols[i]}_{model_type}_fold_{fold}.txt')

    return fold, y_pred_train, y_pred_val, y_pred_test, F1_score_trains, F1_score_vals, feature_importances, lgbm_shaps



n_splits = 8

# Time tracking
start_time = time.time()

splits = balanced_kfold_split(feature_train, df, n_splits=n_splits,
                              group_column='ADHD_Outcome', # Sex_F
                              # df_buff=df[(df['Sex_F']==1) & (df['ADHD_Outcome']==1)]
                              seed=42,
                             )

splits = [[feature_train.index, feature_train.index]]

n_splits = len(splits)

best_hypr_dict = {
'ADHD_Outcome':{
    # lgb 1 fold
    'lgb_reg_1' : [
        {'reg_alpha': 4.455169907706503e-08, 'reg_lambda': 0.0012155452141569558, 'learning_rate': 0.01, 'n_estimators': 200, 'max_leaves': 118, 'min_child_samples': 9},
        {'reg_alpha': 0.0014322603999347813, 'reg_lambda': 0.017887504299023544, 'learning_rate': 0.01, 'n_estimators': 500, 'max_leaves': 107, 'min_child_samples': 40}
    ],
    # lgb 8 folds
    'lgb_reg_8':[
        {'reg_alpha': 7.21654762517268e-07, 'reg_lambda': 1.7302178976767463e-06, 'learning_rate': 0.01, 'n_estimators': 200, 'max_leaves': 77, 'min_child_samples': 14},
        {'reg_alpha': 0.0001545532323089221, 'reg_lambda': 2.353979413723042e-05, 'learning_rate': 0.01, 'n_estimators': 200, 'max_leaves': 130, 'min_child_samples': 19},
    ],
    # lgbclf 1 fold
    # lgbclf 8 folds
    'lgb_clf_8': [
        {'reg_alpha': 6.057278408593891e-05, 'reg_lambda': 0.001249829196522666, 'learning_rate': 0.04, 'n_estimators': 200, 'max_leaves': 15, 'min_child_samples': 9},
    ],

    # xgb 1 folds
    'xgb_reg_1':[
        {'reg_alpha': 0.029033820445384596, 'reg_lambda': 4.74054550093646, 'learning_rate': 0.01, 'n_estimators': 227, 'max_depth': 7, 'max_leaves': 72},
        {'reg_alpha': 1.3172799701422377, 'reg_lambda': 0.0007695940253933907, 'learning_rate': 0.01, 'n_estimators': 213, 'max_depth': 9, 'max_leaves': 28},
        {'reg_alpha': 1.4521971699116607, 'reg_lambda': 1.489761594015251e-05, 'learning_rate': 0.01, 'n_estimators': 234, 'max_depth': 5, 'max_leaves': 74},
        {'reg_alpha': 1.3369587538390495, 'reg_lambda': 1.0065560753084458e-08, 'learning_rate': 0.01, 'n_estimators': 197, 'max_depth': 5, 'max_leaves': 109},
        {'reg_alpha': 1.1759442280666481, 'reg_lambda': 0.00044596125226314574, 'learning_rate': 0.01, 'n_estimators': 193, 'max_depth': 8, 'max_leaves': 28},
        {'reg_alpha': 1.361667961454591, 'reg_lambda': 4.459283632001179e-06, 'learning_rate': 0.01, 'n_estimators': 200, 'max_depth': 5, 'max_leaves': 73},
        {'reg_alpha': 3.1875098760107643, 'reg_lambda': 1.4252354685374041e-05, 'learning_rate': 0.01, 'n_estimators': 328, 'max_depth': 6, 'max_leaves': 70},
        {'reg_alpha': 1.8518674080021942, 'reg_lambda': 0.11597077081822346, 'learning_rate': 0.01, 'n_estimators': 272, 'max_depth': 7, 'max_leaves': 121},
    ],
    # xgb 8 folds
    'xgb_reg_8': [
        {'reg_alpha': 9.939092939542823, 'reg_lambda': 0.03321335519659749, 'learning_rate': 0.01, 'n_estimators': 104, 'max_depth': 5, 'max_leaves': 110}
    ],
    # xgbclf 8 folds
    'xgb_clf_8':[
        {'reg_alpha': 7.34563570994724e-08, 'reg_lambda': 0.0003272206976903343, 'learning_rate': 0.01, 'n_estimators': 228, 'max_depth': 8, 'max_leaves': 130},
    ],
},
'Sex_F':{
    # lgb 1 fold

    'lgb_reg_1': [
        {'reg_alpha': 1.246942210779823, 'reg_lambda': 2.5155216944449413e-06, 'learning_rate': 0.09, 'n_estimators': 300, 'max_leaves': 33, 'min_child_samples': 34},
        {'reg_alpha': 5.35780081242787, 'reg_lambda': 0.007195987127870962, 'learning_rate': 0.09999999999999999, 'n_estimators': 500, 'max_leaves': 124, 'min_child_samples': 45},
        {'reg_alpha': 2.30724102951642, 'reg_lambda': 1.576755394055527e-07, 'learning_rate': 0.08, 'n_estimators': 400, 'max_leaves': 62, 'min_child_samples': 29},
        {'reg_alpha': 4.868930058080973, 'reg_lambda': 3.2388939558136083, 'learning_rate': 0.08, 'n_estimators': 400, 'max_leaves': 53, 'min_child_samples': 42},
        {'reg_alpha': 4.258789560980834, 'reg_lambda': 0.00017535261659203767, 'learning_rate': 0.09, 'n_estimators': 500, 'max_leaves': 119, 'min_child_samples': 42},
        {'reg_alpha': 0.9502602345156709, 'reg_lambda': 0.01681412200856409, 'learning_rate': 0.060000000000000005, 'n_estimators': 200, 'max_leaves': 37, 'min_child_samples': 44},
    ],
    # # lgb 8 fold
    'lgb_reg_8': [
        {'reg_alpha': 1.7116594317226732, 'reg_lambda': 5.697803992056026, 'learning_rate': 0.08, 'n_estimators': 400, 'max_leaves': 29, 'min_child_samples': 38},
        {'reg_alpha': 1.0097998663971504, 'reg_lambda': 2.1071847869742293e-05, 'learning_rate': 0.08, 'n_estimators': 500, 'max_leaves': 15, 'min_child_samples': 41},
        {'reg_alpha': 2.3257113562606175, 'reg_lambda': 5.738586992184839, 'learning_rate': 0.08, 'n_estimators': 500, 'max_leaves': 53, 'min_child_samples': 39},
        {'reg_alpha': 3.1347560659062945, 'reg_lambda': 7.079193366751104, 'learning_rate': 0.08, 'n_estimators': 500, 'max_leaves': 32, 'min_child_samples': 37},
    ],
    # xgb 1 fold
    'xgb_reg_1': [
        {'reg_alpha': 0.3648559885746107, 'reg_lambda': 2.59285720838081e-07, 'learning_rate': 0.09, 'n_estimators': 270, 'max_depth': 9, 'max_leaves': 21},
    ],
},
}

submit_df = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')

ft_imps = {'ADHD_Outcome': [], 'Sex_F': []}
lgbm_shap_scores = {'ADHD_Outcome': [], 'Sex_F': []}

for target in ['ADHD_Outcome', 'Sex_F']:
# for target in ['ADHD_Outcome']:
# for target in ['Sex_F']:
    print(f"START TRAINING TARGET: {target}")
    CONFIG.target_cols = [target]
    feature_train, target_train, weights_train, feature_test = get_data(df, test, save_out=False)

    # We don't use the functional connectome features from ADHD predict!
    if target=='ADHD_Outcome':
        feature_train = feature_train.drop([col for col in feature_test.columns if 'throw' in col], axis=1)
        feature_test = feature_test.drop([col for col in feature_test.columns if 'throw' in col], axis=1)

    oof_all = np.zeros((len(feature_test), target_train.values.shape[-1]))
    oof_len = 0
    for k in tqdm(best_hypr_dict[target].keys()):
        model_type, n_folds = k.rsplit('_', maxsplit=1)
        n_folds = int(n_folds)
        splits = balanced_kfold_split(feature_train, df, n_splits=n_folds,
                              group_column='ADHD_Outcome',
                              seed=42,
                             )

        n_splits = len(splits)

        # Run training for each hyperparameter set
        for best_hypr in best_hypr_dict[target][k]:

            n_jobs_x = min(8, multiprocessing.cpu_count()//4) # ensure we have at least 4 cores for each fold

            results = Parallel(n_jobs=n_jobs_x, backend='threading')(
                delayed(train_fold)(fold, min(len(splits), n_jobs_x), train_index, val_index,
                                    feature_train.copy(deep=False), target_train.copy(), feature_test.copy(), weights_train,
                                    deepcopy(best_hypr), combined.copy(deep=False),
                                    custom_allow_cols = [],
                                    model_type=model_type,
                                    skip_opt=False,
                                    save_end=0
                                   )
                for fold, (train_index, val_index) in tqdm(enumerate(splits), total=len(splits))
            )

            # Sort results to prevent origin order
            results = sorted(results, key=lambda x: x[0])
            
            oof_len += n_splits

            f1_scores_train = []
            f1_scores = []

            y_pred_vals = []
            
            for i, fold_result in enumerate(results):
                idx, y_pred_train_, y_pred_val_, y_pred_test_, f1_score_train, F1_score, feature_importances, lgbm_shaps = fold_result

                oof_all += y_pred_test_
                y_pred_vals.append(y_pred_val_)
                ft_imps[target].append(feature_importances)
                lgbm_shap_scores[target].append(lgbm_shaps)
                f1_scores_train.append(f1_score_train)
                f1_scores.append(F1_score)


    # Final predictions: average the accumulated predictions
    final_predictions_proba = oof_all / oof_len
    final_predictions = np.concatenate([(final_predictions_proba[:,i] > CONFIG.th[tgx]).astype(int)[:, None] for i, tgx in enumerate(CONFIG.target_cols)], axis=1)

    print(f"Final predictions mean: {final_predictions.mean(axis=0)}")
    print(f"Amount 1 in predictions:  {final_predictions.sum(axis=0)}")

    for i, c in enumerate(CONFIG.target_cols):
        final_predictions[:,i] = (final_predictions_proba[:,i] > CONFIG.th[c]).astype(int)
        submit_df[c] = final_predictions[:,i]
print('Total runtime:', time.time()-start_time)
submit_df.to_csv('/kaggle/working/submission.csv', index=False)
submit_df.head(10)


# check feature important
feature_importances = ft_imps['ADHD_Outcome'][0][0]
fig, axs = plt.subplots(1, 1, figsize=(10, 6))

palette = sns.color_palette("RdYlGn_r", len(feature_train.columns))
# lgbm_importances = pd.Series(model.feature_importances_, index=feature_train.columns).sort_values(ascending=False)
sns.barplot(y=feature_importances.index, x=feature_importances.values, orient='h', palette=palette)

plt.title('ADHD_Outcome Model Feature Important')
plt.tight_layout()
plt.show()



# check feature important
feature_importances = ft_imps['Sex_F'][0][0][:60]
fig, axs = plt.subplots(1, 1, figsize=(10, 6))

palette = sns.color_palette("RdYlGn_r", len(feature_train.columns))
sns.barplot(y=feature_importances.index, x=feature_importances.values, orient='h', palette=palette)

plt.title('Sex_F Model Feature Important')
plt.tight_layout()
plt.show()



%%writefile optim.py
import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import f1_score as f1_score_calc
from lightgbm import LGBMRegressor, LGBMClassifier
import argparse
import time
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import random
import multiprocessing
import os
SEED = 42
random.seed(42)
np.random.seed(SEED)

def F1(y_true, y_pred, threshold=0.5, weight=None):
    x = f1_score_calc(y_true, (y_pred > threshold).astype(int), sample_weight=weight)
    # print(x)
    return x

def balanced_kfold_split(df_xx, df_x, n_splits=1, group_column='ADHD_Outcome', df_buff=None, seed=42):
    """
    Custom balanced k-fold split for a dataframe ensuring balanced distribution for each groups
    Args:
        df_xx: DataFrame to be split
        df_x: Origin DataFrame containing the target variable
        n_splits: Number of splits
        group_column: Column name to be used for grouping
        df_buff: Buffer DataFrame to be used in all folds
    """
    if n_splits==1:
        return [[df_xx.index, df_xx.index]]

    # Align group column from df_x to df
    df = df_xx.copy()
    df[group_column] = df_x[group_column].values  # Ensure correct order of target in df
    buffer_indices = None
    if df_buff is not None:
        buffer_indices = df_buff.index.tolist()

    # Initialize folds
    folds = [[] for _ in range(n_splits)]
    groups = df[group_column].unique()

    for group in groups:
        group_indices = df[df[group_column] == group].index.tolist()
        np.random.seed(seed)
        np.random.shuffle(group_indices)  # Shuffle indices for the group

        # Distribute the group indices across folds
        for i, idx in enumerate(group_indices):
            folds[i % n_splits].append(idx)

    # Create train and validation indices for each fold
    splits = []
    for i in range(n_splits):
        val_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        if buffer_indices:
            buffer_indices_tmp = [idx for idx in buffer_indices if idx not in train_indices]
            print('buff', len(buffer_indices_tmp))
            train_indices = np.array(list(train_indices)+buffer_indices_tmp)
        splits.append((train_indices, val_indices))
    np.random.seed(42)
    return splits

def optim_th(final_pred, gt=None):
    # gt = {'ADHD_Outcome': [1,1,1,1,...],
    #      'Sex_F': [1,1,1,1,...]
    # }

    # Compute F1 scores and find the best threshold
    thresholds = np.linspace(0, 1, 100)
    best_theshold_dict = {}
    best_score_dict = {}

    for idx, c in enumerate(CONFIG.target_cols):
        adhd_scores = []
        splits = range(1)
        for t in thresholds:
            x = 0
            for i in range(len(splits)):
                weights = [1 for _ in gt[c]]
                if len(final_pred.shape)==1:
                    final_pred = final_pred[:,None]
                tmp_score = F1(gt[c], final_pred[:len(gt[c])][:, idx], t, weight=weights)
                x += tmp_score
            adhd_scores.append(x/len(splits))
        max_score = max(adhd_scores)
        # boolean mask of where scores == max
        # we find list all threshold given best score
        mask = np.array(adhd_scores) == max_score
        best_theshold_dict[c] = list(thresholds[mask])
        # print('best_adhd_threshold', best_adhd_threshold)
        best_score_dict[c] = max_score
    return best_theshold_dict, best_score_dict

def train_fold(fold, n_jobs, train_index, val_index,
               feature_train, target_train, feature_test, weights_train, best_hypr, model_type='lgb',only_run_fold=None, global_parallel=8):

    if only_run_fold is not None and only_run_fold!=fold:
        print('SKIP FOLD')
        return None

    if 'Sex_F' not in CONFIG.target_cols:
        feature_train = feature_train.drop([col for col in feature_test.columns if 'throw' in col], axis=1)
        feature_test = feature_test.drop([col for col in feature_test.columns if 'throw' in col], axis=1)
        
    # Create train and validation sets
    X_train, X_val = feature_train.iloc[train_index], feature_train.iloc[val_index]
    y_train, y_val = target_train.iloc[train_index], target_train.iloc[val_index]
    # y_train, y_val = y_train, y_val
    weight_train, weight_val = weights_train.iloc[train_index], weights_train.iloc[val_index]
    
    if 'lgb' in model_type:

        model = LGBMRegressor(**best_hypr,
        # model = LGBMClassifier(**best_hypr,
                              objective='binary',
                              n_jobs=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
                              random_state=42,
                              extra_trees=True,
                              # weight_column='weight',
                             )

    elif 'xgb' in model_type:
        model = XGBRegressor(objective='binary:logistic', **best_hypr,
        # model = XGBClassifier(objective='binary:logistic', **best_hypr,
                          device='cpu',
                        random_state=42,
                        enable_categorical=True,
                          nthread=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
                        )

    elif model_type=='cat':
        model = CatBoostClassifier(loss_function="CrossEntropy", **best_hypr, # MultiLogloss, MultiCrossEntropy
            verbose=False,
            random_seed=42,
            thread_count=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
            grow_policy='Lossguide',
            task_type = 'CPU',
            cat_features=list(X_train.select_dtypes(include=['category']).columns),

        )
    model = MultiOutputRegressor(model)
    buf = 0
    # # Train the model
    model.fit(X_train.drop(columns=CONFIG.target_cols), y_train.values,
             # sample_weight=weight_train['weight'].values.ravel()
             )

    y_pred_val = model.predict(X_val.drop(columns=CONFIG.target_cols))
    y_pred_test = model.predict(feature_test)
    # if fold==0:
    #     print(y_pred_test)
    return fold, y_pred_val, y_val, y_pred_test

# Objective function for Optuna, runs optimization for each fold
def objective(trial, feature_train, target_train, weights_train, feature_test, pseudo_test_label, splits, model_type='lgb', only_run_fold=None, global_parallel=8):
    if model_type=='lgb':
        best_hypr = {
            "verbosity": -1,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(200, 501, 100))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
            'min_child_samples': trial.suggest_int('min_child_samples', 7, 50),
        }
    elif model_type=='xgb':
        best_hypr = {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(100, 501, 10))),
            'max_depth': trial.suggest_categorical("max_depth", list(range(5, 11, 1))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
        }
    elif model_type=='cat':
        best_hypr = {
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(100, 501, 100))),
            'max_depth': trial.suggest_categorical("max_depth", list(range(5, 12, 1))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
        }


    n_jobs_x = min(8, multiprocessing.cpu_count()//4) # ensure we have at least 4 cores for each fold
    # Use joblib to parallelize the fold training
    results = Parallel(n_jobs=n_jobs_x, backend='threading')(
        delayed(train_fold)(fold, min(len(splits), n_jobs_x), train_index, val_index,
                            feature_train.copy(deep=False), target_train.copy(), feature_test.copy(), weights_train,
                            best_hypr, model_type, only_run_fold, global_parallel
                           )
        for fold, (train_index, val_index) in enumerate(splits)
    )
    results = sorted(results, key=lambda x: x[0])

    # Aggregate predictions and scores

    best_threshold_oof = {k: [] for k in CONFIG.target_cols}
    best_score_oof = {k: 0 for k in CONFIG.target_cols}
    for i, fold_result in enumerate(results):
        idx, y_pred_val, y_val, y_pred_test = fold_result
        for i,col in enumerate(CONFIG.target_cols):
            best_threshold, best_score = optim_th(y_pred_val, {k: y_val.values[:,i] for i, k in enumerate(CONFIG.target_cols)})
            for k, v in best_threshold.items():
                best_threshold_oof[k].extend(v)
                best_score_oof[k] += best_score[k]
        
    if len(splits) == 1:
        # threshold optimizer to 0.5
        err = 0
        for col in CONFIG.target_cols:
            err += np.abs(0.5-np.mean(best_threshold_oof[col]))

        test_f1 = []
        # print(pseudo_test_label.values.shape)
        for i,col in enumerate(CONFIG.target_cols):
            test_f1.append(F1(pseudo_test_label.values[:,i], y_pred_test[:,i]))
            
        return (np.mean(list(best_score_oof.values()))+np.mean(test_f1))/2 - err*1.2
    else:
        return np.mean(list(best_score_oof.values()))


# Main function to handle 8-fold cross-validation and Optuna optimization
def run_optimization(feature_train, df, target_train, weights_train, feature_test, model_type='lgb', only_run_fold=None, global_parallel=8):
    
    # Initialize KFold for 8 splits
    splits = balanced_kfold_split(feature_train, df, n_splits=8, group_column='ADHD_Outcome')
    
    # splits = [[feature_train.index, feature_train.index]] # Use for whole data hyperparamters optimizer
    pseudo_test_label = None
    if len(splits)==1:
        print('Use pseudo label for whole data optimizer')
        sub = pd.read_csv('/kaggle/working/submission.csv')
        pseudo_test_label = sub[CONFIG.target_cols]
        
    print('NUM SPLITS:', len(splits))
    best_hyperparameters = []

    # Optimize hyperparameters for this fold using Optuna
    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, feature_train, target_train, weights_train, feature_test, pseudo_test_label,
                                           splits, model_type, only_run_fold, global_parallel),
                   n_trials=10 if 'Sex_F' not in CONFIG.target_cols else 200 # Sex_F optimization run slow, so reduce n_trials
                  )

    best_hyperparameters.append(study.best_trial.params)
    print(f"Best params: {study.best_trial.params}")

    return best_hyperparameters

class CONFIG:
    th = {"ADHD_Outcome": 0.7, "Sex_F": 0.15}
    target_cols = ['ADHD_Outcome']
CONFIG.target_cols = ['Sex_F']
CONFIG.target_cols = ['ADHD_Outcome']
# CONFIG.target_cols = ['ADHD_Outcome', 'Sex_F']

if not os.path.isfile('/kaggle/working/feature_train.parquet'):
    print('File not found, make sure you save_out in get_data function.')
    
feature_train = pd.read_parquet("feature_train.parquet", engine="fastparquet")
feature_test = pd.read_parquet("feature_test.parquet", engine="fastparquet")
weights_train = pd.read_parquet("weights_train.parquet", engine="fastparquet")
df = pd.read_parquet("df.parquet", engine="fastparquet")
target_train = feature_train[CONFIG.target_cols].copy()

categorical_columns=[
        "Basic_Demos_Study_Site",
        "Basic_Demos_Study_Site",
        "MRI_Track_Scan_Location",
        "Basic_Demos_Enroll_Year",
        "PreInt_Demos_Fam_Child_Ethnicity",
        "PreInt_Demos_Fam_Child_Race",
        'Barratt_Barratt_P1_Occ',
        'Barratt_Barratt_P2_Occ',
        'Barratt_Barratt_P1_Edu',
        'Barratt_Barratt_P2_Edu',
        ]
combined = pd.concat([feature_train,feature_test],axis=0,ignore_index=True)
for col in categorical_columns:
    if col in feature_test.columns.tolist():
        combined[col] = combined[col].astype('category')
    else:
        print('No', col)

# print(len(combined))

feature_test = combined.iloc[len(feature_train):].reset_index(drop=True).copy()
# print(len(feature_test))
feature_train = combined.iloc[:len(feature_train)].copy()
feature_train.drop([x for x in ["ADHD_Outcome", "Sex_F"] if x not in CONFIG.target_cols and x in list(feature_train.columns)], axis=1, inplace=True)
feature_test.drop([x for x in ["ADHD_Outcome", "Sex_F"] if x in list(feature_test.columns)], axis=1, inplace=True)


if __name__ == "__main__":
    print('START')
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_run_fold", type=int, required=False, default=None, help="Fold number to run (0-7)")
    parser.add_argument("--global_parallel", type=int, required=True, default=None, help="Max parallel")
    parser.add_argument("--model_type", type=str, required=True, default='lgb', help="Model type")
    args = parser.parse_args()
    print('RUN CONFIG:\n', args)
    # Replace with your actual data (feature_train, target_train, feature_test)
    best_hyperparameters = run_optimization(feature_train, df, target_train, weights_train, feature_test,
                                            model_type=args.model_type,
                                            only_run_fold=args.only_run_fold,
                                           global_parallel=args.global_parallel)

    if args.only_run_fold is None:
        # Output the best hyperparameters for all folds

        print("Best hyperparameters:")
        print(best_hyperparameters)
        for idx, params in enumerate(best_hyperparameters):
            print(f"Fold {idx + 1}: {params}")
        # Save the results to a file
        with open(f'/kaggle/working/best_hyperparameters.txt', 'w') as f:
            for idx, params in enumerate(best_hyperparameters):
                f.writelines([f"Fold {idx + 1}:\n"])
                for key, value in params.items():
                    f.writelines([f"{key}: {value}\n"])
    else:
        print("Best hyperparameters for fold:", args.only_run_fold)
        print(f"Fold: {best_hyperparameters[0]}")
        # Save the results to a file
        with open(f'/kaggle/working/best_hyperparameters_fold_{args.only_run_fold}.txt', 'w') as f:
            for idx, params in enumerate(best_hyperparameters):
                f.writelines([f"Fold {args.only_run_fold}:\n"])
                for key, value in params.items():
                    f.writelines([f"{key}: {value}\n"])


# Use for 8 folds optimizer
# !python optim.py --global_parallel 1 --model_type xgb
# !python optim.py --global_parallel 1 --model_type lgb


# # use for whole data optimizer
# import subprocess
# import threading
# from colorama import Fore, Style

# # Total folds and maximum parallel processes
# total_folds = 8
# max_parallel = 2

# # Track running processes
# processes = []

# # Assign unique colors to folds
# COLORS = [
#     Fore.RED,
#     Fore.GREEN,
#     Fore.YELLOW,
#     Fore.BLUE,
#     Fore.MAGENTA,
#     Fore.CYAN,
#     Fore.WHITE,
#     Fore.LIGHTGREEN_EX,
# ]

# # Function to read and print logs from a process in real-time
# def stream_output(process, fold):
#     color = COLORS[fold % len(COLORS)]
#     for line in iter(process.stdout.readline, b""):
#         if line.strip():
#             print(f"{color}[Fold {fold}] {line.strip()}{Style.RESET_ALL}")
#     process.stdout.close()

# # Start the first batch of processes
# for fold in range(max_parallel):
#     print(f"Starting fold {fold}")
#     process = subprocess.Popen(
#         ["python", "optim.py", "--global_parallel", str(max_parallel), "--model_type", 'lgb'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         bufsize=1,
#         text=True,
#     )
#     threading.Thread(target=stream_output, args=(process, fold), daemon=True).start()
#     processes.append((fold, process))

# # Manage remaining processes
# for fold in range(max_parallel, total_folds):
#     # Wait for any process to finish
#     while True:
#         for i, (running_fold, process) in enumerate(processes):
#             if process.poll() is not None:  # Process finished
#                 print(f"Fold {running_fold} finished. Starting fold {fold}")
#                 process = subprocess.Popen(
#                     ["python", "optim.py", "--global_parallel", str(max_parallel), "--model_type", 'lgb'],
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.STDOUT,
#                     bufsize=1,
#                     text=True,
#                 )
#                 threading.Thread(target=stream_output, args=(process, fold), daemon=True).start()
#                 processes[i] = (fold, process)
#                 break
#         else:
#             continue
#         break

# # Wait for all remaining processes to finish
# for running_fold, process in processes:
#     process.wait()
#     print(f"Fold {running_fold} finished.")

