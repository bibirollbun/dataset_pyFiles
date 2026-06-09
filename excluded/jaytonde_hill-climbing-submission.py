!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install --no-index -U --find-links=/kaggle/input/tabm-tabular-dl-library tabm==0.0.1.dev0
!pip install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


!pip -q install /kaggle/input/tabpfn-v2/tabpfn-2.0.0-py3-none-any.whl


!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/torchtuples-0.2.2-py3-none-any.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/feather-format-0.4.1.tar.gz
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/pyzstd-0.16.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/pyppmd-1.1.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/pybcj-1.0.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/multivolumefile-0.2.3-py3-none-any.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/inflate64-1.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/Brotli-1.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/py7zr-0.22.0-py3-none-any.whl
!pip install --no-index /kaggle/input/cibmtr-pip-install-pycox/pycox-0.3.0-py3-none-any.whl


!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl
#!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q /kaggle/input/wheelhouse-cibmtr/scikit_learn-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/einops-0.7.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl


!pip install /kaggle/input/wheelhouse-cibmtr/torchsurv-0.1.4-py3-none-any.whl


# !pip install -q autogluon --no-index --find-links=file:///kaggle/input/autogluon/v1.0.0


# !pip install --no-index /kaggle/input/autogluon/v1.0.0/scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.common-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.features-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.multimodal-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.tabular-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.timeseries-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon.core-1.0.0-py3-none-any.whl
# !pip install --no-index /kaggle/input/autogluon/v1.0.0/autogluon-1.0.0-py3-none-any.whl


import numpy as np
import pandas as pd
from scipy.stats import rankdata 

import sys
sys.path.append('/kaggle/input/tabm-tabular-dl-library')

import os
import tabm
import math
import torch
import random
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import rtdl_num_embeddings
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.model_selection import KFold
from scipy.stats import rankdata 
from colorama import Fore, Style
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from tabm_reference import Model, make_parameter_groups
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from pytorch_tabnet.tab_model import TabNetRegressor


experiments = [
"/kaggle/input/xgboost-exp-01",
"/kaggle/input/catboost-exp-01", 
"/kaggle/input/lgbm-exp-01",
"/kaggle/input/xgboost-exp-02",
"/kaggle/input/catboost-exp-02",
"/kaggle/input/lgbm-exp-03",
"/kaggle/input/catboost-exp-03",
"/kaggle/input/tabm-exp-01",
"/kaggle/input/nn-exp-01",
"/kaggle/input/tn-exp-01",
"/kaggle/input/tf-exp-01",
"/kaggle/input/svr-exp-01",
"/kaggle/input/abd-exp-01",
"/kaggle/input/catboost-exp-04",
"/kaggle/input/lgbm-exp-04",
"/kaggle/input/tabm-exp-02",
"/kaggle/input/ds-exp-01",
"/kaggle/input/nn-exp-02",
"/kaggle/input/nn-exp-04",
"/kaggle/input/catboost-exp-05",
"/kaggle/input/xgboost-exp-05",
"/kaggle/input/lgbm-exp-05",
"/kaggle/input/tn-exp-02",
"/kaggle/input/ag-exp-01",
"/kaggle/input/vr-exp-01",
"/kaggle/input/tt-exp-01",
"/kaggle/input/en-exp-01",
"/kaggle/input/en-exp-02",
"/kaggle/input/nn-exp-05",
"/kaggle/input/rf-exp-05",
"/kaggle/input/mcts-exp-02",
"/kaggle/input/catboost-exp-06",
"/kaggle/input/xgboost-exp-06",
"/kaggle/input/lgbm-exp-06",
"/kaggle/input/nn-exp-06",
"/kaggle/input/xgboost-exp-07",
"/kaggle/input/xgboost-exp-09",
"/kaggle/input/prlnn-exp-01",
"/kaggle/input/ri-exp-06",
"/kaggle/input/xgboost-exp-10",
"/kaggle/input/lasso-exp-01",
"/kaggle/input/lir-exp-01",
"/kaggle/input/svr-exp-06",
"/kaggle/input/et-exp-01",
"/kaggle/input/cnn-exp-01",
"/kaggle/input/ts-exp-01"
]

len(experiments)


cat_c = ['dri_score','psych_disturb', 'cyto_score', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 'vent_hist',
 'renal_issue','pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type',
 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match',
 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group',
 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac','pulm_moderate']

f_fe = [
    'year_hct', 'dri_score_High', 'comorbidity_score', 'conditioning_intensity_None', 
    'karnofsky_score', 'donor_age', 'age_at_hct', 'mrd_hct_None', 
    'cyto_score_detail_Poor', 'dri_score_Intermediate', 'conditioning_intensity_RIC', 
    'cyto_score_Poor', 'hla_match_a_high', 'prim_disease_hct_ALL', 
    'gvhd_proph_FK+ MMF +- others', 
    'dri_score_High - TED AML case missing cytogenetics', 'sex_match_F-M', 
    'pulm_severe_Yes', 'cmv_status_-/+', 'hla_nmdp_6', 'cardiac_Yes', 
    'race_group_Black or African-American', 'sex_match_M-M', 'prim_disease_hct_AML', 
    'mrd_hct_Negative', 'donor_related_Related', 'hla_match_a_low', 
    'cyto_score_detail_None', 'cyto_score_Favorable', 'sex_match_M-F', 
    'arrhythmia_No', 'prior_tumor_No', 'in_vivo_tcd_Yes', 
    'race_group_More than one race', 'sex_match_F-F', 'hla_match_drb1_high', 
    'donor_related_Unrelated', 'tbi_status_No TBI', 'cyto_score_detail_Favorable', 
    'pulm_severe_No', 'tce_imm_match_None', 'mrd_hct_Positive', 
    'prim_disease_hct_MDS', 'diabetes_Yes', 'cmv_status_+/-', 
    'gvhd_proph_FKalone', 'prior_tumor_Not done', 'melphalan_dose_MEL', 
    'diabetes_No', 'arrhythmia_None', 'gvhd_proph_Cyclophosphamide +- others', 
    'hla_low_res_8', 'gvhd_proph_CSA + MMF +- others(not FK)', 'hepatic_severe_No', 
    'hla_low_res_6', 'graft_type_Bone marrow', 'cmv_status_+/+', 
    'prim_disease_hct_IEA', 'hla_match_dqb1_high', 'hla_match_dqb1_low', 
    'hla_match_b_low', 'dri_score_N/A - pediatric', 'dri_score_TBD cytogenetics', 
    'conditioning_intensity_MAC', 'obesity_No', 'tce_match_None', 
    'in_vivo_tcd_None', 'race_group_White', 'tce_div_match_None', 
    'hla_high_res_10', 'prod_type_BM', 'prim_disease_hct_IIS', 
    'hla_match_c_high', 'hla_match_c_low', 'prod_type_PB', 'hla_low_res_10', 
    'cyto_score_None', 'cmv_status_-/-', 'prior_tumor_Yes', 
    'conditioning_intensity_NMA', 'arrhythmia_Not done', 'cardiac_None',
    'tce_imm_match_G/G', 'prim_disease_hct_NHL', 'cyto_score_detail_Not tested', 
    'dri_score_Low', 'ethnicity_Not Hispanic or Latino', 'hla_match_b_high', 
    'race_group_Asian', 'melphalan_dose_N/A Mel not given', 'hepatic_mild_None', 
    'psych_disturb_No', 'tbi_status_TBI +- Other =cGy', 
    'cyto_score_detail_Intermediate', 'in_vivo_tcd_No', 'conditioning_intensity_TBD', 
    'hla_match_drb1_low', 'graft_type_Peripheral blood', 'hla_high_res_8', 
    'hla_high_res_6', 'prim_disease_hct_HIS', 'cyto_score_Intermediate', 
    'cyto_score_TBD', 'donor_related_Multiple donor (non-UCB)', 
    'pulm_moderate_Yes', 'tce_imm_match_P/P', 'tbi_status_TBI +- Other >cGy', 
    'vent_hist_Yes', 'tbi_status_TBI + Cy +- Other', 'tce_div_match_HvG non-permissive', 
    'cyto_score_detail_TBD', 'gvhd_proph_Cyclophosphamide alone', 
    'tce_div_match_Permissive mismatched', 'obesity_None', 'tce_match_Permissive', 
    'pulm_severe_None', 'rheum_issue_Yes', 'tce_div_match_GvH non-permissive', 
    'cardiac_No', 'dri_score_Very high', 'diabetes_Not done', 'rituximab_None', 
    'tce_match_GvH non-permissive', 'tce_imm_match_H/H', 'gvhd_proph_None', 
    'prim_disease_hct_SAA', 'rituximab_No', 'vent_hist_No', 'hepatic_severe_Yes', 
    'tce_imm_match_G/B', 'pulm_moderate_No', 'vent_hist_None', 
    'gvhd_proph_TDEPLETION alone', 'dri_score_N/A - non-malignant indication', 
    'race_group_Native Hawaiian or other Pacific Islander', 'prim_disease_hct_PCD', 
    'rheum_issue_Not done', 'cyto_score_Other', 'dri_score_None', 'ethnicity_None', 
    'dri_score_Intermediate - TED AML case missing cytogenetics', 
    'cmv_status_None', 'melphalan_dose_None', 'gvhd_proph_FK+ MTX +- others(not MMF)', 
    'psych_disturb_Yes', 'ethnicity_Hispanic or Latino', 'pulm_severe_Not done', 
    'renal_issue_None', 'peptic_ulcer_No', 'donor_related_None', 
    'prim_disease_hct_AI', 'tbi_status_TBI +- Other -cGy unknown dose', 
    'hepatic_severe_Not done', 'peptic_ulcer_None', 
    'tce_div_match_Bi-directional non-permissive', 'renal_issue_No', 
    'arrhythmia_Yes', 'tce_match_Fully matched', 'pulm_moderate_None', 
    'rituximab_Yes'
]


from tqdm import tqdm
from IPython.display import clear_output
from lifelines import KaplanMeierFitter
pd.options.display.max_columns = None
from lifelines.utils import concordance_index

SEED     = 114514
n_splits = 10

sp = '/kaggle/input/abdbase/AbdML/my_main.py'
tp = '/kaggle/working/main.py'

with open(sp, 'r', encoding='utf-8') as file:
    content = file.read()
with open(tp, 'w', encoding='utf-8') as file:
    file.write(content)

from main import AbdBase

def update(df):
    global cat_c

    for c in cat_c:
        df[c] = df[c].fillna('None').astype('category')

    j_ch=',[]{}:"\\<'
    for ch in j_ch:
        for c in cat_c:
            df[c] = df[c].apply(lambda x:str(x).replace(ch,''))
                
    return df

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf                    = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    return survival_probabilities

def update_target_with_survival_probabilities(df, time_col='efs_time', event_col='efs'):

    race_group          = sorted(df['race_group'].unique())
    survival_probs_dict = {}
    for race in race_group:
        race_df                   = df[df['race_group'] == race]
        survival_probs_dict[race] = transform_survival_probability(race_df, time_col, event_col)
    for race in race_group:
        df.loc[df['race_group'] == race, 'target'] = survival_probs_dict[race]
    df.loc[df[event_col] == 0, 'target'] -= 0.15
    
    return df


def c_index_score(modeloff, model_name, weights=None):
    y_true = train_solution 
    y_pred = train_solution[["ID"]].copy()

    if isinstance(modeloff, (list, tuple, np.ndarray)) and all(isinstance(m, np.ndarray) for m in modeloff):
        if weights is None:
            weights = [1] * len(modeloff)
        
        assert len(modeloff) == len(weights), "The number of models must match the number of weights."
        
        combined_modeloff = sum(weight * model for weight, model in zip(weights, modeloff))
        y_pred["prediction"] = combined_modeloff
    else:
        y_pred["prediction"] = modeloff

    c_index = base.CIBMTR_score(y_true.copy(), y_pred.copy(), "ID")
    print(Fore.YELLOW + f"The Score of {model_name} is: {c_index:.4f}")

def year_tf(df):
    df['cos_year'] = np.cos(df['year_hct'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year_hct'] * (2 * np.pi) / 100)
    return df



def get_abd_features(train, test):
    train = update(train)
    test  = update(test)

    train        = update_target_with_survival_probabilities(train, time_col='efs_time', event_col='efs')

    
    ohe_cols     = {'cat_c': cat_c}

    base         = AbdBase(
                            train_data=train, 
                            test_data=test, 
                            target_column='target',
                            gpu=False,
                            problem_type="regression", 
                            metric="mae", 
                            seed=SEED,
                            ohe_fe=ohe_cols,
                            n_splits=10,
                            early_stop=True,
                            num_classes=0,
                            cat_features=None,
                            fold_type='RKF'
                     )
    

    base.X_train = base.X_train[f_fe]
    base.X_test  = base.X_test[f_fe]
    
    base.X_train = year_tf(base.X_train)
    base.X_test  = year_tf(base.X_test)

    return base.X_test


class CFG:
    folds = 10

# https://www.kaggle.com/datasets/jsday96/mcts-tabm-models/data?select=TabMRegressor.py
class TabMRegressor:
    def __init__(
        self,
        arch_type: str        = 'tabm-mini',
        backbone: dict        = {'type': 'MLP', 'n_blocks': 3, 'd_block': 512, 'dropout': 0.1},
        d_embedding: int      = 64,  # Only used for 'tabm-mini'
        bin_count: int        = 48,  # Only used for 'tabm-mini'
        k: int                = 32,
        learning_rate: float  = 1e-4,
        weight_decay: float   = 1e-3,
        clip_grad_norm: bool  = True,
        max_epochs: int       = 100,
        patience: int         = 15,
        batch_size: int       = 32,
        compile_model: bool   = False,
        device: Optional[str] = 'cuda:0',
        random_state: int     = 0,
        verbose: bool         = True
    ):
        self.arch_type = arch_type
        self.backbone = backbone
        self.d_embedding = d_embedding
        self.bin_count = bin_count
        self.k = k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        eval_set: Tuple[pd.DataFrame, np.array]
    ):
        # PREPROCESS DATA.
        X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        # CREATE MODEL & TRAINING ALGO.
        bins = rtdl_num_embeddings.compute_bins(X_cont_train, n_bins=self.bin_count) if self.arch_type == 'tabm-mini' else None
        self.model = Model(
            n_num_features=X_cont_train.shape[1],
            cat_cardinalities=cat_cardinalities,
            n_classes=None,
            backbone=self.backbone,
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': self.d_embedding,
                    'activation': True,
                    'version': 'B',
                }
            ),
            arch_type=self.arch_type,
            k=self.k,
        ).to(self.device)
        optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.compile_model:
            self.model = torch.compile(self.model)

        loss_fn = torch.nn.MSELoss().to(self.device)
        # TRAIN & TEST MODEL.
        best = {
            'epoch': -1,
            'eval_loss': math.inf,
            'model_state_dict': None,
        }
        remaining_patience = self.patience
        epoch_size = math.ceil(len(X) / self.batch_size)


        for epoch in range(self.max_epochs):
            # TRAIN.
            optimizer.zero_grad()
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            progress_bar = tqdm(progress_bar, desc=f'Epoch {epoch}', total=epoch_size) if self.verbose else progress_bar
            for batch_idx in progress_bar:
                self.model.train()

                with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                    y_pred = self.model(
                        X_cont_train[batch_idx],
                        X_cat_train[batch_idx],
                    ).squeeze(-1).float()

                loss = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())


             # EVALUATE.
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_idx in torch.arange(0, len(y_val), self.batch_size, device=self.device):
                    y_pred = self.model(
                        X_cont_val[batch_idx:batch_idx+self.batch_size],
                        X_cat_val[batch_idx:batch_idx+self.batch_size],
                    ).squeeze(-1).float()

                    loss = loss_fn(y_pred.flatten(0, 1), y_val[batch_idx:batch_idx+self.batch_size].repeat_interleave(self.k))
                    val_losses.append(loss.item())


            # PRINT INFO.
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)
            if self.verbose:
                print(f'Epoch {epoch} | Train Loss: {mean_train_loss} | Val Loss: {mean_val_loss}')


            # COMPARE TO BEST.
            if mean_val_loss < best['eval_loss']:
                best['epoch'] = epoch
                best['eval_loss'] = mean_val_loss
                best['model_state_dict'] = self.model.state_dict()
                remaining_patience = self.patience
                
                if self.verbose:
                    print('ðŸŒ¸ New best epoch! ðŸŒ¸')
            else:
                remaining_patience -= 1

            # EARLY STOPPING.
            if remaining_patience == 0:
                break

            # RESTORE BEST MODEL.
            self.model.load_state_dict(best['model_state_dict'])


    def predict(
        self,
        X: pd.DataFrame,
        batch_size: Optional[int] = 8096
    ) -> np.ndarray:
        # PREPROCESS DATA.
        X_cat, X_cont, _, _ = self._preprocess_data(X, y=None, training=False)

        # PREDICT.
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
                y_pred.append(
                    self.model(
                        X_cont[batch_idx:batch_idx+batch_size],
                        X_cat[batch_idx:batch_idx+batch_size],
                    ).squeeze(-1).float().cpu().numpy()
                )

        y_pred = np.concatenate(y_pred)


        # DENORMALIZE TARGETS.
        y_pred = y_pred * self._target_std + self._target_mean


        # COMPUTE ENSEMBLE MEAN.
        y_pred = np.mean(y_pred, axis=1)

        return y_pred


    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, training: bool):
        # PICK NON-CONSTANT COLUMNS.
        if training:
            self._non_constant_columns = X.columns[X.nunique() > 1]

        X = X[self._non_constant_columns]

        # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
        categorical_features = [col for col in X.columns if X[col].dtype.name == 'object']
        X_cat = X[categorical_features].to_numpy()
        X_cont = X.drop(columns=categorical_features).to_numpy()

        # ENCODE CATEGORICAL FEATURES.
        cat_cardinalities = [X[col].nunique() for col in categorical_features]

        if training:
            self._categorical_encoders = [
                OrdinalEncoder()
                for _ in range(X_cat.shape[1])
            ]
        X_cat = np.concatenate([
            encoder.fit_transform(X_cat[:, i:i+1])
            for i, encoder in enumerate(self._categorical_encoders)
        ], axis=1)

        # NORMALIZE TARGETS.
        if training:
            self._target_mean = y.mean()
            self._target_std = y.std()

            y = (y - self._target_mean) / self._target_std


        # SCALE CONTINUOUS FEATURES.
        if training:
            noise = (
                np.random.default_rng(0)
                .normal(0.0, 1e-5, X_cont.shape)
                .astype(X_cont.dtype)
            )
            self._cont_feature_preprocessor = QuantileTransformer(
                n_quantiles=max(min(len(X) // 30, 1000), 10),
                output_distribution='normal',
                subsample=10**9,
            ).fit(X_cont + noise)

        X_cont = self._cont_feature_preprocessor.transform(X_cont)


        # CONVERT TO TENSORS.
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)

        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, cat_cardinalities, y



def get_tabm_features(data):
    RMV = ["ID","efs","efs_time","y","fold"]
    FEATURES = [c for c in data.columns if not c in RMV]
    
    RMV              = ['ID']
    X_test           = data.drop(RMV, axis=1)
    y_pred           = data[['ID']]
    
    #print("X_test shape:", X_test.shape, '\n')
    
    cat_cols         = X_test.select_dtypes(include=['object']).columns.tolist()
    num_cols         = X_test.select_dtypes(exclude=['object']).columns.tolist()
    
    # Preprocessing categorical
    imputer          = SimpleImputer(strategy='constant', fill_value='NAN')
    X_test[cat_cols] = imputer.fit_transform(X_test[cat_cols])

    # Preprocessing numerical
    imputer          = SimpleImputer(strategy="median")
    X_test[num_cols] = imputer.fit_transform(X_test[num_cols])

    return X_test,FEATURES


def prepare_features(model_path, train, test):

    RMV = ["ID","efs","efs_time","y"]
    FEATURES = [c for c in train.columns if not c in RMV]
    #print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    

    CATS = []
    for c in FEATURES:
        if train[c].dtype=="object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c]  = test[c].fillna("NAN")
        elif "DeepTabels" in model_path or "tn" in model_path or "svr" in model_path:
            print(f"preparing features for : {model_path}")
            train[c] = train[c].fillna(-1)
            test[c]  = test[c].fillna(-1)
            
        
    #print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
    
    combined = pd.concat([train,test],axis=0,ignore_index=True)
    #print("Combined data shape:", combined.shape )
    
    # LABEL ENCODE CATEGORICAL FEATURES
    #print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
    for c in FEATURES:
    
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            #print(f"{c}, ",end="")
            combined[c],_ = combined[c].factorize()
            combined[c]  -= combined[c].min()
            combined[c]   = combined[c].astype("int32")
            combined[c]   = combined[c].astype("category")
            
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype =="float64":
                combined[c]      = combined[c].astype("float32")
            if combined[c].dtype =="int64":
                combined[c]      = combined[c].astype("int32")
        
    train = combined.iloc[:len(train)].copy()
    test  = combined.iloc[len(train):].reset_index(drop=True).copy()
                
    return train, test, FEATURES


def get_lgbm_exp_08(train, test):
    RMV = ["ID","efs","efs_time","fold"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    for c in FEATURES:
        if train[c].dtype=="object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
            train[c] = train[c].astype('category')
            test[c] = test[c].astype('category')
        else:
            train[c] = train[c].fillna(-1)
            test[c] = test[c].fillna(-1) 

    return test, FEATURES


def get_nn_features(train, test):
    
    CAT_SIZE = []
    CAT_EMB  = []
    NUMS     = []
    CATS     = []

    RMV = ["ID","efs","efs_time","y","fold"]
    FEATURES = [c for c in train.columns if not c in RMV]
    
    for c in FEATURES:
        if train[c].dtype=="object":
            train[c] = train[c].fillna("NAN")
            test[c]  = test[c].fillna("NAN")
            CATS.append(c)
        elif not "age" in c:
            train[c] = train[c].astype("str")
            test[c]  = test[c].astype("str")
            CATS.append(c)


    combined = pd.concat([train,test],axis=0,ignore_index=True)
    for c in FEATURES:
        if c in CATS:
            # LABEL ENCODE
            combined[c],_ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            #combined[c] = combined[c].astype("category")

            n = combined[c].nunique()
            mn = combined[c].min()
            mx = combined[c].max()
            #print(f'{c} has ({n}) unique values')
    
            CAT_SIZE.append(mx+1) 
            CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
        else:
            if combined[c].dtype=="float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype=="int64":
                combined[c] = combined[c].astype("int32")
                
            m = combined[c].mean()
            s = combined[c].std()
            combined[c] = (combined[c]-m)/s
            combined[c] = combined[c].fillna(0)
            
            NUMS.append(c)

    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    return test[CATS], test[NUMS]


def get_tf_features(train, test):
    RMV = ["ID","efs","efs_time","y","y_na","fold"]
    FEATURES = [c for c in train.columns if not c in RMV]


    test                             = test.replace('Not done', 'missing')
    test                             = test.replace('Not tested', 'missing')
    
    test['na_count']                 = test.isna().sum(axis=1)
    test['age_karnofsky']            = test['age_at_hct'] * test['karnofsky_score']
    test['age_comorbidity']          = test['age_at_hct'] * test['comorbidity_score']
    test['donor_recipient_age_diff'] = abs(test['donor_age'] - test['age_at_hct'])
    test['hla_match_ratio']          = (test['hla_high_res_8'] + test['hla_low_res_8']) / 16
    test['age_squared']              = test['age_at_hct'] ** 2
    test['karnofsky_squared']        = test['karnofsky_score'] ** 2
    test['16?']                      = np.where(test['age_at_hct']<=16,1,0)
    
    FEATURES.extend(["na_count", "age_karnofsky", "age_comorbidity", "donor_recipient_age_diff", "hla_match_ratio", "age_squared", "karnofsky_squared", "16?"])

    CATS = []
    for c in FEATURES:
        if test[c].dtype=="object":
            CATS.append(c)
            test[c] = test[c].fillna("missing")

    for c in FEATURES:
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            #print(f"{c}, ",end="")
            test[c],_ = test[c].factorize()
            test[c]  -= test[c].min()
            test[c]   = test[c].astype("int32")
            test[c]   = test[c].astype("category")
        else:
            if test[c].dtype == "float64":
                test[c]      = test[c].astype("float32")
            if test[c].dtype =="int64":
                test[c]      = test[c].astype("int32")
    
    return test, FEATURES


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")
CATEGORICAL_VARIABLES = [
                            'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',
                            'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
                            'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
                            'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail',
                            'ethnicity', 'race_group', 'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match',
                            'tce_match', 'tce_div_match', 'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd',
                            'conditioning_intensity'
                        ]
HLA_COLUMNS = [
                    'hla_match_a_low', 'hla_match_a_high', 'hla_match_b_low', 'hla_match_b_high',
                    'hla_match_c_low', 'hla_match_c_high', 'hla_match_dqb1_low', 'hla_match_dqb1_high',
                    'hla_match_drb1_low', 'hla_match_drb1_high', 'hla_nmdp_6', 'hla_low_res_6',
                    'hla_high_res_6', 'hla_low_res_8', 'hla_high_res_8', 'hla_low_res_10', 'hla_high_res_10'
              ]
OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES       = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES
TARGET_VARIABLES          = ['efs_time', 'efs']
ID_COLUMN                 = ["ID"]

def load_and_preprocess_data(train, test):
    
    # Label encoding
    label_encoders = {}
    train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].fillna("Unknown")
    test[CATEGORICAL_VARIABLES]  = test[CATEGORICAL_VARIABLES].fillna("Unknown")

    for cat_var in CATEGORICAL_VARIABLES:
    # Get unique values from both train and test
        unique_values = pd.concat([train[cat_var], test[cat_var]]).unique()
        
        # Create and fit label encoder
        label_encoders[cat_var] = LabelEncoder()
        label_encoders[cat_var].fit(unique_values)
        
        # Transform both datasets
        train[cat_var] = label_encoders[cat_var].transform(train[cat_var])
        test[cat_var] = label_encoders[cat_var].transform(test[cat_var])
    
    # Fill numeric missing values
    train[NUMERICAL_VARIABLES] = train[NUMERICAL_VARIABLES].fillna(train[NUMERICAL_VARIABLES].median())
    test[NUMERICAL_VARIABLES]  = test[NUMERICAL_VARIABLES].fillna(test[NUMERICAL_VARIABLES].median())
    
    # Feature engineering
    train['year_hct'] = train['year_hct'] - 2000
    test['year_hct']  = test['year_hct'] - 2000
    
    return train, test

def prepare_fold_data(train_data, test_data):
    # Prepare standardization
    cols_standardize = NUMERICAL_VARIABLES
    cols_leave       = CATEGORICAL_VARIABLES
    standardize      = [([col], StandardScaler()) for col in cols_standardize]
    leave            = [(col, None) for col in cols_leave]
    x_mapper         = DataFrameMapper(standardize + leave)
    
    # Transform data
    x_train = x_mapper.fit_transform(train_data).astype('float32')
    x_test  = x_mapper.transform(test_data).astype('float32')
    
    # Get targets
    get_target = lambda df: (df['efs_time'].values, df['efs'].values)
    y_train = get_target(train_data)
    
    return x_test


def get_ds_features(train, test):
    x_test  = prepare_fold_data(train, test)
    return x_test

def create_model():
    in_features = len(NUMERICAL_VARIABLES) + len(CATEGORICAL_VARIABLES)
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False
    
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, 
                                 batch_norm, dropout, output_bias=output_bias)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    return model


from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import CoxPHFitter
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
import numpy as np

device="cuda"

def update_target_with_probabilities(df, probability_func, target_name, time_col='efs_time', event_col='efs', sep=0):
    race_group = sorted(df['race_group'].unique())
    probs_dict = {}
    
    # Compute probabilities for each race group
    for race in race_group:
        race_df = df[df['race_group'] == race]
        probs_dict[race] = probability_func(race_df, time_col, event_col)

     # Update target values using the target_name parameter
    for race in race_group:
        df.loc[df['race_group'] == race, target_name] = probs_dict[race]
    
    # Adjust target for non-events
    df.loc[df[event_col] == 0, target_name] -= sep
    
    return df[[event_col,target_name]]

# Other functions remain the same
def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def KaplanMeier(in_data, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=in_data[time_col], event_observed=in_data[event_col])
    return kmf.survival_function_at_times(in_data[time_col]).values.flatten()

def NelsonAalen(in_data, time_col='efs_time', event_col='efs'):
    naf = NelsonAalenFitter()
    naf.fit(durations=in_data[time_col], event_observed=in_data[event_col])
    return -1 * naf.cumulative_hazard_at_times(in_data[time_col]).values.flatten()

def chris_nn(df, time_col='efs_time', event_col='efs'):
    train = df.copy()
    train["y"] = train[time_col].values
    mx = train.loc[train[event_col]==1, time_col].max()
    mn = train.loc[train[event_col]==0, time_col].min()
    train.loc[train[event_col]==0, "y"] = train.loc[train[event_col]==0, "y"] + mx - mn
    train.y = train.y.rank()
    train.loc[train[event_col]==0, "y"] += 1*len(train)
    train.y = train.y / train.y.max()
    train.y = np.log(train.y)
    train.y -= train.y.mean()
    train.y *= -1.0
    return train.y.values


def compare_risk_transforms(df, time_col='efs_time', event_col='efs'):
    transforms = {}
    transform_configs = {
        'chris_nn_0': (chris_nn, 0),
        'chris_nn_0.3': (chris_nn, 0.3),
        'chris_nn_0.5': (chris_nn, 0.5),
        'chris_nn_0.6': (chris_nn, 0.6),
        'chris_nn_m0.3': (chris_nn, -0.3),
        
        'NelsonAalen_0': (NelsonAalen, 0),
        'NelsonAalen_005': (NelsonAalen, 0.05),
        'NelsonAalen_01': (NelsonAalen, 0.1),
        'NelsonAalen_015': (NelsonAalen, 0.15),
        'NelsonAalen_02': (NelsonAalen, 0.2),
        'NelsonAalen_025': (NelsonAalen, 0.25),

        'KaplanMeier_005': (KaplanMeier, 0.05),
        'KaplanMeier_01': (KaplanMeier, 0.1),
        'KaplanMeier_015': (KaplanMeier, 0.15),
        'KaplanMeier_02': (KaplanMeier, 0.2),
        'KaplanMeier_025': (KaplanMeier, 0.25),
    }
    
    # Create transforms with named targets
    for name, (func, sep) in transform_configs.items():
        transforms[name] = update_target_with_probabilities(
            df.copy(), func, target_name=name, sep=sep
        )
    
    return transforms


def get_tabm_features_02(train, test):

    transforms = compare_risk_transforms(train)

    #--------------------------------------------------------------------------------------------#
    target_names = []
    for i, (name, risk) in enumerate(transforms.items()):
        if name == 'KaplanMeier_015':
            train['label'] = risk[name]

    #--------------------------------------------------------------------------------------------#
    combined = pd.concat([train, test], axis=0)

    RMV      = ["ID","efs","efs_time", "label", "y", "efs_time2", "fold"]
    FEATURES = [c for c in train.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    
    train['isna'] = train.isna().sum(axis=1)
    test['isna']  = test.isna().sum(axis=1)
    #--------------------------------------------------------------------------------------------#

    CATS = []
    for c in FEATURES:
        num_unique = combined[c].nunique()
        if num_unique < 100:
            CATS.append(c)
            train[c] = train[c].fillna(999)
            test[c]  = test[c].fillna(999)
    print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
    
    NUMS = [c for c in FEATURES if not c in CATS]+['isna']
    #--------------------------------------------------------------------------------------------#

    combined = pd.concat([train,test],axis=0,ignore_index=True)

    # LABEL ENCODE CATEGORICAL FEATURES
    print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
    for c in FEATURES:
    
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            print(f"{c}, ",end="")
            combined[c],_ = combined[c].factorize()
            combined[c]  -= combined[c].min()
            combined[c]   = combined[c].astype("int32")
            combined[c]   = combined[c].astype("category")
            
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype == "float64":
                combined[c]      = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c]      = combined[c].astype("int32")
        
    cat_unique = combined[CATS].nunique().to_list()
    
    
    for c in NUMS:
        combined[c] = combined[c].fillna(0)
    
    train = combined.iloc[:len(train)].copy()
    test  = combined.iloc[len(train):].reset_index(drop=True).copy()
    
    #--------------------------------------------------------------------------------------------#
    
    cats_index  = [train[FEATURES].columns.get_loc(cat) for cat in CATS]
    
    scaler      = StandardScaler()
    train[NUMS] = scaler.fit_transform(train[NUMS])
    test[NUMS]  = scaler.transform(test[NUMS])

    #--------------------------------------------------------------------------------------------#
    X_num_test = test[NUMS].values
    X_cat_test = test[CATS].values

    test_dl    = DataLoader(TensorDataset(torch.tensor(X_num_test, dtype=torch.float32), torch.tensor(X_cat_test, dtype=torch.int64)), batch_size=1024, shuffle=False)

    return test_dl


import functools
from typing import List

import pytorch_lightning as pl
import numpy as np
import torch
from lifelines.utils import concordance_index
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_tabular.models.common.layers import ODST
from torch import nn
from pytorch_lightning.utilities import grad_norm


class CatEmbeddings(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        """
        projection_dim: The dimension of the final output after projecting the concatenated embeddings into a lower-dimensional space.
        categorical_cardinality: A list where each element represents the number of unique categories (cardinality) in each categorical feature.
        embedding_dim: The size of the embedding space for each categorical feature.
        self.embeddings: list of embedding layers for each categorical feature.
        self.projection: sequential neural network that goes from the embedding to the output projection dimension with GELU activation.
        """
        super(CatEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        """
        Apply the projection on concatened embeddings that contains all categorical features.
        """
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)


class NN(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        """
        continuous_dim: The number of continuous features.
        categorical_cardinality: A list of integers representing the number of unique categories in each categorical feature.
        embedding_dim: The dimensionality of the embedding space for each categorical feature.
        projection_dim: The size of the projected output space for the categorical embeddings.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        self.embeddings: previous embeddings for categorical data.
        self.mlp: defines an MLP model with an ODST layer followed by batch normalization and dropout.
        self.out: linear output layer that maps the output of the MLP to a single value
        self.dropout: defines dropout
        Weights initialization with xavier normal algorithm and biases with zeros.
        """
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        """
        Create embedding layers for categorical data, concatenate with continous variables.
        Add dropout and goes through MLP and return raw output and 1-dimensional output as well.
        """
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    """
    calculates all possible 2-combinations (pairs) of a tensor of indices from 0 to N-1, 
    and caches the result using functools.lru_cache for optimization
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class LitNN(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.1,
            margin: float = 0.5,
            race_index: int = 0
    ):
        """
        continuous_dim: The number of continuous input features.
        categorical_cardinality: A list of integers, where each element corresponds to the number of unique categories for each categorical feature.
        embedding_dim: The dimension of the embeddings for the categorical features.
        projection_dim: The dimension of the projected space after embedding concatenation.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        aux_weight: Weight used for auxiliary tasks.
        margin: Margin used in some loss functions.
        race_index: An index that refer to race_group in the input data.
        """
        super(LitNN, self).__init__()
        self.save_hyperparameters()

        # Creates an instance of the NN model defined above
        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []

        # Defines a small feedforward neural network that performs an auxiliary task with 1-dimensional output
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        """
        Output loss and race_group loss.
        """
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
        """
        Most important part of the model : loss function used for training.
        We face survival data with event indicators along with time-to-event.

        This function computes the main loss by the following the steps :
        * create all data pairs with "combinations" function (= all "two subjects" combinations)
        * make sure that we have at least 1 event in each pair
        * convert y to +1 or -1 depending on the correct ranking
        * loss is computed using a margin-based hinge loss
        * mask is applied to ensure only valid pairs are being used (censored data can't be ranked with event in some cases)
        * average loss on all pairs is returned
        """
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Defines all invalid comparisons :
        * Case 1: "Left outlived Right" but Right is censored
        * Case 2: "Right outlived Left" but Left is censored
        Masks for case 1 and case 2 are combined using |= operator and inverted using ~ to create a "valid pair mask"
        """
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        """
        This method defines how the model processes each batch during validation
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate c-index accounting for each race_group or global.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        """
        Calculate c-index accounting for each race_group
        """
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()


    def configure_optimizers(self):
        """
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

def get_nn_exp_04_model(shape, transformers, categorical_cols):
    hparams = {
            "embedding_dim"    : 16,
            "projection_dim"   : 112,
            "hidden_dim"       : 56,
            "lr"               : 0.06464861983337984,
            "dropout"          : 0.05463240181423116,
            "aux_weight"       : 0.26545778308743806,
            "margin"           : 0.2588153271003354,
            "weight_decay"     : 0.0002773544957610778
        }
    model = LitNN(
        continuous_dim          = shape,
        categorical_cardinality = [len(t.classes_) for t in transformers],
        race_index              = categorical_cols.index("race_group"),
        **hparams
    )
    return model


import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
from warnings import filterwarnings

filterwarnings('ignore')

def get_feature_types(train):
    """
    Utility function to return categorical and numerical column names.
    """
    categorical_cols = [col for i, col in enumerate(train.columns) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]

    if "fold" in categorical_cols:
        categorical_cols.remove("fold")
        print("removing fold column")
    RMV              = ["ID", "efs", "efs_time", "y", "fold"]
    FEATURES         = [c for c in train.columns if not c in RMV]
    numerical        = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical

def init_dl(X_cat, X_num, df, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
    return dl_train


def get_X_cat(df, cat_cols, transformers=None):
    """
    Apply a specific categorical data transformer or a LabelEncoder if None.
    """
    if transformers is None:
        transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
    return transformers, np.array(
        [transformer.transform(df[col]) for col, transformer in zip(cat_cols, transformers)]
    ).T

def get_categoricals(train, val):
    """
    Remove constant categorical columns and transform them using LabelEncoder.
    Return the label-transformers for each categorical column, categorical dataframes and numerical columns.
    """
    categorical_cols, numerical = get_feature_types(train)
    remove = []
    for col in categorical_cols:
        if train[col].nunique() == 1:
            remove.append(col)
        ind = ~val[col].isin(train[col])
        if ind.any():
            val.loc[ind, col] = np.nan
    categorical_cols = [col for col in categorical_cols if col not in remove]
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers

def preprocess_data(train, val):
    """
    Standardize numerical variables and transform (Label-encode) categoricals.
    Fill NA values with mean for numerical.
    Create torch dataloaders to prepare data for training and evaluation.
    """
    X_cat_train, X_cat_val, numerical, transformers = get_categoricals(train, val)
    scaler                                          = StandardScaler()
    imp                                             = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    X_num_train                                     = imp.fit_transform(train[numerical])
    X_num_train                                     = scaler.fit_transform(X_num_train)
    X_num_val                                       = imp.transform(val[numerical])
    X_num_val                                       = scaler.transform(X_num_val)
    dl_train                                        = init_dl(X_cat_train, X_num_train, train, training=True)
    dl_val                                          = init_dl(X_cat_val, X_num_val, val)
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers

def add_features_04(df):
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['year_hct'] -= 2000
    return df

def get_nn_exp_04_features(train, test):
    train = add_features_04(train)
    test = add_features_04(test)
    test['efs_time']             = 1
    test['efs']                  = 1
    categorical_cols, numerical  = get_feature_types(train)

    X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
    
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols


import functools
from typing import List

import pytorch_lightning as pl
import numpy as np
import torch
from lifelines.utils import concordance_index
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_tabular.models.common.layers import ODST
from torch import nn
from pytorch_lightning.utilities import grad_norm


class CatEmbeddings(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        super(CatEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)



class NN(nn.Module):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class FocalPairwiseLoss(nn.Module):
    """
    Focal loss adaptation for pairwise ranking in survival analysis.
    """
    def __init__(self, margin: float = 0.5, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            margin: Margin for the hinge loss component
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            alpha: Balancing parameter for positive/negative pairs
        """
        super(FocalPairwiseLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_diff, y, mask):
        """
        Args:
            pred_diff: Difference in predictions between pairs
            y: Ground truth (-1 or 1) indicating correct ranking
            mask: Valid pairs mask
        """
        # Calculate base hinge loss
        base_loss = nn.functional.relu(-y * pred_diff + self.margin)
        
        # Calculate probability of correct ranking
        prob_correct = torch.sigmoid(y * pred_diff)
        
        # Calculate focal weights
        focal_weight = (1 - prob_correct) ** self.gamma
        
        # Apply class balancing
        alpha_weight = torch.where(y > 0, self.alpha, 1 - self.alpha)
        
        # Combine all components
        loss = focal_weight * alpha_weight * base_loss
        
        # Apply mask and average
        loss = (loss.double() * mask.double()).sum() / mask.sum()
        return loss


class LitNN(pl.LightningModule):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.1,
            margin: float = 0.5,
            focal_gamma: float = 2.0,
            focal_alpha: float = 0.25,
            race_index: int = 0
    ):
        super(LitNN, self).__init__()
        self.save_hyperparameters()

        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []

        # Initialize focal loss
        self.focal_loss = FocalPairwiseLoss(
            margin=self.hparams.margin,
            gamma=self.hparams.focal_gamma,
            alpha=self.hparams.focal_alpha
        )

        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def calc_loss(self, y, y_hat, efs):
        """
        Updated loss calculation using focal loss
        """
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y_target = 2 * (y_left > y_right).int() - 1
        pred_diff = pred_left - pred_right
        mask = self.get_mask(comb, efs, y_left, y_right)
        return self.focal_loss(pred_diff, y_target, mask)

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        """
        Output loss and race_group loss.
        """
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
        """
        Most important part of the model : loss function used for training.
        We face survival data with event indicators along with time-to-event.

        This function computes the main loss by the following the steps :
        * create all data pairs with "combinations" function (= all "two subjects" combinations)
        * make sure that we have at least 1 event in each pair
        * convert y to +1 or -1 depending on the correct ranking
        * loss is computed using a margin-based hinge loss
        * mask is applied to ensure only valid pairs are being used (censored data can't be ranked with event in some cases)
        * average loss on all pairs is returned
        """
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Defines all invalid comparisons :
        * Case 1: "Left outlived Right" but Right is censored
        * Case 2: "Right outlived Left" but Left is censored
        Masks for case 1 and case 2 are combined using |= operator and inverted using ~ to create a "valid pair mask"
        """
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        """
        This method defines how the model processes each batch during validation
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate c-index accounting for each race_group or global.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        """
        Calculate c-index accounting for each race_group
        """
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()


    def configure_optimizers(self):
        """
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def get_nn_exp_05_model(shape, transformers, categorical_cols):
    hparams = {
            "embedding_dim"    : 16,
            "projection_dim"   : 112,
            "hidden_dim"       : 56,
            "lr"               : 0.06464861983337984,
            "dropout"          : 0.05463240181423116,
            "aux_weight"       : 0.26545778308743806,
            "margin"           : 0.2588153271003354,
            "weight_decay"     : 0.0002773544957610778
        }
    model = LitNN(
        continuous_dim          = shape,
        categorical_cardinality = [len(t.classes_) for t in transformers],
        race_index              = categorical_cols.index("race_group"),
        **hparams
    )
    return model


import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
from warnings import filterwarnings

filterwarnings('ignore')

def get_feature_types(train):
    """
    Utility function to return categorical and numerical column names.
    """
    categorical_cols = [col for i, col in enumerate(train.columns) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]

    if "fold" in categorical_cols:
        categorical_cols.remove("fold")
        print("removing fold column")
    RMV              = ["ID", "efs", "efs_time", "y", "fold"]
    FEATURES         = [c for c in train.columns if not c in RMV]
    numerical        = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical

def init_dl(X_cat, X_num, df, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
    return dl_train


def get_X_cat(df, cat_cols, transformers=None):
    """
    Apply a specific categorical data transformer or a LabelEncoder if None.
    """
    if transformers is None:
        transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
    return transformers, np.array(
        [transformer.transform(df[col]) for col, transformer in zip(cat_cols, transformers)]
    ).T

def get_categoricals(train, val):
    """
    Remove constant categorical columns and transform them using LabelEncoder.
    Return the label-transformers for each categorical column, categorical dataframes and numerical columns.
    """
    categorical_cols, numerical = get_feature_types(train)
    remove = []
    for col in categorical_cols:
        if train[col].nunique() == 1:
            remove.append(col)
        ind = ~val[col].isin(train[col])
        if ind.any():
            val.loc[ind, col] = np.nan
    categorical_cols = [col for col in categorical_cols if col not in remove]
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers

def preprocess_data(train, val):
    """
    Standardize numerical variables and transform (Label-encode) categoricals.
    Fill NA values with mean for numerical.
    Create torch dataloaders to prepare data for training and evaluation.
    """
    X_cat_train, X_cat_val, numerical, transformers = get_categoricals(train, val)
    scaler                                          = StandardScaler()
    imp                                             = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    X_num_train                                     = imp.fit_transform(train[numerical])
    X_num_train                                     = scaler.fit_transform(X_num_train)
    X_num_val                                       = imp.transform(val[numerical])
    X_num_val                                       = scaler.transform(X_num_val)
    dl_train                                        = init_dl(X_cat_train, X_num_train, train, training=True)
    dl_val                                          = init_dl(X_cat_val, X_num_val, val)
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers

def add_features_04(df):
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['year_hct'] -= 2000
    return df

def get_nn_exp_05_features(train, test):
    train = add_features_04(train)
    test = add_features_04(test)
    test['efs_time']             = 1
    test['efs']                  = 1
    categorical_cols, numerical  = get_feature_types(train)

    X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
    
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols


CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail', 

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match',

    # Medication/operation related data
    'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 'conditioning_intensity'
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',
    
    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]


OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


def preprocess_data_06(df):
    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
    df[OTHER_NUMERICAL_VARIABLES] = df[OTHER_NUMERICAL_VARIABLES].fillna(df[OTHER_NUMERICAL_VARIABLES].median())

    return df

def processing_06(train, test):

        
    train = preprocess_data_06(train)
    test = preprocess_data_06(test)
    
    def features_engineering(df):
        # Change year_hct to relative year from 2000
        df['year_hct'] = df['year_hct'] - 2000
        
        return df
    
    train = features_engineering(train)
    test = features_engineering(test)
    
    train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
    test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')
    
    FEATURES = train.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()

    return test, FEATURES


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from pycox.preprocessing.label_transforms import LabTransCoxTime
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models.loss import CoxPHLoss
import torch
import torchtuples as tt


def get_nn_exp_02_model(in_features=213):

    net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1)
        )
    model = CoxPH(net, tt.optim.Adam(weight_decay=1e-4))

    return model

def preprocess_all_data(train_df, test_df, numerical_columns, categorical_columns):
    """Preprocess all data using the same transformation"""
    # Initialize the preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Not Available")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )
    
    # Fit preprocessor on entire training data
    train_features = train_df[["ID"] + numerical_columns + categorical_columns]
    preprocessor.fit(train_features)
    
    # Transform training and test data
    X_train_processed = preprocessor.transform(train_features)
    X_test_processed = preprocessor.transform(test_df[["ID"] + numerical_columns + categorical_columns])

    # Convert to DataFrames with feature names
    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=train_features.index
    ).astype("float32")
    
    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    ).astype("float32")
    
    return X_train_processed, X_test_processed, preprocessor


def get_nn_exp_02_features(train, test):
    FOLDS           = 10
    data_dictionary = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
    column_event    = "efs"
    column_duration = "efs_time"
    
    categorical_columns = data_dictionary.loc[data_dictionary["type"]=="Categorical","variable"].to_list()
    numerical_columns   = data_dictionary.loc[data_dictionary["type"]=="Numerical","variable"].to_list()
    categorical_columns.remove(column_event)
    numerical_columns.remove(column_duration)
    
    print("Data loaded succesfully!")

    # Preprocess all data first
    X_train_processed, X_test_processed, preprocessor = preprocess_all_data(
        train_df, test_df, numerical_columns, categorical_columns
    )

    return X_test_processed


hct_ci_mapping = {
    "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},  
    "cardiac": {"No": 0, "Not done": 0, "Yes": 1}, 
    "diabetes": {"No": 0, "Not done": 0, "Yes": 1},  
    "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
    "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    "psych_disturb": {"No": 0, "Not done": 0, "Yes": 1}, 
    "obesity": {"No": 0, "Not done": 0, "Yes": 1}, 
    "rheum_issue": {"No": 0, "Not done": 0, "Yes": 2},
    "peptic_ulcer": {"No": 0, "Not done": 0, "Yes": 2},  
    "renal_issue": {"No": 0, "Not done": 0, "Yes": 2}, 
    "prior_tumor": {"No": 0, "Not done": 0, "Yes": 3}, 
    "pulm_moderate": {"No": 0, "Not done": 0, "Yes": 2}, 
    "pulm_severe": {"No": 0, "Not done": 0, "Yes": 3},  
}

def calculate_hct_ci_score(row, mapping):
    """
    This function calculates the hct_ci score

    Args:
        row (pd.Series): Patient Clinical Data
        mapping (dict): HCT-CI score mapping

    Returns:
        int: HCT-CI score
    """

    score = 0
    if "hepatic_severe" in row and row["hepatic_severe"] == "Yes":
        score += mapping["hepatic_severe"]["Yes"]
    elif "hepatic_mild" in row and row["hepatic_mild"] == "Yes":
        score += mapping["hepatic_mild"]["Yes"]
    if "pulm_moderate" in row and row["pulm_moderate"] == "Yes":
        score += mapping["pulm_moderate"]["Yes"]
    elif "pulm_severe" in row and row["pulm_severe"] == "Yes":
        score += mapping["pulm_severe"]["Yes"]

    # Other Conditions
    for condition, mapping_values in mapping.items():
        if condition not in ["hepatic_mild", "hepatic_severe","pulm_moderate", "pulm_severe"] and condition in row:
            score += mapping_values.get(row[condition], 0)

    return score


def cat2num(df):
    df['conditioning_intensity'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None})
    
    df['tbi_status'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None})

    df['dri_score'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0})
    
    df['cyto_score'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None})

    df['cyto_score_detail'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None})
    
    return df

def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']):
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low'] 

def add_features_05(df):
    df["hct_ci_score"]             = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df                             = cat2num(df)
    df['hla_combined_low']         = df['hla_low_res_10']
    df['hla_combined_low']         = df.apply(fill_hla_combined_low, axis=1)
    df['hla_match_ratio']          = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    df['years_since_2000']         = df['year_hct'] - 2000
    df['null_count']               = df.isnull().sum(axis=1)
    df['ci_score_danger']          = df['hct_ci_score'].apply(lambda x: 2 if x >= 3 else 1 if x >= 1 else 0)
    return df


def get_exp_05_features(train, test, model_path):

    RMV = ["ID","efs","efs_time","y", "fold"]
    FEATURES = [c for c in train.columns if not c in RMV]

    train = add_features_05(train)
    test = add_features_05(test)

    CATS = []
    for c in FEATURES:
        if train[c].dtype=="object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
        elif "rf-exp-05" in model_path:
            train[c] = train[c].fillna(-1)
            test[c] = test[c].fillna(-1)

    combined = pd.concat([train,test],axis=0,ignore_index=True)

    for c in FEATURES:
    
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            #print(f"{c}, ",end="")
            combined[c],_ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
            
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype=="float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype=="int64":
                combined[c] = combined[c].astype("int32")
    
    # for c in cat2num:
    #     combined[c] = combined[c].astype("int32")
    
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    FEATURES += ["hct_ci_score", 'donor_recipient_age_diff', "hla_combined_low", "hla_match_ratio", 
             "years_since_2000", "null_count","ci_score_danger"]

    return train, test, FEATURES, CATS


def get_tn_exp_02_features(train, test):
    RMV      = ["ID","efs","efs_time","y", "fold"]
    FEATURES = [c for c in train.columns if not c in RMV]

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    CATS = []
    CAT_SIZE = []
    CAT_EMB = []
    NUMS = []
    
    for c in FEATURES:
        if train[c].dtype == "object":
            combined[c] = combined[c].fillna("NAN")
            CATS.append(c)
        elif "age" not in c:
            combined[c] = combined[c].astype("str")
            CATS.append(c)

    for c in CATS:
        combined[c], _ = combined[c].factorize(sort=True)
        combined[c] = combined[c].astype("int32")

        unique_vals = combined[c].nunique()
        CAT_SIZE.append(unique_vals + 1)  
        CAT_EMB.append(int(np.ceil(np.sqrt(unique_vals + 1))))  

    for c in FEATURES:
        if c not in CATS:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
            
            m = combined[c].mean()
            s = combined[c].std()
            combined[c] = (combined[c] - m) / s
            combined[c] = combined[c].fillna(0)
            
            NUMS.append(c)

    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    categorical_dims = {f: size for f, size in zip(CATS, CAT_SIZE)}
    cat_dims = [categorical_dims[f] for f in CATS]
    cat_emb_dim = [min(50, (dim + 1) // 2) for dim in cat_dims]
    cat_idxs = [i for i, f in enumerate(FEATURES) if f in CATS]

    return test, FEATURES


def cyto_score(x):
    if x in ['TBD', 'Not tested', 'Other']:
        return 'No info'
    else:
        return x

def dri_score(x):
    if x in ['TBD cytogenetics','N/A - disease not classifiable','Missing disease status']:
        return 'No info'
    else:
        return x

def tbi_status(x):
    if x in ['TBI +- Other, -cGy, single','TBI +- Other, -cGy, fractionated','TBI +- Other, -cGy, unknown dose ','TBI +- Other, unknown dose']:
        return 'No test'
    else:
        return x

def cyto_score_detail(x):
    if x in ['TBD','Not tested']:
        return 'No info'
    else: 
        return x

def conditioning_intensity(x):
    if x in ['TBD','No drugs reported','N/A, F(pre-TED) not submitted']:
        return 'No info'
    else:
        return x


from sklearn.preprocessing import LabelEncoder
def transform(df, cat_cols, num_cols, OneHotList, NoOneHotList, train=True):
    df_NoOneHot={}
    
    df_NoOneHot['cyto_score']=df['cyto_score'].apply(cyto_score)
    df_NoOneHot['dri_score']=df['dri_score'].apply(dri_score)
    df_NoOneHot['tbi_status']=df['tbi_status'].apply(tbi_status)
    df_NoOneHot['cyto_score_detail']=df['cyto_score_detail'].apply(cyto_score_detail)
    df_NoOneHot['conditioning_intensity']=df['conditioning_intensity'].apply(conditioning_intensity)
    df_NoOneHot['cmv_status']=df['cmv_status']
    df_NoOneHot['tce_imm_match']=df['tce_imm_match']
    df_NoOneHot['tce_match']=df['tce_match']
    df_NoOneHot['sex_match']=df['sex_match']
    df_NoOneHot['melphalan_dose']=df['melphalan_dose']

    le=LabelEncoder()

    df_encoded = pd.DataFrame()
    df_onehot = pd.DataFrame()
    for column in NoOneHotList:
        df_NoOneHot[column] =  le.fit_transform(df_NoOneHot[column]) 
    
        df_encoded[column]=df_NoOneHot[column]

    for column in OneHotList:
        df_onehot[column]= le.fit_transform(df[column])

    if train:
        df_final=pd.concat([df_onehot,df_encoded,df[num_cols]],axis=1)
        naf = NelsonAalenFitter()
        
        naf.fit(df_final['efs_time'], df['efs'])
        df_final['naf_label'] = -naf.cumulative_hazard_at_times(df_final['efs_time']).values
        df_final.loc[df_final['efs'] == 0, 'naf_label'] -= 0.2
        
        kmf = KaplanMeierFitter()
        kmf.fit(df_final['efs_time'], df_final['efs'])
        df_final['km_label'] = kmf.survival_function_at_times(df_final['efs_time']).values
        df_final.loc[df_final['efs'] == 0, 'km_label'] -= 0.2

    else:
        test_cols = [x for x in list(num_cols) if x not in ['efs','efs_time']]
        df_final=pd.concat([df_onehot,df_encoded,df[test_cols]],axis=1)  

    df_final=df_final.drop('ID',axis=1)

    return df_final
    
def get_vr_exp_01_features(train, test_df):
    target_cols  = ['efs', 'efs_time', 'km_label', 'naf_label', 'fold']
    NoOneHotList=['dri_score','cyto_score','tbi_status','cmv_status','tce_imm_match','cyto_score_detail','conditioning_intensity','tce_match','sex_match','melphalan_dose']
    
    cat_cols = train.select_dtypes(include=['object']).columns
    num_cols = train.select_dtypes(include=['int64','float64']).columns

    NoOneHotList = ['dri_score','cyto_score','tbi_status','cmv_status','tce_imm_match','cyto_score_detail','conditioning_intensity','tce_match','sex_match','melphalan_dose']
    OneHotList   = [column for column in train.columns if column not in NoOneHotList and column in cat_cols]

    if 'gvhd_proph' in OneHotList:
        OneHotList.remove('gvhd_proph')

    train=transform(train, cat_cols, num_cols, OneHotList, NoOneHotList, train=True)
    test=transform(test_df, cat_cols, num_cols, OneHotList, NoOneHotList, train=False)

    X_test            = test.drop(columns=target_cols, errors='ignore')
    # cat_cols = list[cat_cols]
    # if 'gvhd_proph' in cat_cols:
    #     cat_cols.remove('gvhd_proph')
   
    # X_test[cat_cols]  = X_test[cat_cols].astype('category')

    return X_test


import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, num_categories, num_cont, dim=64, depth=6, heads=8, dropout=0.1):
        super().__init__()
        
        # Store dimensions
        self.num_categories = len(num_categories)  # 35 categorical features
        self.dim = dim
        
        # Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, dim) for n_cat in num_categories
        ])
        
        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        # Numerical Feature Processing
        self.num_bn = nn.BatchNorm1d(num_cont)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_cont, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Calculate combined dimension
        # After mean pooling categorical features: dim
        # After processing numerical features: dim
        total_dim = dim * 2  # Concatenated dimension
        
        # Combined MLP with corrected input dimension
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x_cat, x_num):
        # Process categorical features
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))
        x_cat = torch.stack(embeddings, 1)  # [batch_size, num_cat, dim]
        
        # Apply transformer and pool
        x_cat = self.transformer(x_cat)      # [batch_size, num_cat, dim]
        x_cat = x_cat.mean(dim=1)           # [batch_size, dim]
        
        # Process numerical features
        x_num = self.num_bn(x_num)
        x_num = self.num_mlp(x_num)         # [batch_size, dim]
        
        # Combine features
        x = torch.cat([x_cat, x_num], dim=1)  # [batch_size, dim*2]
        
        # Final prediction
        return self.mlp(x)


def get_tt_exp_01_model(num_categories,num_features):
    model = TabTransformer(
            num_categories=num_categories,
            num_cont=len(num_features),
            dim=64,
            depth=6,
            heads=8
        ).to("cuda")

    return model

# Custom Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, categorical, numerical, targets):
        self.categorical = torch.tensor(categorical, dtype=torch.long)
        self.numerical   = torch.tensor(numerical, dtype=torch.float32)
        self.targets     = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.categorical[idx], self.numerical[idx], self.targets[idx]

def get_tt_exp_01_features(train, test):

    RMV = ["ID","efs","efs_time","y","fold"]
    FEATURES = [c for c in train.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

    CATS = []
    NUMS = []
    for c in FEATURES:
        if train[c].dtype=="object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
        else:
            NUMS.append(c)
            train[c] = train[c].fillna(0)
            test[c] = test[c].fillna(0)
            
    print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
    print(f"In these features, there are {len(NUMS)} NUMERICAL FEATURES: {NUMS}")

    combined = pd.concat([train,test],axis=0,ignore_index=True)
    #print("Combined data shape:", combined.shape )
    
    # LABEL ENCODE CATEGORICAL FEATURES
    print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
    for c in FEATURES:
    
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            print(f"{c}, ",end="")
            combined[c],_ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
            
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype=="float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype=="int64":
                combined[c] = combined[c].astype("int32")
        
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()
    

    test_ds = MedicalDataset(
            categorical=test[CATS].values,
            numerical=test[NUMS].values,
            targets=np.zeros(len(test))  # Dummy targets
        )
    test_loader = DataLoader(test_ds, batch_size=64*2)

    num_categories = [train[col].nunique() for col in CATS]

    return test_loader, num_categories, NUMS


import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SurvivalDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32) if targets is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx].to(device), self.targets[idx].to(device)
        return self.features[idx].to(device)


from torch import nn

class MCDropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout_prob=0.3):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layers       = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


def get_mcts_exp_02_features(train, test):
    RMV = ["ID","efs","efs_time","y","fold"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    NUMS = []
    for c in FEATURES:
        if train[c].dtype=="object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
        else:
            NUMS.append(c)
            train[c] = train[c].fillna(-1)
            test[c] = test[c].fillna(-1)


    combined = pd.concat([train,test],axis=0,ignore_index=True)

    for c in FEATURES:
        if c in CATS:
            combined[c],_ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
        else:
            if combined[c].dtype=="float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype=="int64":
                combined[c] = combined[c].astype("int32")
    
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    return test, FEATURES


from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_rf_encoders(train, test, FEATURES, CATS):

    # Initialize imputers
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Separate numerical and categorical features
    num_features = [f for f in FEATURES if f not in CATS]
    X_train_encoded = train[FEATURES].copy()
    X_test_encoded = test[FEATURES].copy()
    
    # Handle numerical features
    X_train_encoded[num_features] = num_imputer.fit_transform(train[num_features])
    X_test_encoded[num_features] = num_imputer.transform(test[num_features])
    
    # Handle categorical features
    label_encoders = {}

    for cat in CATS:
        # First impute missing values
        X_train_encoded[cat] = cat_imputer.fit_transform(train[FEATURES][cat].values.reshape(-1, 1)).ravel()
        X_test_encoded[cat] = cat_imputer.transform(test[FEATURES][cat].values.reshape(-1, 1)).ravel()
        
        # Then encode categories
        label_encoders[cat] = LabelEncoder()
        X_train_encoded[cat] = label_encoders[cat].fit_transform(X_train_encoded[cat])
        X_test_encoded[cat] = label_encoders[cat].transform(X_test_encoded[cat])

    return X_test_encoded


def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match.astype("object")
    return df

def get_xgboost_exp_09_features(train, test):

    test       = add_features(test)
    train      = add_features(train)

    RMV        = ["ID", "efs", "efs_time", "y", "fold"]
    FEATURES   = [c for c in train.columns if not c in RMV]
    
    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    for c in FEATURES:
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
    
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
                
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    return test, FEATURES


def add_features_xgboost_exp_10(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match.astype("object")
    return df
    
def get_xgboost_exp_10_features(train, test):
    test = add_features_xgboost_exp_10(test)
    train = add_features_xgboost_exp_10(train)
    train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

    
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for c in FEATURES:

        if c in CATS:
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")

        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()

    return FEATURES, test


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import mean_squared_error
import pandas as pd
import math

channel1_columns = ["ethnicity", "race_group","sex_match", "donor_related", 'age_at_hct', 'donor_age', 'year_hct']
                    #'sex_match_bool'
channel2_columns = ['cmv_status', 'cyto_score', 'cyto_score_detail', 
                    'mrd_hct', 'hct_ci_score', 'dri_score', 'mrd_cyto']
                    #'hla_combined_low'
channel3_columns = ['psych_disturb', 'diabetes','arrhythmia', 'vent_hist', 'renal_issue', 'pulm_severe','obesity', 
                    'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'rheum_issue', 'hepatic_mild', 'cardiac', 'pulm_moderate']
channel4_columns = ['rituximab', 'conditioning_intensity', 'tbi_status', 'prod_type', 'in_vivo_tcd', 'melphalan_dose']

def create_channel_from_df(df, columns, target_size):
    h, w = target_size
    values = df[columns].values.flatten()  
    channel = np.zeros((h, w))  

    for i, value in enumerate(values):
        row, col = divmod(i, w)  
        if row < h:
            channel[row, col] = value
    
    return channel

# 1. Add data validation and preprocessing
def pad_channel(channel, target_size=4):
    h, w = channel.shape
    padded = np.zeros((target_size, target_size))
    padded[:h, :w] = channel
    return padded

# 2. Add NaN checking functions
def check_for_nans(tensor_or_array, name):
    if isinstance(tensor_or_array, torch.Tensor):
        has_nan = torch.isnan(tensor_or_array).any().item()
    else:
        has_nan = np.isnan(tensor_or_array).any()
    
    if has_nan:
        print(f"WARNING: {name} contains NaN values!")
    return has_nan

# 3. Modified Dataset class with data validation
class ChannelDataset(Dataset):
    def __init__(self, df, test_mode=False):
        # Validate and clean data
        df_clean = df.copy()
        
        # Check for NaNs in the input dataframe
        if df_clean.isnull().any().any():
            print(f"WARNING: Input dataframe contains {df_clean.isnull().sum().sum()} NaN values")
            # Fill NaN values with 0 (or another appropriate strategy)
            df_clean = df_clean.fillna(0)
        
        # Process channels (assuming channel columns are defined elsewhere)
        self.channel1 = np.array([pad_channel(create_channel_from_df(df_clean.loc[i, :], channel1_columns, (4, 4))) for i in df_clean.index])
        self.channel2 = np.array([pad_channel(create_channel_from_df(df_clean.loc[i, :], channel2_columns, (4, 4))) for i in df_clean.index])
        self.channel3 = np.array([pad_channel(create_channel_from_df(df_clean.loc[i, :], channel3_columns, (4, 4))) for i in df_clean.index])
        self.channel4 = np.array([pad_channel(create_channel_from_df(df_clean.loc[i, :], channel4_columns, (4, 4))) for i in df_clean.index])
        
        self.test_mode = test_mode
        if not test_mode:
            self.y = df_clean["y"].values.astype(np.float32)
    
    def __len__(self):
        return len(self.channel1)
    
    def __getitem__(self, idx):
        combined_channels = np.stack([self.channel1[idx], self.channel2[idx], self.channel3[idx], self.channel4[idx]], axis=0)
        x = torch.tensor(combined_channels, dtype=torch.float32)
        if self.test_mode:
            return x
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y

# 4. Improved model with proper initialization and numerical stability
class SurvivalRegressionCNN(nn.Module):
    def __init__(self):
        super(SurvivalRegressionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout rate
            nn.Linear(128, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']): 
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low'] 


def cyto_score_sum(df):
    def compute_cyto_score_sum(row):
        if row["cyto_score_detail"] in ["Favorable", "Intermediate", "TBD", "Poor", "Not tested"]:
            return row["cyto_score_detail"]
        elif row["cyto_score"] in ["Poor", "Intermediate", "Favorable", "TBD", "Not tested"]:
            return row["cyto_score"]
        else:
            return None  # ê²°ì¸¡ì¹˜ ì²˜ë¦¬

    df["cyto_score_sum"] = df.apply(compute_cyto_score_sum, axis=1)
    return df


def mrd_cyto_score(df):
    def compute_mrd_cyto(row):
        if row["mrd_hct"] == "Positive":
            return {
                "Favorable": 10, "Intermediate": 8, "TBD": 7,
                "Poor": 4, "Not tested": 3
            }.get(row["cyto_score_sum"], 11)  # ê¸°ë³¸ê°’ 11

        elif row["mrd_hct"] == "Negative":
            return {
                "Favorable": 9, "Intermediate": 6, "TBD": 5,
                "Poor": 4, "Not tested": 3
            }.get(row["cyto_score_sum"], 11)  # ê¸°ë³¸ê°’ 11

        else:
            return 12  # `mrd_hct`ê°€ Positive/Negativeê°€ ì•„ë‹Œ ê²½ìš°

    df["mrd_cyto"] = df.apply(compute_mrd_cyto, axis=1)
    df = df.drop(columns = ["cyto_score_sum"])
    return df

def add_features_cnn_exp_01(df):
    # sex_match = df.sex_match.astype(str)
    # sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    # df['sex_match_bool'] = sex_match.astype("object")
    df = cyto_score_sum(df)
    df = mrd_cyto_score(df)
    df["hct_ci_score"] = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    return df

def get_cnn_exp_01_features(test):
    test = add_features_cnn_exp_01(test)
    test['hla_combined_low'] = test['hla_low_res_10']
    test['hla_combined_low'] = test.apply(fill_hla_combined_low, axis=1)
    test_dataset = ChannelDataset(test, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_loader



def infer(model: torch.nn.Module, dataloader_test: torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        x, (event, time) = next(iter(dataloader_test))
        log_hz = model(x)
    return log_hz, event, time


class Custom_dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        event = torch.tensor(sample["efs"]).bool()
        time = torch.tensor(sample["efs_time"]).float()
        x = torch.tensor(sample.drop(["efs", "efs_time"]).values).float()
        return x, (event, time)


def load_cox_model(num_features: int):
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features),
        torch.nn.Linear(num_features, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(64, 1),
    )
    return cox_model


def add_features_ts_exp_01(df):
    """
    Create some new features to help the model focus on specific patterns.
    """
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['year_hct'] -= 2000
    
    return df
    
target_cols = ["efs", "efs_time"]
drop_cols = ["ID"]

def get_ts_exp_01_features(train_df, test_df):
    train_df = add_features_ts_exp_01(train_df)
    test_df = add_features_ts_exp_01(test_df)
    
    cat_cols = [col for col in train_df.select_dtypes(include=["object"]).columns if col not in target_cols + drop_cols]
    num_cols = [col for col in train_df.columns if col not in cat_cols + target_cols + drop_cols]

    for col in cat_cols:
        train_df[col].fillna("Unknown", inplace=True)
        test_df[col].fillna("Unknown", inplace=True)
    
        labels = train_df[col].unique()
        for i in labels:
            train_df[f"{col}_{i}"] = train_df[col].apply(lambda x: 1 if x == i else 0)
            test_df[f"{col}_{i}"] = test_df[col].apply(lambda x: 1 if x == i else 0)
    
        if col != "race_group":
            train_df.drop(columns=[col], axis=1, inplace=True)
        test_df.drop(columns=[col], axis=1, inplace=True)

    # Numerical Features
    for col in num_cols:
        if col != "fold":
            imputer = SimpleImputer(strategy='mean')
            train_df[col] = imputer.fit_transform(train_df[col].values.reshape(-1, 1))
            test_df[col] = imputer.transform(test_df[col].values.reshape(-1, 1))

    train_df = train_df.drop(columns=drop_cols, axis=1)
    test_df = test_df.drop(columns=drop_cols, axis=1)

    train_df = train_df.drop(columns=["race_group"], axis=1)
    test_df["efs"] = np.nan
    test_df["efs_time"] = np.nan
    
    scaler = StandardScaler()
    inputs = [col for col in train_df.columns if col not in target_cols and col !="fold"]
    train_df[inputs] = scaler.fit_transform(train_df[inputs])
    test_df[inputs] = scaler.transform(test_df[inputs])
    
    dataloader_test = DataLoader(Custom_dataset(test_df), batch_size=len(test_df), shuffle=False)
    return train_df, dataloader_test


import pickle
import io
from tqdm import tqdm
from joblib import load
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


FOLDS   = 10
scaler  = StandardScaler()
imputer = KNNImputer(n_neighbors=5, weights='uniform')


def inference(model_path, train, test_df):

    test_predictions = np.zeros(len(test_df))
    bin_predictions = np.zeros(len(test_df))


    if "nn-exp-04" in model_path or "nn-exp-05" in model_path or "nn-exp-06" in model_path or "prlnn-exp-01" in model_path or "ts-exp-01" in model_path:
        pass
    else:
        train = train.drop("fold", axis=1)
    
    path = model_path.split('/')[-1]
    file = path.replace('-','_')


    if "/kaggle/input/lgbm-exp-03" == model_path:
        with open(f"/kaggle/input/k/jaytonde/{path}/{file}.pkl", 'rb') as f:
                models = pickle.load(f)
    elif "ds" in model_path or "nn-exp-02" in model_path or "nn-exp-04" in model_path or "nn-exp-05" in model_path:
        pass
    elif "xgboost-exp-05" in model_path or "catboost-exp-05" in model_path or "lgbm-exp-05" in model_path or "en-exp-02" in model_path or "rf-exp-05" in model_path or "xgboost-exp-06" in model_path or "catboost-exp-06" in model_path or "lgbm-exp-06" in model_path or "nn-exp-06" in model_path or "prlnn-exp-01" in model_path or "ri-exp-06" in model_path or "svr-exp-06" in model_path:
        
        if "nn-exp-06" in model_path or "prlnn-exp-01" in model_path:
            print(f"Loading classification model for : nn-exp-06 or prlnn-exp-01")
            cls_file_path = f"/kaggle/input/{path}/{file}_cls.pkl"
            print(f"Classification file path : {cls_file_path}")
            with open(cls_file_path, 'rb') as f:
                models_cls = pickle.load(f)
        else:
            print(f"Loading classification model for other than nn-exp-06")
            with open(f"/kaggle/input/{path}/{file}_cls.pkl", 'rb') as f:
                models_cls = pickle.load(f)
    
            with open(f"/kaggle/input/{path}/{file}.pkl", 'rb') as f:
                models = pickle.load(f)
    elif "tt-exp-01" in model_path or "mcts-exp-02" in model_path or "cnn-exp-01" in model_path or "ts-exp-01" in model_path:
        pass
        
    else:
        if "dt" not in model_path:
            with open(f"/kaggle/input/{path}/{file}.pkl", 'rb') as f:
                models = pickle.load(f)
            
    print("All models are loaded successfully....!")
    
    

    for fold in tqdm(range(FOLDS)):
        print(f"Fold : {fold}")

        if "ds" in model_path:
            model = create_model()
            model.load_net(f'/kaggle/input/ds-exp-01/ds_fold_{fold}.pt')
        elif "nn-exp-02" in model_path or "nn-exp-04" in model_path or "nn-exp-05" in model_path or "tt-exp-01" in model_path or "mcts-exp-02" in model_path or "nn-exp-06" in model_path or "prlnn-exp-01" in model_path or "cnn-exp-01" in model_path or "ts-exp-01" in model_path:
            pass
        else:
            model  = models[fold]
            
            
        if "svr" in model_path:
            if "svr-exp-06" in model_path:
                if fold == 0:
                    print(f"Inferencing : {model_path}")
                    train, test, FEATURES, CATS = get_exp_05_features(train.copy(), test_df.copy(), model_path)
                    imputer = KNNImputer(n_neighbors=5, weights='uniform')
                    scaler  = StandardScaler()
    
                bin_preds   = models_cls[fold].predict_proba(test[FEATURES])[:, 1]
            else:
                if fold==0:
                    print(f"Inferencing : {model_path}")
                    train, test, FEATURES = prepare_features(model_path, train.copy(), test_df.copy())
                    imputer = KNNImputer(n_neighbors=5, weights='uniform')
                    scaler  = StandardScaler()
        
            # Handle missing values
            train_imputed         = imputer.fit_transform(train[FEATURES].copy())
            test_imputed          = imputer.transform(test[FEATURES])

            # Convert back to DataFrame to maintain feature names
            train_imputed         = pd.DataFrame(train_imputed, columns=FEATURES, index=train.index)
            test_imputed          = pd.DataFrame(test_imputed, columns=FEATURES, index=test.index)
            
            # Scale features
            scaler.fit(train_imputed)
            
            test_scaled           = scaler.transform(test_imputed)
            fold_preds            = model.predict(test_scaled)                
            
        elif "tn-exp-01" in model_path:
            if fold == 0:
                print(f"Inferencing tn: {model_path}")
                train, test, FEATURES   = prepare_features(model_path, train, test_df)
            fold_preds              = model.predict(test[FEATURES].values).flatten()
            
        elif "nn-exp-01" in model_path and "prl" not in model_path and "cnn" not in model_path:
            if fold == 0:
                print(f"Inferencing nn-exp-01: {model_path}")
                X_cat, X_num  = get_nn_features(train, test_df.copy())
            fold_preds        = model.predict([X_cat.values, X_num.values])
            fold_preds        = fold_preds.flatten()

        elif "nn-exp-02" in model_path:
            if fold == 0:
                print(f"Inferencing nn-exp-02: {model_path}")
                test = get_nn_exp_02_features(train.copy(), test_df.copy())
            model          = get_nn_exp_02_model()
            model.load_net(f'/kaggle/input/nn-exp-02/nn_fold_{fold}.pt')
            fold_preds     = model.predict(test.copy().to_numpy()).flatten()

        elif "nn-exp-04" in model_path:
            if fold == 0:
                print(f"Inferencing nn-exp-04: {model_path}")
                
            tmp_train = train[train["fold"]!=fold].copy()
            tmp_train = tmp_train.drop("fold", axis=1)
            
            X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols = get_nn_exp_04_features(tmp_train.copy(), test_df.copy())

            model      = get_nn_exp_04_model(6, transformers, categorical_cols)

            state_dict = torch.load(f'/kaggle/input/nn-exp-04/fold_{fold}/model.pt')
            model.load_state_dict(state_dict)

            
            pred, _    = model.cuda().eval()(
                            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
                            torch.tensor(X_num_val, dtype=torch.float32).cuda()
                        )

            fold_preds =  pred.detach().cpu().numpy()

        elif "nn-exp-05" in model_path:
            if fold == 0:
                print(f"Inferencing nn-exp-05: {model_path}")
                
            tmp_train = train[train["fold"]!=fold].copy()
            tmp_train = tmp_train.drop("fold", axis=1)
            
            X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols = get_nn_exp_05_features(tmp_train.copy(), test_df.copy())

            model      = get_nn_exp_05_model(6, transformers, categorical_cols)
            state_dict = torch.load(f'/kaggle/input/nn-exp-05/fold_{fold}/model.pt')
        
            model.load_state_dict(state_dict)

            pred, _    = model.cuda().eval()(
                            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
                            torch.tensor(X_num_val, dtype=torch.float32).cuda()
                        )
           

            fold_preds =  pred.detach().cpu().numpy()

        elif "nn-exp-06" in model_path:
            if fold == 0:
                print(f"Inferencing nn-exp-06: {model_path}")
                
            tmp_train = train[train["fold"]!=fold].copy()
            tmp_train = tmp_train.drop("fold", axis=1)

            test, FEATURES = processing_06(tmp_train.copy(), test_df.copy())
            X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols = get_nn_exp_05_features(tmp_train.copy(), test_df.copy())

            model      = get_nn_exp_05_model(6, transformers, categorical_cols)
            state_dict = torch.load(f'/kaggle/input/nn-exp-06/fold_{fold}/model.pt')
            
            model.load_state_dict(state_dict)

            pred, _    = model.cuda().eval()(
                            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
                            torch.tensor(X_num_val, dtype=torch.float32).cuda()
                        )
            
            bin_preds   = models_cls[fold].predict_proba(test[FEATURES])[:, 1]
            fold_preds  =  pred.detach().cpu().numpy()

        elif "prlnn-exp-01" in model_path:
            if fold == 0:
                print(f"Inferencing prlnn-exp-01: {model_path}")
                
            tmp_train = train[train["fold"]!=fold].copy()
            tmp_train = tmp_train.drop("fold", axis=1)

            test, FEATURES = processing_06(tmp_train.copy(), test_df.copy())
            X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, categorical_cols = get_nn_exp_05_features(tmp_train.copy(), test_df.copy())

            model      = get_nn_exp_05_model(6, transformers, categorical_cols)
            state_dict = torch.load(f'/kaggle/input/prlnn-exp-01/fold_{fold}/model.pt')

            model.load_state_dict(state_dict)

            
            pred, _    = model.cuda().eval()(
                            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
                            torch.tensor(X_num_val, dtype=torch.float32).cuda()
                        )
            bin_preds   = models_cls[fold].predict_proba(test[FEATURES])[:, 1]
            #print(f"Binary preds for fold : {fold} is : {bin_preds}")
            fold_preds  =  pred.detach().cpu().numpy()

        elif "tf" in model_path:
            if fold == 0:
                print(f"Inferencing tf: {model_path}")
                test, FEATURES = get_tf_features(train.copy(),test_df.copy())
                fold_preds     = model.predict(test[FEATURES].copy())

        elif "tabm" in model_path:
            if "02" in model_path:
                model.eval()
                if fold == 0:
                    print(f"Inferencing tabm 02: {model_path}")
                    test_dl = get_tabm_features_02(train.copy(), test_df.copy())
    
                test_pred_list = []
                with torch.no_grad():
                    for test_tensor in test_dl:
                        X_num_test, X_cat_test = [t.to(device) for t in test_tensor]
                        output                 = model(X_num_test, X_cat_test).squeeze(-1)
                        test_pred_list.append(output.mean(1).cpu().numpy())
                fold_preds = np.concatenate([p for p in test_pred_list])
            else:
                if fold == 0:
                    print(f"Inferencing tabm 01: {model_path}")
                    test, FEATURES = get_tabm_features(test_df.copy())
                fold_preds         = model.predict(test[FEATURES].copy())
                
        elif "abd" in model_path:
            if fold == 0:
                print(f"Inferencing abd: {model_path}")
                test = get_abd_features(train_df.copy(), test_df.copy())
            fold_preds = model.predict(test.copy())   

        elif "ds" in model_path:
            if fold == 0:
                print(f"Inferencing ds: {model_path}")
                train, test = load_and_preprocess_data(train_df.copy(), test_df.copy())
                test        = get_ds_features(train.copy(), test.copy())
            fold_preds = model.predict(test).flatten()

        elif "xgboost-exp-05" in model_path or "catboost-exp-05" in model_path or "lgbm-exp-05" in model_path or "en-exp-02" in model_path or "rf-exp-05" in model_path or "xgboost-exp-06" in model_path or "catboost-exp-06" in model_path or "lgbm-exp-06" in model_path or "ri-exp-06" in model_path:
            if fold == 0:
                train_tmp, test, FEATURES, CATS = get_exp_05_features(train.copy(), test_df.copy(), model_path)

            if "rf-exp-05" in model_path:
                test = get_rf_encoders(train_tmp, test, FEATURES, CATS)

            bin_preds   = models_cls[fold].predict_proba(test[FEATURES])[:, 1]

            if isinstance(model, dict):
                model_en = make_pipeline(
                                model['imputer'],
                                model['model']
                            )
                fold_preds  = model_en.predict(test[FEATURES].copy())
            else:
                fold_preds  = model.predict(test[FEATURES].copy())

            print(fold_preds)

        elif "tn-exp-02" in model_path:
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                test, FEATURES = get_tn_exp_02_features(train.copy(), test_df.copy())
            fold_preds              = model.predict(test[FEATURES].values).squeeze()

        elif "vr-exp-01" in model_path:
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                test = get_vr_exp_01_features(train.copy(), test_df.copy())
                
            test_naf          = model['naf_model'].predict(test)
            test_km           = model['km_model'].predict(test)
            fold_preds        = ((test_naf + test_km) / 2)

        elif "tt-exp-01" in model_path:
            DEVICE = "cuda"
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                test_loader, num_categories,num_features = get_tt_exp_01_features(train.copy(), test_df.copy())

            model = get_tt_exp_01_model(num_categories,num_features)
            model.load_state_dict(torch.load(f'{model_path}/tabtransformer_fold{fold}.pt'))
            
            test_preds_fold = []
            with torch.no_grad():
                DEVICE = "cuda"
                for cat, num, _ in test_loader:
                    cat = cat.to(DEVICE)
                    num = num.to(DEVICE)
                    pred = model(cat, num).squeeze()
                    test_preds_fold.extend(pred.cpu().numpy())
            fold_preds = test_preds_fold

        elif "mcts-exp-02" in model_path:
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                test, FEATURES = get_mcts_exp_02_features(train.copy(), test_df.copy())

            checkpoint = torch.load(f'/kaggle/input/mcts-exp-02/fold_{fold}_model.pth')
            imputer = checkpoint['imputer']
            scaler = checkpoint['scaler']
            X_test_imp    = imputer.transform(test[FEATURES])
            X_test_scaled = scaler.transform(X_test_imp)
            
            # Predict with MC Dropout
            test_dataset = SurvivalDataset(X_test_scaled)
            test_loader  = DataLoader(test_dataset, batch_size=256)
            
            model = MCDropoutNet(input_size=len(FEATURES)).to(device)
            model.load_state_dict(checkpoint['model_state'])
            
            model.train()  # Keep dropout active
            with torch.no_grad():
                mc_preds = []
                for X_batch in test_loader:
                    batch_preds = torch.stack([model(X_batch).squeeze() for _ in range(30)])
                    mc_preds.append(batch_preds.mean(0).cpu().numpy())
            fold_preds=np.concatenate(mc_preds)

        elif "xgboost-exp-09" in model_path:
            if fold==0:
                print(f"Inferencing : xgboost-exp-09")
                test, FEATURES = get_xgboost_exp_09_features(train.copy(), test_df.copy())
            fold_preds = model.predict(test[FEATURES]) 

        elif "lgbm-exp-08" in model_path:
            if fold == 0:
                print(f"Inferencing : xgboost-exp-08->")
                test, FEATURES = get_lgbm_exp_08(train.copy(), test_df.copy())
            fold_preds = model.predict(test[FEATURES])
        elif "xgboost-exp-10" in model_path:
            if fold == 0:
                print(f"Inferencing : xgboost-exp-10")
                FEATURES, test = get_xgboost_exp_10_features(train.copy(), test_df.copy())
            fold_preds = model.predict(test[FEATURES])   

        elif "lir-exp-01" in model_path:
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                train, test, FEATURES   = prepare_features(model_path, train.copy(), test_df.copy())

            model_lir = make_pipeline(
                            model['imputer'],
                            model['scaler'],
                            model['model']
                        )
            fold_preds  = model_lir.predict(test[FEATURES].copy())

        elif "cnn-exp-01" in model_path:
            if fold == 0:
                test_loader = get_cnn_exp_01_features(test_df.copy())

            fold_predictions = []
            num_folds        = 10
            
            
            model = SurvivalRegressionCNN()
            model.load_state_dict(torch.load(f'/kaggle/input/cnn-exp-01/saved_models/fold_{fold}_final_model.pt'))
            model.eval()
            
            fold_preds = []
            with torch.no_grad():
                for x in test_loader:
                    outputs = model(x).squeeze()
                    fold_preds.extend(outputs.cpu().numpy())

        elif "ts-exp-01" in model_path:
            if fold == 0:
                trn_df, dataloader_test = get_ts_exp_01_features(train_df.copy(), test_df.copy())

            trn_df     = trn_df[trn_df["fold"]!=fold]
            trn_df_fold = trn_df.copy()
            trn_df_fold.drop(["fold"], axis=1, inplace=True)
            
            if fold == 0:
                dataloader_train = DataLoader(Custom_dataset(trn_df_fold.copy()), batch_size=2048)
                num_features     = next(iter(dataloader_train))[0].size(1)
            
            model            = load_cox_model(num_features)
            model.load_state_dict(torch.load(f'/kaggle/input/ts-exp-01/cox_model_fold_{fold}.pt'))
            
            fold_preds = infer(model, dataloader_test)[0].view(-1).numpy()
        else: 
            if fold == 0:
                print(f"Inferencing others: {model_path}")
                train, test, FEATURES   = prepare_features(model_path, train.copy(), test_df.copy())

            if isinstance(model, dict):
                model_en = make_pipeline(
                                model['imputer'],
                                model['model']
                            )
                fold_preds  = model_en.predict(test[FEATURES].copy())
            else:
                fold_preds              = model.predict(test[FEATURES].copy())
            
        test_predictions += fold_preds

        if "xgboost-exp-05" in model_path or "catboost-exp-05" in model_path or "lgbm-exp-05" in model_path or "en-exp-02" in model_path or "rf-exp-05" in model_path or "xgboost-exp-06" in model_path or "catboost-exp-06" in model_path or "lgbm-exp-06" in model_path or "nn-exp-06" in model_path or "prlnn-exp-01" in model_path or "ri-exp-06" in model_path or "svr-exp-06" in model_path:
            bin_predictions += bin_preds
    
    # Get the average predictionr
    if "nn-exp-04" in model_path:
        return -test_predictions
    elif "nn-exp-06" in model_path or 'xgboost-exp-07' in model_path or "prlnn-exp-01" in model_path:
        pass
    else:
        test_predictions /= FOLDS

    if "xgboost-exp-05" in model_path or "catboost-exp-05" in model_path or "lgbm-exp-05" in model_path or "en-exp-02" in model_path or "rf-exp-05" in model_path or "xgboost-exp-06" in model_path or "catboost-exp-06" in model_path or "lgbm-exp-06" in model_path or "ri-exp-06" in model_path or "svr-exp-06" in model_path:
        bin_predictions                             = (bin_predictions / FOLDS > 0.5).astype(int)

        pred_classifier_np                          = np.array(bin_predictions)
        prediction_np                               = np.array(test_predictions)
        
        combined_pred                               = np.column_stack((pred_classifier_np, prediction_np))
        combined_pred[combined_pred[:, 0] == 1, 1] += 0.1
        
        return combined_pred[:, 1]
    elif "nn-exp-06" in model_path or "prlnn-exp-01" in model_path:
        print(f"Returning : nn-exp-06")
        #return -test_predictions
        bin_predictions                             = (bin_predictions / FOLDS > 0.5).astype(int)
        pred_classifier_np                          = np.array(bin_predictions)

        test_predictions                            = -test_predictions
        prediction_np                               = np.array(test_predictions)
        #print(test_predictions)
        
        combined_pred                               = np.column_stack((pred_classifier_np, prediction_np))
        combined_pred[combined_pred[:, 0] == 1, 1] += 0.2
        
        return combined_pred[:, 1]
    
    return test_predictions


train_df = pd.read_excel("/kaggle/input/cibmtr-2024-dataset/random_folding.xlsx")
test_df  = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


experiments


first_best_index = '/kaggle/input/prlnn-exp-01'
experiments.remove(first_best_index)
experiments


initial_test_preds = inference(first_best_index, train_df, test_df)
initial_test_preds


best_score = 0.6887243282521093


model_weights = {
'/kaggle/input/catboost-exp-05': 0.48000000000000087,
'/kaggle/input/xgboost-exp-09': 0.2700000000000007,
'/kaggle/input/lgbm-exp-08': -0.20999999999999974,
'/kaggle/input/catboost-exp-01': 0.10000000000000053,
'/kaggle/input/lasso-exp-01': -0.03999999999999959,
'/kaggle/input/nn-exp-04': 0.12000000000000055,
'/kaggle/input/svr-exp-06': -0.0499999999999996,
'/kaggle/input/ds-exp-01': 0.04000000000000048,
'/kaggle/input/rf-exp-05': -0.05999999999999961,
'/kaggle/input/svr-exp-01': -0.03999999999999959,
'/kaggle/input/tf-exp-01': 0.04000000000000048,
'/kaggle/input/catboost-exp-03': -0.03999999999999959,
'/kaggle/input/catboost-exp-06': 0.0600000000000005,
'/kaggle/input/tn-exp-02': 0.03000000000000047,
'/kaggle/input/xgboost-exp-02': -0.0499999999999996,
'/kaggle/input/catboost-exp-04': 0.020000000000000462,
'/kaggle/input/lgbm-exp-03': -0.029999999999999583,
'/kaggle/input/ts-exp-01': -0.019999999999999574,
'/kaggle/input/xgboost-exp-10': 0.04000000000000048,
'/kaggle/input/xgboost-exp-06': -0.029999999999999583,
'/kaggle/input/en-exp-02': 0.020000000000000462,
'/kaggle/input/ri-exp-06': -0.029999999999999583,
'/kaggle/input/tabm-exp-02': 0.0600000000000005,
'/kaggle/input/nn-exp-01': -0.05999999999999961,
'/kaggle/input/lgbm-exp-01': -0.0499999999999996,
'/kaggle/input/nn-exp-06': -0.0499999999999996,
'/kaggle/input/xgboost-exp-05': 0.04000000000000048,
'/kaggle/input/lir-exp-01': -0.019999999999999574,
'/kaggle/input/lgbm-exp-04': 0.020000000000000462,
'/kaggle/input/nn-exp-05': 0.010000000000000453,
'/kaggle/input/lgbm-exp-06': -0.009999999999999565,
'/kaggle/input/nn-exp-02': 0.010000000000000453,
'/kaggle/input/xgboost-exp-01': 4.440892098500626e-16,
'/kaggle/input/tt-exp-01': 4.440892098500626e-16,
#'/kaggle/input/et-exp-01': 4.440892098500626e-16
}


initial_test_preds 


train_df               = pd.read_excel("/kaggle/input/cibmtr-2024-dataset/random_folding.xlsx")
test_df                = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
test_preds             = np.zeros(len(test_df))

preds_dict = {}
for model, weight in model_weights.items():
   print(f"exp name       : {model}\n")
   test_df                = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
   test_preds             = inference(model, train_df.copy(), test_df.copy())
   print(f"{model} have preds {test_preds}")
   test_preds             = (1-weight) * rankdata(initial_test_preds) + weight * rankdata(test_preds)
   initial_test_preds     = test_preds


sub            = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = test_preds
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()







