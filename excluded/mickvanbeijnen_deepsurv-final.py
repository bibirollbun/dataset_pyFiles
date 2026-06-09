!pip install /kaggle/input/cibmtr-pip-install/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-pip-install/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/cibmtr-pip-install/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-pip-install/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/cibmtr-pip-install/lifelines-0.30.0-py3-none-any.whl


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


from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")

pd.set_option('display.max_columns', 100)

train = pd.read_csv(ROOT_DATA_PATH.joinpath("train.csv"))
test = pd.read_csv(ROOT_DATA_PATH.joinpath("test.csv"))


# Variables for tuning the Feature Engineering Process
age_bin_size = 10
treatment_categorical = False

add_new_features = True
remove_original_features = False


# Feature Engineering

# Missing values per patient
if (add_new_features == True):
    train['missing_count'] = train.isnull().sum(axis=1)
    test['missing_count'] = test.isnull().sum(axis=1)

# Binning of the patient and donor age
if(max(train['age_at_hct']) > 73): # Check whether binning has already been done
    train['age_at_hct'] = train['age_at_hct'] // age_bin_size
    test['age_at_hct'] = test['age_at_hct'] // age_bin_size
    
    train['donor_age'] = train['donor_age'] // age_bin_size
    test['donor_age'] = test['donor_age'] // age_bin_size

# Normalization of operation year
if(min(train['year_hct']) > 2000): # Check whether normalization has already been done
    train['year_hct'] = train['year_hct'] - 2000
    test['year_hct'] = test['year_hct'] - 2000

# Cohort (based off of McDonald et al. https://pubmed.ncbi.nlm.nih.gov/31958813/)
if (add_new_features == True):
    train['cohort'] = (train['year_hct'] >= 13)
    test['cohort'] = (test['year_hct'] >= 13)

if (remove_original_features == True):
    train.drop('year_hct', axis=1, inplace=True)
    test.drop('year_hct', axis=1, inplace=True)

# Same sex
if (add_new_features == True):
    train['same_sex'] = (train['sex_match'] == 'M-M') | (train['sex_match'] == 'F-F')
    test['same_sex'] = (test['sex_match'] == 'M-M') | (test['sex_match'] == 'F-F')

if (remove_original_features == True):
    train.drop('sex_match', axis=1, inplace=True)
    test.drop('sex_match', axis=1, inplace=True)

# Treatment 
if (add_new_features == True):
    if (treatment_categorical == True):
        train['treatment'] = (train['melphalan_dose'] == 'MEL') | (train['rituximab'] == 'Yes') \
        | (train['in_vivo_tcd'] == 'Yes')
        test['treatment'] = (test['melphalan_dose'] == 'MEL') | (test['rituximab'] == 'Yes') \
        | (test['in_vivo_tcd'] == 'Yes')          
    else:
        train['treatment'] = (train['melphalan_dose'] == 'MEL').astype(int) \
        + (train['rituximab'] == 'Yes').astype(int) \
        + (train['in_vivo_tcd'] == 'Yes').astype(int) \
        + ((train['gvhd_proph'] != 'No GvHD Prophylaxis') & (train['gvhd_proph'] != 'Parent Q = yes, but no agent')).astype(int) \
        + (train['conditioning_intensity'] != 'No drugs reported').astype(int) \
        + (train['tbi_status'] != 'No TBI').astype(int)
        test['treatment'] = (test['melphalan_dose'] == 'MEL').astype(int) \
        + (test['rituximab'] == 'Yes').astype(int) \
        + (test['in_vivo_tcd'] == 'Yes').astype(int) \
        + ((test['gvhd_proph'] != 'No GvHD Prophylaxis') & (test['gvhd_proph'] != 'Parent Q = yes, but no agent')).astype(int) \
        + (test['conditioning_intensity'] != 'No drugs reported').astype(int) \
        + (test['tbi_status'] != 'No TBI').astype(int)            

if (remove_original_features == True):
    if (treatment_categorical == True):
        train.drop(['melphalan_dose', 'rituximab', 'in_vivo_tcd'], axis=1, inplace=True)
        test.drop(['melphalan_dose', 'rituximab', 'in_vivo_tcd'], axis=1, inplace=True)
    else:
        train.drop(['melphalan_dose', 'rituximab', 'in_vivo_tcd', 'gvhd_proph',
                        'conditioning_intensity', 'tbi_status'], axis=1, inplace=True)
        test.drop(['melphalan_dose', 'rituximab', 'in_vivo_tcd', 'gvhd_proph',
                        'conditioning_intensity', 'tbi_status'], axis=1, inplace=True)

# Risk Factors
risk_factors = ['psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor']

if (add_new_features == True):
    risk_factor_counts_train = train[risk_factors].eq('Yes').sum(axis=1)
    mrd_hct_risk_train = (train['mrd_hct'] == 'Positive').astype(int)
    train['risk_factor_count'] = risk_factor_counts_train + mrd_hct_risk_train
    
    risk_factor_counts_test = test[risk_factors].eq('Yes').sum(axis=1)
    mrd_hct_risk_test = (test['mrd_hct'] == 'Positive').astype(int)
    test['risk_factor_count'] = risk_factor_counts_test + mrd_hct_risk_test

if (remove_original_features == True):
    train.drop(risk_factors, axis=1, inplace=True)
    test.drop(risk_factors, axis=1, inplace=True)
    train.drop('mrd_hct', axis=1, inplace=True)
    test.drop('mrd_hct', axis=1, inplace=True)

# Combine HLA features, since all of the hla_match* features have identical values for each patient
# The same goes for the hla* features
if (add_new_features == True):
    train['hla_match'] = train['hla_match_a_low'] 
    test['hla_match'] = test['hla_match_a_low'] 
    
    train['hla'] = train['hla_nmdp_6']
    test['hla'] = test['hla_nmdp_6']

if (remove_original_features == True):
    hla_features = ['hla_match_a_low', 'hla_match_a_high', 'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high', 'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high', 'hla_nmdp_6', 'hla_low_res_6', 'hla_high_res_6',
    'hla_low_res_8', 'hla_high_res_8', 'hla_low_res_10', 'hla_high_res_10']
    train.drop(hla_features, axis=1, inplace=True)
    test.drop(hla_features, axis=1, inplace=True)


ORIGINAL_CATEGORICAL_VARIABLES = [
    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 

    # Biological matching with donor
    'sex_match', 

    # HLA Features
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

ORIGINAL_NUMERICAL_VARIABLES = []

ENGINEERED_CATEGORICAL_VARIABLES = ['cohort', 'same_sex', 'hla_match', 'hla']

ENGINEERED_NUMERICAL_VARIABLES = ['missing_count', 'risk_factor_count']

CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'cyto_score', 'cyto_score_detail',

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match'
]

NUMERICAL_VARIABLES = ['donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']

if (age_bin_size > 1): 
    ORIGINAL_CATEGORICAL_VARIABLES.extend(['year_hct'])
elif (age_bin_size == 1):
    ORIGINAL_NUMERICAL_VARIABLES.extend(['year_hct'])
else:
    raise Exception("Age bin is too small")

if (treatment_categorical == True):
    ENGINEERED_CATEGORICAL_VARIABLES.extend(['treatment'])
    ORIGINAL_CATEGORICAL_VARIABLES.extend(['melphalan_dose', 'rituximab', 'in_vivo_tcd'])
    CATEGORICAL_VARIABLES.extend(['gvhd_proph', 'conditioning_intensity', 'tbi_status'])
else:
    ENGINEERED_NUMERICAL_VARIABLES.extend(['treatment'])
    ORIGINAL_CATEGORICAL_VARIABLES.extend(['melphalan_dose', 'rituximab', 'in_vivo_tcd', 'gvhd_proph',
                        'conditioning_intensity', 'tbi_status'])
    
if (add_new_features == True):
    CATEGORICAL_VARIABLES.extend(ENGINEERED_CATEGORICAL_VARIABLES)
    NUMERICAL_VARIABLES.extend(ENGINEERED_NUMERICAL_VARIABLES)

if (remove_original_features == False):
    CATEGORICAL_VARIABLES.extend(ORIGINAL_CATEGORICAL_VARIABLES)
    NUMERICAL_VARIABLES.extend(ORIGINAL_NUMERICAL_VARIABLES)

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


complete_columns = train.columns[~train.isna().any()].tolist()


# Encode categorical variables w/o keeping nAn
train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype(str).astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype(str).astype('category')
train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].apply(lambda x: x.cat.codes)
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].apply(lambda x: x.cat.codes)


from sklearn.preprocessing import LabelEncoder

def apply_label_encoding(df_train, df_test):
    label_encoder = LabelEncoder()
    # df_train[CATEGORICAL_VARIABLES] = df_train[CATEGORICAL_VARIABLES].fillna("Unknown")
    # df_test[CATEGORICAL_VARIABLES] = df_test[CATEGORICAL_VARIABLES].fillna("Unknown")
    
    for cat_var in CATEGORICAL_VARIABLES:
        df_train[cat_var] = label_encoder.fit_transform(df_train[cat_var])
        df_test[cat_var] = label_encoder.transform(df_test[cat_var])

    return df_train, df_test

train, test = apply_label_encoding(train, test)


# def fill_nan_with_median_for_numeric(df):
#     df[NUMERICAL_VARIABLES] = df[NUMERICAL_VARIABLES].fillna(df[NUMERICAL_VARIABLES].median())
#     return df

# train = fill_nan_with_median_for_numeric(train)
# test = fill_nan_with_median_for_numeric(test)


import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import numpy as np

def impute_numerical_features(df, method):
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'mice':
        imputer = IterativeImputer(max_iter=10, random_state=1234)
    else:
        raise ValueError("Unsupported imputation method")
    
    df[NUMERICAL_VARIABLES] = imputer.fit_transform(df[NUMERICAL_VARIABLES])
    return df

def impute_categorical_features(df, method):
    if method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'missing_category':  # Treat missing as a separate category
        df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna('Missing')
        return df
    elif method == 'mice':
        imputer = IterativeImputer(max_iter=10, random_state=1234)
    else:
        raise ValueError("Unsupported imputation method")
    
    df[CATEGORICAL_VARIABLES] = imputer.fit_transform(df[CATEGORICAL_VARIABLES])
    return df

train = impute_numerical_features(train, method='median')
test = impute_numerical_features(test, method='median')

train = impute_categorical_features(train, method='mode')
test = impute_categorical_features(test, method='mode')


from sklearn.feature_selection import mutual_info_classif

y_mi = train[["efs_time", "efs"]]
x_mi = train.drop(["efs_time", "efs"], axis=1)

# Mutual information for a binary event outcome
mi_scores = mutual_info_classif(x_mi, y_mi["efs"], random_state=0)
for feature, score in sorted(zip(x_mi.columns, mi_scores), key=lambda x: x[1], reverse=True)[:10]:
    print(feature, "MI score:", score)

# Correlation filtering among features (Pearson for numeric features)
corr_matrix = x_mi.corr().abs()  # absolute correlation
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index 
                   if upper_tri.loc[row, col] > 0.90]
# Drop one feature from each highly correlated pair
to_drop = set()
for col, row in high_corr_pairs:
    if col not in to_drop and row not in to_drop:
        to_drop.add(row)  # drop the 'row' feature arbitrarily
        try:
            CATEGORICAL_VARIABLES.remove(row)
        except:
            pass
        try:
            NUMERICAL_VARIABLES.remove(row)
        except:
            pass
train = train.drop(columns=to_drop)
test = test.drop(columns=to_drop)
print("Dropped due to high correlation:", to_drop)

train = train.drop(columns=ID_COLUMN)
test = test.drop(columns=ID_COLUMN)


def randomly_mask(df, cols, mask_prob=0.1, seed=None):
    df = df.copy()
    if seed is not None:
        np.random.seed(seed)
    for col in cols:
        mask = np.random.rand(len(df)) < mask_prob
        df.loc[mask, col] = np.nan  # or a special value if needed
    return df

df_augmented = randomly_mask(train, complete_columns, mask_prob=0.1, seed=42)


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)


df_train = train.copy()
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)

cols_standardize = NUMERICAL_VARIABLES
cols_leave = CATEGORICAL_VARIABLES

standardize = [([col], MinMaxScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
x_preds = x_mapper.transform(test).astype('float32')

get_target = lambda df: (df['efs_time'].values, df['efs'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val


in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_size = 256
batch_norm = True
dropout = 0.1
output_bias = False
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

model = CoxPH(net, tt.optim.Adam)
model.optimizer.set_lr(0.01)



%%time
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
log.plot()


_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print("Validation C-index score : ", ev.concordance_td())


risk_scores = model.predict(x_preds)
risk_scores


sub = pd.read_csv(ROOT_DATA_PATH.joinpath("sample_submission.csv"))
sub.prediction = risk_scores
sub.to_csv("submission.csv",index=False)

