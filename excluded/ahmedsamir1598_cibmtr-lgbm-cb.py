!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy import stats
from warnings import filterwarnings
from xgboost import XGBRegressor, XGBClassifier
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping, callback
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from lifelines import KaplanMeierFitter
import joblib
from colorama import Fore, Back, Style
import time
import optuna
from optuna.samplers import TPESampler
import imblearn
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor

filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')

# Basic exploration
print("Dataset shape:", train.shape)
print("\nFirst 5 rows:")
display(train.head())

# Target variable analysis
print("\nTarget variable distribution:")
print(train['efs'].value_counts(normalize=True))



train.info()


train.nunique()


# Missing values analysis
plt.figure(figsize=(20, 8))
missing = train.isna().mean().sort_values(ascending=False)
missing[missing > 0].plot(kind='bar', title='Missing Values Distribution')
plt.ylabel('Proportion Missing')
plt.show()


# Survival analysis visualization
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for name, grouped_df in train.groupby('efs'):
    kmf.fit(grouped_df["efs_time"], 
            grouped_df["efs"], 
            label=f'Event = {name}')
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.show()


# Categorical variables analysis
cat_cols = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
cat_cols = [c for c in cat_cols if c in train.columns and c not in ['efs', 'efs_time']]

plt.figure(figsize=(20, 30))
for i, col in enumerate(cat_cols[:15]):  # First 15 categorical features
    plt.subplot(5, 3, i+1)
    train[col].value_counts().plot(kind='bar')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


# Numerical variables analysis
num_cols = data_dict[data_dict['type'] == 'Numerical']['variable'].tolist()

# Calculate grid dimensions dynamically
n_cols = 4
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Round up division

plt.figure(figsize=(20, 5*n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(train[col].dropna(), kde=True)  # Add dropna to handle missing values
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


# Correlation analysis
plt.figure(figsize=(20, 10))
corr_matrix = train[num_cols + ['efs_time', 'efs']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(np.log(train['age_at_hct']), bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Year')
plt.show()


# Time-to-event distribution
plt.figure(figsize=(12, 6))
sns.histplot(train['efs_time'], bins=30, kde=True)
plt.title('Distribution of Time to Event')
plt.xlabel('Months')
plt.show()


# Race group analysis
if 'race_group' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train, x='race_group', hue='efs')
    plt.title('Event Distribution by Race Group')
    plt.xticks(rotation=45)
    plt.show()

# Age analysis
if 'age_at_hct' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='efs', y='age_at_hct', data=train)
    plt.title('Age Distribution by Event Status')
    plt.show()

# Comorbidity analysis
if 'comorbidity_score' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='efs', y='comorbidity_score', data=train)
    plt.title('Comorbidity Score Distribution by Event Status')
    plt.show()

# Treatment modality analysis
treatment_cols = ['graft_type', 'conditioning_intensity', 'in_vivo_tcd']
plt.figure(figsize=(18, 6))
for i, col in enumerate(treatment_cols):
    plt.subplot(1, 3, i+1)
    sns.countplot(x=col, hue='efs', data=train)
    plt.title(f'Event Distribution by {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sns.kdeplot(data=train, x="age_at_hct", hue="efs", common_norm=False, fill=True)
plt.xlabel("Age at HCT")
plt.ylabel("Density")
plt.title("KDE Plot of Age at HCT by EFS")
plt.show()


def dropcols(df):
    df_dropped = df.drop(columns = ['mrd_hct', 'tce_match', 'cyto_score_detail'])
    return df_dropped
    
train1 = dropcols(train)
test1 = dropcols(test)


train1['dri_score'].value_counts()


#### fill dri_score with mod
train1['dri_score'] = train1['dri_score'].fillna('Intermediate')
test1['dri_score'] = test1['dri_score'].fillna('Intermediate')


#train1['donor_related'].value_counts()


#### fill donor_related with mod
train1['donor_related'].fillna(train1['donor_related'].mode()[0], inplace=True)
test1['donor_related'].fillna(test1['donor_related'].mode()[0], inplace=True)


#### fill gvhd_proph with mod
train1['gvhd_proph'].fillna(train1['gvhd_proph'].mode()[0], inplace=True)
test1['gvhd_proph'].fillna(test1['gvhd_proph'].mode()[0], inplace=True)


#### fill sex_match with mod
train1['sex_match'].fillna(train1['sex_match'].mode()[0], inplace=True)
test1['sex_match'].fillna(test1['sex_match'].mode()[0], inplace=True)


def recalculate_hla_sums(df):
    # Calculate new columns by summing existing columns after filling NaNs with 0
    df["hla_nmdp_6"] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df["hla_low_res_6"] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_low"].fillna(0)
    df["hla_high_res_6"] = df["hla_match_a_high"].fillna(0) + df["hla_match_b_high"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df["hla_low_res_8"] = (df["hla_match_a_low"].fillna(0) + 
                           df["hla_match_b_low"].fillna(0) + 
                           df["hla_match_c_low"].fillna(0) + 
                           df["hla_match_drb1_low"].fillna(0))
    df["hla_high_res_8"] = (df["hla_match_a_high"].fillna(0) + 
                            df["hla_match_b_high"].fillna(0) + 
                            df["hla_match_c_high"].fillna(0) + 
                            df["hla_match_drb1_high"].fillna(0))
    df["hla_low_res_10"] = (df["hla_match_a_low"].fillna(0) + 
                            df["hla_match_b_low"].fillna(0) + 
                            df["hla_match_c_low"].fillna(0) + 
                            df["hla_match_drb1_low"].fillna(0) +
                            df["hla_match_dqb1_low"].fillna(0))
    df["hla_high_res_10"] = (df["hla_match_a_high"].fillna(0) + 
                             df["hla_match_b_high"].fillna(0) + 
                             df["hla_match_c_high"].fillna(0) + 
                             df["hla_match_drb1_high"].fillna(0) +
                             df["hla_match_dqb1_high"].fillna(0))
    return df


train1 = recalculate_hla_sums(train1)
test1 = recalculate_hla_sums(test1)


# fill categorical with nan
categorical_cols = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
for i in ['mrd_hct', 'tce_match', 'cyto_score_detail']:
    categorical_cols.remove(i)


train1[categorical_cols] = train1[categorical_cols].fillna('nan')
testcatcols = categorical_cols
testcatcols.remove('efs')
test1[categorical_cols] = test1[testcatcols].fillna('nan')


#### Drop low feature importance and low correlation
train1.drop(columns=['renal_issue', 'rituximab'], inplace = True)
test1.drop(columns=['renal_issue', 'rituximab'], inplace = True)


class ParticipantVisibleError(Exception):
    pass


def custom_score(solution, submission, row_id_column_name, prediction_label='prediction', print_info=True):
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_dict = {}
    for race in sorted(merged_df_race_dict.keys()):
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])

        metric_dict[race] = c_index_race

    race_c_index = list(metric_dict.values())
    c_score = float(np.mean(race_c_index) - np.std(race_c_index))
    if print_info:
        print(f"{Fore.GREEN}{Style.BRIGHT}# c-index={c_score:.4f}, mean={np.mean(race_c_index):.4f} std={np.std(race_c_index):.4f}{Style.RESET_ALL}")
    
    return c_score, metric_dict


def display_overall(df):
    
    race_groups = [
        'American Indian or Alaska Native', 'Asian',
       'Black or African-American', 'More than one race',
       'Native Hawaiian or other Pacific Islander', 'White'
    ]
    df['mean'] = df[race_groups].mean(axis=1)
    df['std'] = np.std(df[race_groups], axis=1)
    df['score'] = df['mean'] - df['std']
    df = df.T
    df['Overall'] = df.mean(axis=1)
    temp = df.drop(index=['std']).values
    display(df
            .iloc[:len(race_groups)]
            .style
            .format(precision=4)
            .background_gradient(axis=None, vmin=temp.min(), vmax=temp.max(), cmap="cool")
            .concat(df.iloc[len(race_groups):].style.format(precision=3))
           )

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    
    return y
def CIndexMetric_XGB(y_true, y_pred):
    ds_pred["prediction"] = y_pred
    cindex_score, _ = custom_score(ds_true.copy(), ds_pred.copy(), "ID", print_info=False)
    return -cindex_score

def CIndexMetric_LGB(y_true, y_pred):
    ds_pred["prediction"] = y_pred
    cindex_score, _ = custom_score(ds_true.copy(), ds_pred.copy(), "ID", print_info=False)
    return ('C-Index', cindex_score, True)


#no preprocessing to test metric
train1 = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test1 = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
train1["label"] = transform_survival_probability(train1, time_col='efs_time', event_col='efs')
train1.loc[train1['efs']==0, 'label'] -= 0.2
RMV = ["ID","efs","efs_time","label",'y','kfold']
FEATURES = [c for c in train1.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
CAT_FEATURES = []
for c in FEATURES:
    if train1[c].dtype=="object":
        CAT_FEATURES.append(c)
        train1[c] = train1[c].fillna("NAN")
        test1[c] = test1[c].fillna("NAN")
combined = pd.concat([train1, test1], axis=0, ignore_index=True)
print("The CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CAT_FEATURES:
        print(f"{c}, ", end="")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

train2 = combined.iloc[:len(train1)].copy()
test2 = combined.iloc[len(train1):].reset_index(drop=True).copy()
folds = 5
train2['kfold'] = -1  

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
groups = train2['efs'].astype(str)
for fold, (train_idx, val_idx) in enumerate(skf.split(X=train2, y=groups)):
    train2.loc[val_idx, 'kfold'] = fold


from sklearn.metrics import mean_squared_error, make_scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False) 


train2.head()


from sklearn.preprocessing import LabelEncoder

hparams = {
    'max_depth': 6,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'min_child_weight': 40,
    'gamma': 1,
    'eta': 0.0,
    'reg_lambda': 0.1,
    'reg_alpha': 0.1,
    'eps': 2e-2,
    'eps_mul': 1.01,
    'pos_shift': 0.2
}

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train2))
pred_xgb = np.zeros(len(test2))
pos_shift = hparams.pop('pos_shift')  
for fold, (train_index, test_index) in enumerate(skf.split(train2, train2.race_group)):
    print("#" * 25)
    print(f"### Fold {fold + 1}")
    print("#" * 25)
    x_train = train2.loc[train_index, FEATURES]
    y_train = train2.loc[train_index, "label"]
    x_valid = train2.loc[test_index, FEATURES]
    y_valid = train2.loc[test_index, "label"]
    x_test = test2[FEATURES].copy()
    hparams.update({
        'objective': 'reg:pseudohubererror',
        'max_cat_to_onehot': 10,
        'enable_categorical': True,
        'random_state': 42,
        'monotone_constraints': {
            'hla_high_res_6': -1,
            'hla_high_res_8': -1,
            'hla_low_res_6': -1,
            'hla_match_a_high': -1,
            'hla_match_drb1_low': -1,
            'hla_match_c_low': -1,
            'hla_match_dqb1_low': -1,
            'hla_nmdp_6': -1,
        }
    })
    
    print("n_estimators:", hparams['n_estimators'])
    model_xgb = XGBRegressor(**hparams)
    
    tt = train2.loc[train_idx]
    weights = np.array([0.95 if flag else 1 for flag in ((tt.efs_time < 24) & (tt.efs == 0)).values])
    weights /= np.array([1.3 if flag else 1 for flag in (tt.efs_time > 36)])
    
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        verbose=100,
        sample_weight=weights
    )
    
    oof_preds[test_index] = model_xgb.predict(x_valid)
    pred_xgb += model_xgb.predict(x_test)

pred_xgb /= FOLDS


y_pred_df = train2[['ID']].copy()
y_pred_df["prediction"] = oof_preds
m = score(train.copy(), y_pred_df.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier = {m}")


%%time
seed = 42
DO_Tuning = False  
def lgb_objective(trial):
    params = {
        'objective':         'regression',  
        'verbosity':         -1,
        'n_estimators':      2500, 
        'boosting_type':     'gbdt',
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),  
        'max_depth':         trial.suggest_int('max_depth', 3, 9), 
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300), 
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 10, 200),
    }

    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        
    oof_lgb = np.zeros(len(train2))
    
    for i, (train_index, test_index) in enumerate(kf.split(train2)):
        x_train = train2.loc[train_index, FEATURES]
        y_train = train2.loc[train_index, "label"]
        x_valid = train2.loc[test_index, FEATURES]
        y_valid = train2.loc[test_index, "label"]
        
        model_lgb = LGBMRegressor(**params)
        model_lgb.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric='rmse'
        )
        
        oof_lgb[test_index] = model_lgb.predict(x_valid)
    
    # Calculate RMSE across all folds
    rmse = np.sqrt(mean_squared_error(train2["label"], oof_lgb))
    return rmse
if DO_Tuning:
    start_time = time.time()
    study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))  
    study_lgb.optimize(lgb_objective, n_trials=100)
    end_time = time.time()
    elapsed_time_lgb = end_time - start_time
    print(f"LightGBM tuning took {elapsed_time_lgb:.2f} seconds.")

    best_params = study_lgb.best_params
    lgb_params = {
        'objective': 'regression',
        'verbosity': -1,
        'n_estimators': 2500,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'metric': 'None',
        **best_params
    }
else:
    # Original parameters if tuning is disabled
    lgb_params = {
        'n_estimators': 2500,
        'objective': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'metric': 'None',
        'lambda_l1': 7.518992942960153e-08,
        'lambda_l2': 3.263100871327459e-05,
        'learning_rate': 0.009291595507188742,
        'max_depth': 5,
        'num_leaves': 152,
        'colsample_bytree': 0.45578411461906315,
        'colsample_bynode': 0.8600988396886319,
        'bagging_fraction': 0.6050331021718235,
        'bagging_freq': 7,
        'min_data_in_leaf': 110
    }


lgb_params


%%time
    
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train2))
pred_lgb = np.zeros(len(test2))

for i, (train_index, test_index) in enumerate(kf.split(train2)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train2.loc[train_index,FEATURES].copy()
    y_train = train2.loc[train_index,"label"]    
    x_valid = train2.loc[test_index,FEATURES].copy()
    y_valid = train2.loc[test_index,"label"]
    x_test = test2[FEATURES].copy()

    model_lgb = LGBMRegressor(**lgb_params)
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)]
    )
    
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    pred_lgb += model_lgb.predict(x_test)

pred_lgb /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb


%%time
%%time
seed = 42
DO_Tuning = False  

def cb_objective(trial):
    params = {
        'loss_function': 'RMSE',
        'iterations': 2500,
        'grow_policy': 'Lossguide',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1, log=True),
        'max_depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'random_state': seed,
        'verbose': False,
    }

    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        
    oof_cb = np.zeros(len(train2))
    
    for i, (train_index, test_index) in enumerate(kf.split(train2)):
        x_train = train2.loc[train_index, FEATURES]
        y_train = train2.loc[train_index, "label"]
        x_valid = train2.loc[test_index, FEATURES]
        y_valid = train2.loc[test_index, "label"]
        
        model_cb = CatBoostRegressor(**params)
        model_cb.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            cat_features=CAT_FEATURES
        )
        
        # Corrected: Store predictions in oof_cb instead of oof_lgb
        oof_cb[test_index] = model_cb.predict(x_valid)
    
    # Calculate RMSE across all folds
    rmse = np.sqrt(mean_squared_error(train2["label"], oof_cb))
    return rmse

if DO_Tuning:
    start_time = time.time()
    study_cb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))  
    study_cb.optimize(cb_objective, n_trials=100)
    end_time = time.time()
    elapsed_time_cb = end_time - start_time
    print(f"CB tuning took {elapsed_time_cb:.2f} seconds.")

    best_params = study_cb.best_params
    # Ensure that fixed parameters are included and use consistent parameter names (e.g. 'depth', 'random_seed')
    cb_params = {
        **best_params,
        'loss_function': 'RMSE',
        'iterations': 2500,
        'verbose': False,
        'random_seed': seed
    }

else:
    cb_params = {
        'loss_function':     'RMSE',
        'iterations':        2500,
        'verbose':           False,
        'random_state':      seed,
        'bagging_temperature': 0.50,
        'learning_rate': 0.1,
        'max_depth': 8,
        'l2_leaf_reg': 1.25,
        'min_data_in_leaf': 24,
        'random_strength' : 0.25
            }


cb_params


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train2))
pred_cat = np.zeros(len(test2))

for i, (train_index, test_index) in enumerate(kf.split(train2)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train2.loc[train_index,FEATURES].copy()
    y_train = train2.loc[train_index,"label"]
    x_valid = train2.loc[test_index,FEATURES].copy()
    y_valid = train2.loc[test_index,"label"]
    x_test = test2[FEATURES].copy()

    model_cat = CatBoostRegressor(**cb_params)
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CAT_FEATURES,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


import pandas
y_true = train2[["ID","efs","efs_time","race_group"]].copy()
y_pred = train2[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


from scipy.stats import rankdata 

y_true = train2[["ID","efs","efs_time","race_group"]].copy()
y_pred = train2[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_lgb) + rankdata(oof_cat) + rankdata(oof_preds)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)





sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_lgb) + rankdata(pred_cat) + rankdata(pred_xgb)
sub.to_csv("submission.csv",index=False)


sub

