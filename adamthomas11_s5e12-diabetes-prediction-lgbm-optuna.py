# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy.stats import chi2_contingency
from itertools import combinations

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
import lightgbm as lgb
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from optuna.exceptions import TrialPruned

# Warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# dataframes
df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# test ids (used for submission)
test_ids = test_df['id'].tolist()

# inspect columns
df.columns


def bin_age(df, col='age'):
    bins = [-1, 29, 44, 59, 74, float('inf')]
    labels = ["young", "early_middle", "middle_age", "senior", "elderly"]
    
    df['age_group'] = pd.cut(df[col], bins=bins, labels=labels)
    df['age_sq'] = df[col] **2
    df['age_group_ord'] = df['age_group'].cat.codes
    return df

df = bin_age(df)
test_df = bin_age(test_df)

df.groupby('age_group')['diagnosed_diabetes'].mean()


def bin_alcohol(df, col='alcohol_consumption_per_week'):
    bins = [1, 3, 6, 9]
    labels = ["light", "moderate", "heavy"]
    df['alcohol_group'] = pd.cut(df[col], bins=bins, labels=labels)
    df['alcohol_log'] = np.log1p(df[col])
    return df

df = bin_alcohol(df)
test_df = bin_alcohol(test_df)

df.groupby('alcohol_group')['diagnosed_diabetes'].mean()


def bin_activity(df, col='physical_activity_minutes_per_week'):
    bins = [-1, 149, 299, float('inf')]
    labels = ['inactive', 'moderately_active', 'highly_active']
    
    df['physical_activity_group'] = pd.cut(df[col], bins=bins, labels=labels)
    df['physical_activity_log'] = np.log1p(df['physical_activity_minutes_per_week'])
    return df

df = bin_activity(df)
test_df = bin_activity(test_df)
df.groupby('physical_activity_group')['diagnosed_diabetes'].mean()


# Unsure what diet score refers to / real life categories therefore I cut the data into quartiles
def bin_diet(df, col='diet_score'):
    df['diet_group'] = pd.qcut(df['diet_score'], q=4, 
                               labels=['poor', 'poor_to_medium', 'medium_to_good', 'good'])

    return df

df = bin_diet(df)
test_df = bin_diet(test_df)

df.groupby('diet_group')['diagnosed_diabetes'].mean()


'sleep_hours_per_day'

def bin_sleep(df, col='sleep_hours_per_day'):
    bins = [0, 7, 9, float('inf')] 
    labels = ['short', 'normal', 'long']
    
    df['sleep_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_sleep(df)
test_df = bin_sleep(test_df)

df.groupby('sleep_group')['diagnosed_diabetes'].mean()


def bin_screen_time(df, col='screen_time_hours_per_day'):
    df['screen_time_group'] = pd.qcut(df['screen_time_hours_per_day'], q=4, 
                               labels=['low', 'medium_low', 'medium_high', 'high'])

    return df

df = bin_screen_time(df)
test_df = bin_screen_time(test_df)

df.groupby('screen_time_group')['diagnosed_diabetes'].mean()


def bin_bmi(df, col='bmi'):
    bins = [0, 18.5, 25, 30, float('inf')] 
    labels = ['underweight', 'normal', 'overweight', 'obese']
    
    df['bmi_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_bmi(df)
test_df = bin_bmi(test_df)

df.groupby('bmi_group')['diagnosed_diabetes'].mean()


def bin_waist_hip(df, col='waist_to_hip_ratio'):
    df['waist_hip_group'] = pd.qcut(df['waist_to_hip_ratio'], q=4, 
                               labels=['low', 'medium_low', 'medium_high', 'high'])

    return df

df = bin_waist_hip(df)
test_df = bin_waist_hip(test_df)

df.groupby('waist_hip_group')['diagnosed_diabetes'].mean()


def bin_systolic_bp(df, col='systolic_bp'):
    bins = [0, 120, 130, 140, 180, float('inf')]
    labels = ['normal', 'elevated', 'hypertension_stage_1', 'hypertension_stage_2', 'hypertensive_crisis']
    
    df['systolic_bp_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_systolic_bp(df)
test_df = bin_systolic_bp(test_df)


df.groupby('systolic_bp_group')['diagnosed_diabetes'].mean()


def bin_diastolic_bp(df, col='diastolic_bp'):
    bins = [0, 80, 90, 100, 120, float('inf')]
    labels = ['normal', 'prehypertension', 'hypertension_stage_1', 'hypertension_stage_2', 'hypertensive_crisis']
    
    df['diastolic_bp_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_diastolic_bp(df)
test_df = bin_diastolic_bp(test_df)

df.groupby('diastolic_bp_group')['diagnosed_diabetes'].mean()



def bin_heart_rate(df, col='heart_rate'):
    bins = [0, 60, 100, float('inf')]
    labels = ['bradycardia', 'normal', 'tachycardia']
    
    df['heart_rate_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_heart_rate(df)
test_df = bin_heart_rate(test_df)

df.groupby('heart_rate_group')['diagnosed_diabetes'].mean()


def bin_total_cholesterol(df, col='cholesterol_total'):
    bins = [0, 200, 240, float('inf')]
    labels = ['desirable', 'borderline_high', 'high']
    
    df['total_cholesterol_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_total_cholesterol(df)
test_df = bin_total_cholesterol(test_df)

df.groupby('total_cholesterol_group')['diagnosed_diabetes'].mean()



def bin_hdl(df, col='hdl_cholesterol'):
    bins = [0, 40, 60, float('inf')]
    labels = ['low', 'normal', 'high']
    
    df['hdl_cholesterol_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_hdl(df)
test_df = bin_hdl(test_df)

df.groupby('hdl_cholesterol_group')['diagnosed_diabetes'].mean()


def bin_ldl(df, col='ldl_cholesterol'):
    bins = [0, 100, 130, 160, 190, float('inf')]
    labels = ['optimal', 'near_optimal', 'borderline_high', 'high', 'very_high']
    
    df['ldl_cholesterol_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_ldl(df)
test_df = bin_ldl(test_df)

df.groupby('ldl_cholesterol_group')['diagnosed_diabetes'].mean()


def bin_triglycerides(df, col='triglycerides'):
    bins = [0, 150, 200, 500, float('inf')]
    labels = ['normal', 'borderline_high', 'high', 'very_high']
    
    df['triglycerides_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

df = bin_triglycerides(df)
test_df = bin_triglycerides(test_df)

df.groupby('triglycerides_group')['diagnosed_diabetes'].mean()


# Cramér's V function
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, dof, ex = chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape)
    if k == 1:
        return 0.0
    return np.sqrt(chi2 / (n * (k-1)))

categorical_groups = [
    'bmi_group', 'waist_hip_group', 'systolic_bp_group', 'diastolic_bp_group',
    'heart_rate_group', 'total_cholesterol_group', 'hdl_cholesterol_group',
    'ldl_cholesterol_group', 'triglycerides_group', 'sleep_group', 
    'diet_group', 'physical_activity_group', 'alcohol_group'
]

results = []

for var in categorical_groups:
    v = cramers_v(df[var], df['diagnosed_diabetes'])
    results.append({'variable': var, 'cramers_v': v})

effect_sizes = pd.DataFrame(results).sort_values('cramers_v', ascending=False)
print(effect_sizes)


results = []

# pairwise combinations
for var1, var2 in combinations(categorical_groups, 2):
    interaction_var = df[var1].astype(str) + "_" + df[var2].astype(str)
    # Compute Cramér's V with diagnosed_diabetes
    v = cramers_v(interaction_var, df['diagnosed_diabetes'])
    results.append({'var1': var1, 'var2': var2, 'cramers_v': v})
    
    # Add if effect size is >= 0.10
    if v >= 0.10:
        new_col_name = f"{var1}_{var2}_interaction"
        df[new_col_name] = interaction_var
        test_df[new_col_name] = interaction_var

interaction_effects = pd.DataFrame(results).sort_values('cramers_v', ascending=False)



print(interaction_effects.head(50))


top_interaction = interaction_effects.iloc[0]
var1 = top_interaction['var1']
var2 = top_interaction['var2']

df['interaction'] = df[var1].astype(str) + "_" + df[var2].astype(str)

ct = df.groupby([var1, var2])['diagnosed_diabetes'].mean().unstack()

plt.figure(figsize=(10,7))
sns.heatmap(ct, annot=True, fmt=".2f", cmap='coolwarm')
plt.title(f"Mean Diabetes Rate: {var1} × {var2}")
plt.ylabel(var1)
plt.xlabel(var2)
plt.show()

df = df.drop('interaction', axis = 1)


def num_fe(df):

    ## Cardiometabolic
    # Adiposity
    df['bmi_waisthip_interaction'] = df['bmi'] * df['waist_to_hip_ratio']        # BMI × central obesity
    
    # Blood pressure
    df['systolic_diastolic_interaction'] = df['systolic_bp'] * df['diastolic_bp'] # multiplicative effect
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']                 # systolic - diastolic
    
    # Lipid ratios
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    df['triglycerides_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-6)
    
    ## Lifestyle / Bio
    # Physical activity and adiposity
    df['activity_bmi_interaction'] = df['physical_activity_minutes_per_week'] * df['bmi']
    df['activity_waisthip_interaction'] = df['physical_activity_minutes_per_week'] * df['waist_to_hip_ratio']
    
    # Sleep and diet
    df['sleep_diet_interaction'] = df['sleep_hours_per_day'] * df['diet_score']
    
    # Alcohol and lipids
    df['alcohol_triglycerides_interaction'] = df['alcohol_consumption_per_week'] * df['triglycerides']
    df['alcohol_ldl_interaction'] = df['alcohol_consumption_per_week'] * df['ldl_cholesterol']
    
    ## Sedentary
    # Screen time × activity
    df['screen_activity_interaction'] = df['screen_time_hours_per_day'] * df['physical_activity_minutes_per_week']
    
    # Screen time × BMI
    df['screen_bmi_interaction'] = df['screen_time_hours_per_day'] * df['bmi']
    
    ##Age
    # Age × adiposity
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['age_waisthip_interaction'] = df['age'] * df['waist_to_hip_ratio']
    
    # Age × blood pressure
    df['age_systolic_interaction'] = df['age'] * df['systolic_bp']
    df['age_diastolic_interaction'] = df['age'] * df['diastolic_bp']

    return df

df = num_fe(df)
test_df = num_fe(test_df)


df.columns


df.dtypes.astype(str).value_counts()


# features dropped
dropped = ['bmi_group',
           'hypertension_history',
           'alcohol_group',
           'hdl_cholesterol_group',
           'age_group_ord',
           'triglycerides_group',
           'total_cholesterol_group',
           'waist_hip_group',
           'ldl_cholesterol_group',
           'physical_activity_group',
           'diastolic_bp_group',
           'systolic_bp_group']

df = df.drop(columns = dropped)
test_df = test_df.drop(columns = dropped)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

X = df.drop(columns=['id', 'diagnosed_diabetes'])
y = df['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y
)

for col in categorical_cols:
    if col in X_val.columns:
        X_val[col] = X_val[col].astype('category')
        X_val[col] = X_val[col].cat.set_categories(X_train[col].cat.categories)


X_test = test_df.drop(columns=['id'], errors='ignore').copy()

for col in categorical_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')
        X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)
    else:
     # If the test set is missing a categorical column, create it as NaN with the same categories
        X_test[col] = pd.Categorical([np.nan]*len(X_test), categories=X_train[col].cat.categories)

missing_cols = set(categorical_cols) - set(X_test.columns)
if missing_cols:
    print("Warning: These categorical columns were missing in test and filled with NaN:", missing_cols)



SEED = 5625

def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,

        "learning_rate": trial.suggest_float("learning_rate", 0.010, 0.030, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 150, 260),
        "max_depth": trial.suggest_int("max_depth", 5, 10),

        "min_child_samples": trial.suggest_int("min_child_samples", 60, 140),
        "min_split_gain": trial.suggest_float("min_split_gain", 1.0, 2.2),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.5, 6.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.5, log=True),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.50, 0.80),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.70, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 4, 10),

        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 90, 180),
        "max_bin": trial.suggest_int("max_bin", 200, 255),
        
        # Seeds for reproducibility
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
    
        # Categorical controls
        "max_cat_to_onehot": 4,
        "cat_l2": 10,
        "cat_smooth": 10,
    
        # 
        "n_estimators": 5000
}

    # CV seed (controls train/val split pattern)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5625)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # category alignment
        for col in categorical_cols:
            X_va.loc[:, col] = X_va[col].cat.set_categories(
                X_tr[col].cat.categories
            )

        model = lgb.LGBMClassifier(**params)

        lr = params['learning_rate']

        if lr >= 0.07:
            patience = 20
        elif lr >= 0.04:
            patience = 30
        elif lr >= 0.02:
            patience = 50
        else:
            patience = 80

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            categorical_feature=categorical_cols,
            callbacks=[
                early_stopping(stopping_rounds=patience),
                log_evaluation(0)
            ]
        )

        preds = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, preds)
        cv_scores.append(fold_auc)

        # Optuna pruning
        trial.report(fold_auc, step=fold)
        if trial.should_prune():
            raise TrialPruned()

    return np.mean(cv_scores)


# Optuna sampler with seed (controls hyperparameter selection randomness)
#sampler = optuna.samplers.TPESampler(seed=SEED)

#study = optuna.create_study(direction='maximize', sampler=sampler)
#study.optimize(objective, n_trials=50)



#print("Best params:", study.best_trial.params)
#print("Best CV AUC:", study.best_value)

#best_params = study.best_trial.params



"""SEED =5625

lgbm_final = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'learning_rate': 0.011834199286647737, 
    'num_leaves': 159, 
    'max_depth': 8, 
    'min_child_samples': 104, 
    'min_split_gain': 1.9233750581297469, 
    'lambda_l1': 2.695069680940212, 
    'lambda_l2': 0.053240020266792254, 
    'feature_fraction': 0.5003052071960683, 
    'bagging_fraction': 0.7994568245976457, 
    'bagging_freq': 8, 
    'min_data_in_leaf': 176, 
    'max_bin': 246,
    'seed': SEED,
    'feature_fraction_seed': SEED,
    "bagging_seed": SEED,
    'max_cat_to_onehot': 4,
    'cat_l2': 10,
    'cat_smooth': 10, 
    'n_estimators': 5000
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5625)

cv_scores = []
all_importances = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    X_tr, X_va = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    # ---- fix category alignment ----
    for col in categorical_cols:
        X_va[col] = X_va[col].cat.set_categories(X_tr[col].cat.categories)

    model = lgb.LGBMClassifier(**lgbm_wide)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        categorical_feature=categorical_cols,
        callbacks=[
            early_stopping(stopping_rounds=80),
            log_evaluation(0)
        ]
    )

    preds = model.predict_proba(X_va)[:, 1]
    score = roc_auc_score(y_va, preds)
    cv_scores.append(score)

    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.booster_.feature_importance(importance_type="gain"),
        "fold": fold
    })
    all_importances.append(fold_importance)

importances_df = pd.concat(all_importances, axis=0)

importance_summary = (
    importances_df
    .groupby("feature")["importance"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
    .reset_index()
)

importance_summary.rename(
    columns={"mean": "importance_mean", "std": "importance_std"},
    inplace=True
)

importance_summary["importance_norm"] = (
    importance_summary["importance_mean"] /
    importance_summary["importance_mean"].sum()
)

importance_summary"""


#low_features = importance_summary[
#    importance_summary["importance_mean"] == 0]["feature"].tolist()

#low_features


#to_drop = importance_summary.query("importance_norm < 0.00005")["feature"].tolist()

#to_drop


SEEDS = [5625, 1111, 2222, 3333, 33]

base_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'learning_rate': 0.011834199286647737, 
    'num_leaves': 159, 
    'max_depth': 8, 
    'min_child_samples': 104, 
    'min_split_gain': 1.9233750581297469, 
    'lambda_l1': 2.695069680940212, 
    'lambda_l2': 0.053240020266792254, 
    'feature_fraction': 0.5003052071960683, 
    'bagging_fraction': 0.7994568245976457, 
    'bagging_freq': 8, 
    'min_data_in_leaf': 176, 
    'max_bin': 246,
    'max_cat_to_onehot': 4,
    'cat_l2': 10,
    'cat_smooth': 10, 
    'n_estimators': 5000
}


def train_single_seed(seed, X, y, categorical_cols):
    
    params = base_params.copy()
    params['seed'] = seed
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5625)

    oof = np.zeros(len(X))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):

        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # category alignment
        for col in categorical_cols:
            X_va[col] = X_va[col].cat.set_categories(X_tr[col].cat.categories)

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            categorical_feature=categorical_cols,
            callbacks=[early_stopping(80), log_evaluation(0)]
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        models.append(model)

    return oof, models


oof_all = []
model_bag = []

for seed in SEEDS:
    print(f"Training seed {seed}")
    oof, models = train_single_seed(seed, X, y, categorical_cols)
    oof_all.append(oof)
    model_bag.append(models)


oof_ensemble = np.mean(oof_all, axis=0)
auc_ensemble = roc_auc_score(y, oof_ensemble)
print("Ensembled OOF AUC:", auc_ensemble)


def predict_single_seed(models, X_test, categorical_cols, X_train_reference):
    """
    models: list of trained models returned by train_single_seed
    X_test: test dataframe
    categorical_cols: categorical columns used in training
    X_train_reference: training dataframe used to extract category sets
    """

    X_test = X_test.copy()

    for col in categorical_cols:
        X_test[col] = X_test[col].cat.set_categories(
            X_train_reference[col].cat.categories
        )

    # Predict for each model and collect results in a dict
    preds = {}
    for i, model in enumerate(models):
        preds[f"model_{i}"] = model.predict_proba(X_test)[:, 1]

    # Return df
    return pd.DataFrame(preds, index=X_test.index)


test_pred_df = predict_single_seed(models, X_test, categorical_cols, X_train)


test_pred_df['avg'] = test_pred_df.mean(axis=1)


test_pred_df


prediction = test_pred_df['avg'].tolist()

submission_df = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': prediction
})

submission_df.to_csv('submission.csv', index=False)


submission_df

