# Preprocessing and Modeling
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classifiers and ensembling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
# ---------------------------
# 1. Data Loading and Basic Feature Engineering
# ---------------------------
# Load datasets from Kaggle input paths
train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")


train.head()


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# ---------------------------
# 2. Convert NAME_CONTRACT_TYPE to 0/1
# ---------------------------
# Now you can continue with further feature engineering or modeling steps
# For example:
# ---------------------------
# 3. Additional Feature Engineering
# ---------------------------

train['NAME_CONTRACT_TYPE'] = train['NAME_CONTRACT_TYPE'].map({
    'Cash loans': 0,
    'Revolving loans': 1
})

test['NAME_CONTRACT_TYPE'] = test['NAME_CONTRACT_TYPE'].map({
    'Cash loans': 0,
    'Revolving loans': 1
})
train['CODE_GENDER'] = train['CODE_GENDER'].map({
    'M': 0,
    'F': 1
})

test['CODE_GENDER'] = test['CODE_GENDER'].map({
    'M': 0,
    'F': 1
})
train['FLAG_OWN_CAR'] = train['FLAG_OWN_CAR'].map({
    'N': 0,
    'Y': 1
})

test['FLAG_OWN_CAR'] = test['FLAG_OWN_CAR'].map({
    'N': 0,
    'Y': 1
})

train['FLAG_OWN_REALTY'] = train['FLAG_OWN_REALTY'].map({
    'N': 0,
    'Y': 1
})

test['FLAG_OWN_REALTY'] = test['FLAG_OWN_REALTY'].map({
    'N': 0,
    'Y': 1
})
# Fit the scaler on the training data and transform both train and test data
cols_to_scale = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

# Fit the scaler on the training data and transform both train and test data
scaled_train = scaler.fit_transform(train[cols_to_scale])
scaled_test = scaler.transform(test[cols_to_scale])

# Assign the scaled values to new columns in the respective DataFrames
train[['AMT_INCOME_TOTAL_NORM', 'AMT_CREDIT_NORM', 'AMT_ANNUITY_NORM']] = scaled_train
test[['AMT_INCOME_TOTAL_NORM', 'AMT_CREDIT_NORM', 'AMT_ANNUITY_NORM']] = scaled_test




from sklearn.preprocessing import OneHotEncoder

# Initialize the encoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Fit and transform the training data, then transform the test data
suite_train_encoded = ohe.fit_transform(train[['NAME_TYPE_SUITE']])
suite_test_encoded = ohe.transform(test[['NAME_TYPE_SUITE']])

# Convert the encoded arrays into DataFrames with appropriate column names
suite_train_df = pd.DataFrame(suite_train_encoded, 
                              columns=ohe.get_feature_names_out(['NAME_TYPE_SUITE']),
                              index=train.index)
suite_test_df = pd.DataFrame(suite_test_encoded, 
                             columns=ohe.get_feature_names_out(['NAME_TYPE_SUITE']),
                             index=test.index)

# Drop the original NAME_TYPE_SUITE column and concatenate the new one-hot encoded columns
train = pd.concat([train.drop(columns=['NAME_TYPE_SUITE']), suite_train_df], axis=1)
test = pd.concat([test.drop(columns=['NAME_TYPE_SUITE']), suite_test_df], axis=1)

# Verify the transformation by checking the first few rows of the new columns
print(train.filter(like='NAME_TYPE_SUITE').head())


import pandas as pd

# List of categorical columns to encode
categorical_cols = [
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE'
]

# One-hot encode using pandas on CPU
train = pd.get_dummies(train, columns=categorical_cols, prefix=categorical_cols)
test = pd.get_dummies(test, columns=categorical_cols, prefix=categorical_cols)

# Align train and test
train, test = train.align(test, join='left', axis=1, fill_value=0)


import re
import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
#                        [Topic: Main Execution Pipeline]                     #
###############################################################################
def execute_main_pipeline(debug_mode=False):
    row_limit = 30000 if debug_mode else None
    
    with time_tracker("Loading and Merging application_train & application_test"):
        application_df = load_application_data(DATA_DIR, num_rows=row_limit)
        print("Application DataFrame shape: ", application_df.shape)
    
    with time_tracker("Loading and Merging Bureau & Bureau Balance"):
        bureau_data = process_bureau_data(DATA_DIR, num_rows=row_limit)
        application_df = pd.merge(application_df, bureau_data, on='SK_ID_CURR', how='left')
        print("Bureau DataFrame shape: ", bureau_data.shape)
        del bureau_data; gc.collect()
    
    with time_tracker("Processing Previous Applications"):
        previous_data = process_previous_applications(DATA_DIR, row_limit)
        application_df = pd.merge(application_df, previous_data, on='SK_ID_CURR', how='left')
        print("Previous DataFrame shape: ", previous_data.shape)
        del previous_data; gc.collect()
    
    with time_tracker("Merging POS-CASH, Installments, Credit Card"):
        pos_cash_data = process_pos_cash_data(DATA_DIR, row_limit)
        application_df = pd.merge(application_df, pos_cash_data, on='SK_ID_CURR', how='left')
        print("POS-CASH DataFrame shape: ", pos_cash_data.shape)
        del pos_cash_data; gc.collect()

        installments_data = process_installment_payments(DATA_DIR, row_limit)
        application_df = pd.merge(application_df, installments_data, on='SK_ID_CURR', how='left')
        print("Installments DataFrame shape: ", installments_data.shape)
        del installments_data; gc.collect()

        credit_card_data = process_credit_card_data(DATA_DIR, row_limit)
        application_df = pd.merge(application_df, credit_card_data, on='SK_ID_CURR', how='left')
        print("Credit Card DataFrame shape: ", credit_card_data.shape)
        del credit_card_data; gc.collect()
        
    application_df = add_custom_ratio_features(application_df)
    application_df = reduce_memory_usage(application_df)
    
    cat_features_for_lightgbm = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'WALLSMATERIAL_MODE'
    ]
    
    with time_tracker("Run LightGBM with K-Fold"):
        feature_importance_df = train_lightgbm_kfold(application_df, cat_features_for_lightgbm)
        print(feature_importance_df)

###############################################################################
#                 [Topic: Feature Engineering - Custom Ratios]               #
###############################################################################
def add_custom_ratio_features(main_df):
    main_df['BUREAU_INCOME_CREDIT_RATIO'] = main_df['BUREAU_AMT_CREDIT_SUM_MEAN'] / main_df['AMT_INCOME_TOTAL']
    main_df['BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO'] = main_df['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM'] / main_df['AMT_INCOME_TOTAL']
    
    main_df['CURRENT_TO_APPROVED_CREDIT_MIN_RATIO'] = main_df['APPROVED_AMT_CREDIT_MIN'] / main_df['AMT_CREDIT']
    main_df['CURRENT_TO_APPROVED_CREDIT_MAX_RATIO'] = main_df['APPROVED_AMT_CREDIT_MAX'] / main_df['AMT_CREDIT']
    main_df['CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO'] = main_df['APPROVED_AMT_CREDIT_MEAN'] / main_df['AMT_CREDIT']
    
    main_df['CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO'] = main_df['APPROVED_AMT_ANNUITY_MAX'] / main_df['AMT_ANNUITY']
    main_df['CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO'] = main_df['APPROVED_AMT_ANNUITY_MEAN'] / main_df['AMT_ANNUITY']
    main_df['PAYMENT_MIN_TO_ANNUITY_RATIO'] = main_df['INS_AMT_PAYMENT_MIN'] / main_df['AMT_ANNUITY']
    main_df['PAYMENT_MAX_TO_ANNUITY_RATIO'] = main_df['INS_AMT_PAYMENT_MAX'] / main_df['AMT_ANNUITY']
    main_df['PAYMENT_MEAN_TO_ANNUITY_RATIO'] = main_df['INS_AMT_PAYMENT_MEAN'] / main_df['AMT_ANNUITY']
    
    main_df['CTA_CREDIT_TO_ANNUITY_MAX_RATIO'] = main_df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX'] / main_df['CREDIT_TO_ANNUITY_RATIO']
    main_df['CTA_CREDIT_TO_ANNUITY_MEAN_RATIO'] = main_df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN'] / main_df['CREDIT_TO_ANNUITY_RATIO']
    
    main_df['DAYS_DECISION_MEAN_TO_BIRTH'] = main_df['APPROVED_DAYS_DECISION_MEAN'] / main_df['DAYS_BIRTH']
    main_df['DAYS_CREDIT_MEAN_TO_BIRTH'] = main_df['BUREAU_DAYS_CREDIT_MEAN'] / main_df['DAYS_BIRTH']
    main_df['DAYS_DECISION_MEAN_TO_EMPLOYED'] = main_df['APPROVED_DAYS_DECISION_MEAN'] / main_df['DAYS_EMPLOYED']
    main_df['DAYS_CREDIT_MEAN_TO_EMPLOYED'] = main_df['BUREAU_DAYS_CREDIT_MEAN'] / main_df['DAYS_EMPLOYED']
    
    return main_df

###############################################################################
#                [Topic: Model Training - LightGBM K-Fold]                    #
###############################################################################
def train_lightgbm_kfold(dataframe, categorical_feature=None):
    df_train = dataframe[dataframe['TARGET'].notnull()]
    df_test = dataframe[dataframe['TARGET'].isnull()]
    
    df_train = df_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df_test = df_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    print("Train/valid shape: {}, test shape: {}".format(df_train.shape, df_test.shape))
    
    columns_to_drop = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    predictors = [col for col in df_train.columns if col not in columns_to_drop]

    if not STRATIFIED_KFOLD:
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    else:
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    eval_results = dict()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[predictors], df_train['TARGET'])):
        train_x, train_y = df_train[predictors].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[predictors].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]

        params = {'random_state': RANDOM_SEED, 'nthread': NUM_THREADS}
        model = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS})
        
        if not categorical_feature:
            model.fit(train_x, train_y,
                      eval_set=[(train_x, train_y), (valid_x, valid_y)],
                      eval_metric='auc', verbose=400, early_stopping_rounds=EARLY_STOPPING)
        else:
            model.fit(train_x, train_y,
                      eval_set=[(train_x, train_y), (valid_x, valid_y)],
                      eval_metric='auc', verbose=400, early_stopping_rounds=EARLY_STOPPING,
                      feature_name=list(df_train[predictors].columns),
                      categorical_feature=categorical_feature)

        oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
        sub_preds += model.predict_proba(df_test[predictors], num_iteration=model.best_iteration_)[:, 1] / folds.n_splits

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = predictors
        fold_importance["gain"] = model.booster_.feature_importance(importance_type='gain')
        fold_importance["split"] = model.booster_.feature_importance(importance_type='split')
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
        
        eval_results[f'train_{n_fold+1}'] = model.evals_result_['training']['auc']
        eval_results[f'valid_{n_fold+1}'] = model.evals_result_['valid_1']['auc']

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(df_train['TARGET'], oof_preds))
    df_test['TARGET'] = sub_preds.copy()

    mean_importance = feature_importance_df.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by='gain', ascending=False, inplace=True)
    
    if GENERATE_SUBMISSION_FILES:
        df_train['PREDICTIONS'] = oof_preds.copy()
        df_train.to_csv(f'oof{SUBMISSION_SUFIX}.csv', index=False)
        df_test[['SK_ID_CURR', 'TARGET']].to_csv(f'submission{SUBMISSION_SUFIX}.csv', index=False)
        mean_importance.to_csv(f'feature_importance{SUBMISSION_SUFIX}.csv', index=False)
    
    return mean_importance

###############################################################################
#           [Topic: Data Preprocessing - Application & Bureau Loading]        #
###############################################################################
def load_application_data(path, num_rows=None):
    train_data = pd.read_csv(os.path.join(path, 'application_train.csv'), nrows=num_rows)
    test_data = pd.read_csv(os.path.join(path, 'application_test.csv'), nrows=num_rows)
    
    merged_df = pd.concat([train_data, test_data])
    del train_data, test_data; gc.collect()
    
    merged_df = merged_df[merged_df['CODE_GENDER'] != 'XNA']
    merged_df = merged_df[merged_df['AMT_INCOME_TOTAL'] < 20000000]
    merged_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    merged_df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    doc_cols = [col for col in merged_df.columns if 'FLAG_DOC' in col]
    merged_df['DOCUMENT_COUNT'] = merged_df[doc_cols].sum(axis=1)
    merged_df['NEW_DOC_KURT'] = merged_df[doc_cols].kurtosis(axis=1)
    
    merged_df['AGE_RANGE'] = merged_df['DAYS_BIRTH'].apply(lambda x: categorize_age_group(x))

    merged_df['EXT_SOURCES_PROD'] = merged_df['EXT_SOURCE_1'] * merged_df['EXT_SOURCE_2'] * merged_df['EXT_SOURCE_3']
    merged_df['EXT_SOURCES_WEIGHTED'] = merged_df['EXT_SOURCE_1'] * 2 + merged_df['EXT_SOURCE_2'] * 1 + merged_df['EXT_SOURCE_3'] * 3
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    
    for fn_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feat_name = 'EXT_SOURCES_{}'.format(fn_name.upper())
        merged_df[feat_name] = eval('np.{}'.format(fn_name))(merged_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']], axis=1)

    merged_df['CREDIT_TO_ANNUITY_RATIO'] = merged_df['AMT_CREDIT'] / merged_df['AMT_ANNUITY']
    merged_df['CREDIT_TO_GOODS_RATIO'] = merged_df['AMT_CREDIT'] / merged_df['AMT_GOODS_PRICE']
    
    merged_df['ANNUITY_TO_INCOME_RATIO'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_INCOME_TOTAL']
    merged_df['CREDIT_TO_INCOME_RATIO'] = merged_df['AMT_CREDIT'] / merged_df['AMT_INCOME_TOTAL']
    merged_df['INCOME_TO_EMPLOYED_RATIO'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['DAYS_EMPLOYED']
    merged_df['INCOME_TO_BIRTH_RATIO'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['DAYS_BIRTH']
   
    merged_df['EMPLOYED_TO_BIRTH_RATIO'] = merged_df['DAYS_EMPLOYED'] / merged_df['DAYS_BIRTH']
    merged_df['ID_TO_BIRTH_RATIO'] = merged_df['DAYS_ID_PUBLISH'] / merged_df['DAYS_BIRTH']
    merged_df['CAR_TO_BIRTH_RATIO'] = merged_df['OWN_CAR_AGE'] / merged_df['DAYS_BIRTH']
    merged_df['CAR_TO_EMPLOYED_RATIO'] = merged_df['OWN_CAR_AGE'] / merged_df['DAYS_EMPLOYED']
    merged_df['PHONE_TO_BIRTH_RATIO'] = merged_df['DAYS_LAST_PHONE_CHANGE'] / merged_df['DAYS_BIRTH']

    grp_cols = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
    merged_df = compute_median(merged_df, grp_cols, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
    merged_df = compute_std(merged_df, grp_cols, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    merged_df = compute_mean(merged_df, grp_cols, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    merged_df = compute_std(merged_df, grp_cols, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    merged_df = compute_mean(merged_df, grp_cols, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    merged_df = compute_std(merged_df, grp_cols, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    merged_df = compute_mean(merged_df, grp_cols, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
    merged_df = compute_mean(merged_df, grp_cols, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
    merged_df = compute_std(merged_df, grp_cols, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')

    merged_df, le_encoded_cols = label_encode(merged_df, None)
    
    merged_df = drop_application_cols(merged_df)
    return merged_df

def drop_application_cols(df_app):
    columns_to_drop = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        columns_to_drop.append('FLAG_DOCUMENT_{}'.format(doc_num))
    
    df_app.drop(columns_to_drop, axis=1, inplace=True)
    return df_app

def categorize_age_group(days_birth):
    age_yrs = -days_birth / 365
    if age_yrs < 27: 
        return 1
    elif age_yrs < 40: 
        return 2
    elif age_yrs < 50: 
        return 3
    elif age_yrs < 65: 
        return 4
    elif age_yrs < 99: 
        return 5
    else: 
        return 0

###############################################################################
#                [Topic: Data Preprocessing - Bureau Pipeline]               #
###############################################################################
def process_bureau_data(path, num_rows=None):
    bureau_df = pd.read_csv(os.path.join(path, 'bureau.csv'), nrows=num_rows)
    
    bureau_df['CREDIT_DURATION'] = -bureau_df['DAYS_CREDIT'] + bureau_df['DAYS_CREDIT_ENDDATE']
    bureau_df['ENDDATE_DIF'] = bureau_df['DAYS_CREDIT_ENDDATE'] - bureau_df['DAYS_ENDDATE_FACT']
    bureau_df['DEBT_PERCENTAGE'] = bureau_df['AMT_CREDIT_SUM'] / bureau_df['AMT_CREDIT_SUM_DEBT']
    bureau_df['DEBT_CREDIT_DIFF'] = bureau_df['AMT_CREDIT_SUM'] - bureau_df['AMT_CREDIT_SUM_DEBT']
    bureau_df['CREDIT_TO_ANNUITY_RATIO'] = bureau_df['AMT_CREDIT_SUM'] / bureau_df['AMT_ANNUITY']

    bureau_df, cat_cols = one_hot_encoder(bureau_df, nan_as_category=False)
    
    bureau_bal = process_bureau_balance(path, num_rows)
    bureau_df = bureau_df.merge(bureau_bal, how='left', on='SK_ID_BUREAU')
    
    bureau_df['STATUS_12345'] = 0
    for i in range(1, 6):
        bureau_df['STATUS_12345'] += bureau_df[f'STATUS_{i}']

    feats_to_agg = ['AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM_OVERDUE','AMT_CREDIT_SUM',
                    'AMT_CREDIT_SUM_DEBT','DEBT_PERCENTAGE','DEBT_CREDIT_DIFF',
                    'STATUS_0','STATUS_12345']
    length_agg = bureau_df.groupby('MONTHS_BALANCE_SIZE')[feats_to_agg].mean().reset_index()
    length_agg.rename({f: 'LL_' + f for f in feats_to_agg}, axis=1, inplace=True)
    bureau_df = bureau_df.merge(length_agg, how='left', on='MONTHS_BALANCE_SIZE')
    del length_agg; gc.collect()

    agg_bureau = group_data(bureau_df, 'BUREAU_', BUREAU_AGG)
    
    active_loans = bureau_df[bureau_df['CREDIT_ACTIVE_Active'] == 1]
    agg_bureau = group_and_merge(active_loans, agg_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
    
    closed_loans = bureau_df[bureau_df['CREDIT_ACTIVE_Closed'] == 1]
    agg_bureau = group_and_merge(closed_loans, agg_bureau, 'BUREAU_CLOSED_', BUREAU_CLOSED_AGG)
    del active_loans, closed_loans; gc.collect()
    
    for ctype in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
        ctype_df = bureau_df[bureau_df[f'CREDIT_TYPE_{ctype}'] == 1]
        prefix_ = 'BUREAU_' + ctype.split(' ')[0].upper() + '_'
        agg_bureau = group_and_merge(ctype_df, agg_bureau, prefix_, BUREAU_LOAN_TYPE_AGG)
        del ctype_df; gc.collect()
    
    for tframe in [6, 12]:
        prefix_ = f'BUREAU_LAST{tframe}M_'
        tframe_df = bureau_df[bureau_df['DAYS_CREDIT'] >= -30 * tframe]
        agg_bureau = group_and_merge(tframe_df, agg_bureau, prefix_, BUREAU_TIME_AGG)
        del tframe_df; gc.collect()

    sort_bureau = bureau_df.sort_values(by=['DAYS_CREDIT'])
    last_loan_overdue = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()
    last_loan_overdue.rename({'AMT_CREDIT_MAX_OVERDUE':'BUREAU_LAST_LOAN_MAX_OVERDUE'},axis=1,inplace=True)
    agg_bureau = agg_bureau.merge(last_loan_overdue, on='SK_ID_CURR', how='left')
    
    agg_bureau['BUREAU_DEBT_OVER_CREDIT'] = agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
    agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
    
    return agg_bureau

def process_bureau_balance(path, num_rows=None):
    bb_df = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows=num_rows)
    bb_df, cat_cols = one_hot_encoder(bb_df, nan_as_category=False)
    
    bb_processed = bb_df.groupby('SK_ID_BUREAU')[cat_cols].mean().reset_index()
    
    agg_dict = {'MONTHS_BALANCE': ['min','max','mean','size']}
    bb_processed = group_and_merge(bb_df, bb_processed, '', agg_dict, 'SK_ID_BUREAU')
    del bb_df; gc.collect()
    
    return bb_processed

###############################################################################
#        [Topic: Data Preprocessing - Previous Applications Pipeline]         #
###############################################################################
def process_previous_applications(path, num_rows=None):
    prev_df = pd.read_csv(os.path.join(path, 'previous_application.csv'), nrows=num_rows)
    pay_df = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows=num_rows)

    ohe_cols = [
        'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
        'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
        'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE'
    ]
    prev_df, cat_columns = one_hot_encoder(prev_df, ohe_cols, nan_as_category=False)

    prev_df['APPLICATION_CREDIT_DIFF'] = prev_df['AMT_APPLICATION'] - prev_df['AMT_CREDIT']
    prev_df['APPLICATION_CREDIT_RATIO'] = prev_df['AMT_APPLICATION'] / prev_df['AMT_CREDIT']
    prev_df['CREDIT_TO_ANNUITY_RATIO'] = prev_df['AMT_CREDIT'] / prev_df['AMT_ANNUITY']
    prev_df['DOWN_PAYMENT_TO_CREDIT'] = prev_df['AMT_DOWN_PAYMENT'] / prev_df['AMT_CREDIT']
    
    total_payment_ = prev_df['AMT_ANNUITY'] * prev_df['CNT_PAYMENT']
    prev_df['SIMPLE_INTERESTS'] = (total_payment_ / prev_df['AMT_CREDIT'] - 1) / prev_df['CNT_PAYMENT']

    approved_df = prev_df[prev_df['NAME_CONTRACT_STATUS_Approved'] == 1]
    active_loans_df = approved_df[approved_df['DAYS_LAST_DUE'] == 365243]
    
    active_pay_df = pay_df[pay_df['SK_ID_PREV'].isin(active_loans_df['SK_ID_PREV'])]
    active_pay_agg = active_pay_df.groupby('SK_ID_PREV')[['AMT_INSTALMENT','AMT_PAYMENT']].sum().reset_index()
    active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']
    
    active_loans_df = active_loans_df.merge(active_pay_agg, on='SK_ID_PREV', how='left')
    active_loans_df['REMAINING_DEBT'] = active_loans_df['AMT_CREDIT'] - active_loans_df['AMT_PAYMENT']
    active_loans_df['REPAYMENT_RATIO'] = active_loans_df['AMT_PAYMENT'] / active_loans_df['AMT_CREDIT']
    
    active_agg_df = group_data(active_loans_df, 'PREV_ACTIVE_', PREVIOUS_ACTIVE_AGG)
    active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM'] / active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']
    del active_pay_df, active_pay_agg, active_loans_df; gc.collect()

    for col_ in ['DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']:
        prev_df[col_].replace(365243, np.nan, inplace=True)

    prev_df['DAYS_LAST_DUE_DIFF'] = prev_df['DAYS_LAST_DUE_1ST_VERSION'] - prev_df['DAYS_LAST_DUE']
    approved_df['DAYS_LAST_DUE_DIFF'] = approved_df['DAYS_LAST_DUE_1ST_VERSION'] - approved_df['DAYS_LAST_DUE']
    
    cat_agg = {key: ['mean'] for key in cat_columns}
    
    agg_prev = group_data(prev_df, 'PREV_', {**PREVIOUS_AGG, **cat_agg})
    
    agg_prev = agg_prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
    del active_agg_df; gc.collect()
    
    agg_prev = group_and_merge(approved_df, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
    refused_df = prev_df[prev_df['NAME_CONTRACT_STATUS_Refused'] == 1]
    agg_prev = group_and_merge(refused_df, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
    del approved_df, refused_df; gc.collect()
    
    for loan_type_ in ['Consumer loans', 'Cash loans']:
        type_subset = prev_df[prev_df[f'NAME_CONTRACT_TYPE_{loan_type_}'] == 1]
        prefix_ = 'PREV_' + loan_type_.split(" ")[0] + '_'
        agg_prev = group_and_merge(type_subset, agg_prev, prefix_, PREVIOUS_LOAN_TYPE_AGG)
        del type_subset; gc.collect()

    pay_df['LATE_PAYMENT'] = pay_df['DAYS_ENTRY_PAYMENT'] - pay_df['DAYS_INSTALMENT']
    pay_df['LATE_PAYMENT'] = pay_df['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
    dpd_ids = pay_df[pay_df['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()
    agg_prev = group_and_merge(prev_df[prev_df['SK_ID_PREV'].isin(dpd_ids)], agg_prev,
                               'PREV_LATE_', PREVIOUS_LATE_PAYMENTS_AGG)
    del dpd_ids; gc.collect()
    
    for tf_ in [12, 24]:
        prefix_ = f'PREV_LAST{tf_}M_'
        tf_df = prev_df[prev_df['DAYS_DECISION'] >= -30 * tf_]
        agg_prev = group_and_merge(tf_df, agg_prev, prefix_, PREVIOUS_TIME_AGG)
        del tf_df; gc.collect()
    
    del prev_df; gc.collect()
    return agg_prev

###############################################################################
#              [Topic: Data Preprocessing - POS-CASH Pipeline]               #
###############################################################################
def process_pos_cash_data(path, num_rows=None):
    pos_df = pd.read_csv(os.path.join(path, 'POS_CASH_balance.csv'), nrows=num_rows)
    pos_df, cat_cols = one_hot_encoder(pos_df, nan_as_category=False)
    
    pos_df['LATE_PAYMENT'] = pos_df['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    
    cat_agg_ = {col: ['mean'] for col in cat_cols}
    pos_agg = group_data(pos_df, 'POS_', {**POS_CASH_AGG, **cat_agg_})
    
    sort_pos_df = pos_df.sort_values(by=['SK_ID_PREV','MONTHS_BALANCE'])
    gp_ = sort_pos_df.groupby('SK_ID_PREV')
    new_df = pd.DataFrame()
    new_df['SK_ID_CURR'] = gp_['SK_ID_CURR'].first()
    new_df['MONTHS_BALANCE_MAX'] = gp_['MONTHS_BALANCE'].max()
    new_df['POS_LOAN_COMPLETED_MEAN'] = gp_['NAME_CONTRACT_STATUS_Completed'].mean()
    new_df['POS_COMPLETED_BEFORE_MEAN'] = gp_['CNT_INSTALMENT'].first() - gp_['CNT_INSTALMENT'].last()
    new_df['POS_COMPLETED_BEFORE_MEAN'] = new_df.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0
                                                       and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)
    new_df['POS_REMAINING_INSTALMENTS'] = gp_['CNT_INSTALMENT_FUTURE'].last()
    new_df['POS_REMAINING_INSTALMENTS_RATIO'] = gp_['CNT_INSTALMENT_FUTURE'].last() / gp_['CNT_INSTALMENT'].last()
    
    df_gp_ = new_df.groupby('SK_ID_CURR').sum().reset_index()
    df_gp_.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace=True)
    pos_agg = pd.merge(pos_agg, df_gp_, on='SK_ID_CURR', how='left')
    del new_df, gp_, df_gp_, sort_pos_df; gc.collect()

    pos_df = compute_sum(pos_df, ['SK_ID_PREV'], 'LATE_PAYMENT', 'LATE_PAYMENT_SUM')
    
    last_month_ids = pos_df.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    sorted_pos_df = pos_df.sort_values(by=['SK_ID_PREV','MONTHS_BALANCE'])
    gp_last = sorted_pos_df.iloc[last_month_ids].groupby('SK_ID_CURR').tail(3)
    gp_mean_ = gp_last.groupby('SK_ID_CURR').mean().reset_index()
    pos_agg = pd.merge(pos_agg, gp_mean_[['SK_ID_CURR','LATE_PAYMENT_SUM']], on='SK_ID_CURR', how='left')

    columns_to_remove = [
        'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
        'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
        'POS_NAME_CONTRACT_STATUS_XNA_MEAN'
    ]
    pos_agg.drop(columns_to_remove, axis=1, inplace=True)
    return pos_agg

###############################################################################
#          [Topic: Data Preprocessing - Installments Pipeline]               #
###############################################################################
def process_installment_payments(path, num_rows=None):
    pay_df = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows=num_rows)
    
    pay_df = compute_sum(pay_df, ['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')
    
    pay_df['PAYMENT_DIFFERENCE'] = pay_df['AMT_INSTALMENT'] - pay_df['AMT_PAYMENT_GROUPED']
    pay_df['PAYMENT_RATIO'] = pay_df['AMT_INSTALMENT'] / pay_df['AMT_PAYMENT_GROUPED']
    pay_df['PAID_OVER_AMOUNT'] = pay_df['AMT_PAYMENT'] - pay_df['AMT_INSTALMENT']
    pay_df['PAID_OVER'] = (pay_df['PAID_OVER_AMOUNT'] > 0).astype(int)
    
    pay_df['DPD'] = pay_df['DAYS_ENTRY_PAYMENT'] - pay_df['DAYS_INSTALMENT']
    pay_df['DPD'] = pay_df['DPD'].apply(lambda x: 0 if x <= 0 else x)
    pay_df['DBD'] = pay_df['DAYS_INSTALMENT'] - pay_df['DAYS_ENTRY_PAYMENT']
    pay_df['DBD'] = pay_df['DBD'].apply(lambda x: 0 if x <= 0 else x)
    
    pay_df['LATE_PAYMENT'] = pay_df['DBD'].apply(lambda x: 1 if x > 0 else 0)
    pay_df['INSTALMENT_PAYMENT_RATIO'] = pay_df['AMT_PAYMENT'] / pay_df['AMT_INSTALMENT']
    pay_df['LATE_PAYMENT_RATIO'] = pay_df.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)
    pay_df['SIGNIFICANT_LATE_PAYMENT'] = pay_df['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
    pay_df['DPD_7'] = pay_df['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    pay_df['DPD_15'] = pay_df['DPD'].apply(lambda x: 1 if x >= 15 else 0)
    
    installments_agg = group_data(pay_df, 'INS_', INSTALLMENTS_AGG)
    
    for m_ in [36, 60]:
        recent_ids = pay_df[pay_df['DAYS_INSTALMENT'] >= -30*m_]['SK_ID_PREV'].unique()
        pay_recent_ = pay_df[pay_df['SK_ID_PREV'].isin(recent_ids)]
        prefix_ = f'INS_{m_}M_'
        installments_agg = group_and_merge(pay_recent_, installments_agg, prefix_, INSTALLMENTS_TIME_AGG)

    group_feats = ['SK_ID_CURR','SK_ID_PREV','DPD','LATE_PAYMENT','PAID_OVER_AMOUNT','PAID_OVER','DAYS_INSTALMENT']
    gp_ = pay_df[group_feats].groupby('SK_ID_CURR')
    func_ = partial(compute_trend_in_installments, periods=INSTALLMENTS_LAST_K_TREND_PERIODS)
    g_ = parallel_apply(gp_, func_, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
    installments_agg = installments_agg.merge(g_, on='SK_ID_CURR', how='left')

    g2_ = parallel_apply(gp_, compute_installments_last_loan, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
    installments_agg = installments_agg.merge(g2_, on='SK_ID_CURR', how='left')
    
    return installments_agg

def compute_trend_in_installments(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    features_ = {}
    for period in periods:
        sub_ = gr_.iloc[:period]
        features_ = add_trend_feature(features_, sub_, 'DPD', f'{period}_TREND_')
        features_ = add_trend_feature(features_, sub_, 'PAID_OVER_AMOUNT', f'{period}_TREND_')
    return features_

def compute_installments_last_loan(gr):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    last_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_id]
    
    feats_ = {}
    feats_ = add_features_in_group(feats_, gr_, 'DPD', ['sum','mean','max','std'], 'LAST_LOAN_')
    feats_ = add_features_in_group(feats_, gr_, 'LATE_PAYMENT', ['count','mean'], 'LAST_LOAN_')
    feats_ = add_features_in_group(feats_, gr_, 'PAID_OVER_AMOUNT', ['sum','mean','max','min','std'], 'LAST_LOAN_')
    feats_ = add_features_in_group(feats_, gr_, 'PAID_OVER', ['count','mean'], 'LAST_LOAN_')
    return feats_

###############################################################################
#            [Topic: Data Preprocessing - Credit Card Pipeline]              #
###############################################################################
def process_credit_card_data(path, num_rows=None):
    cc_df = pd.read_csv(os.path.join(path, 'credit_card_balance.csv'), nrows=num_rows)
    cc_df, cat_cols = one_hot_encoder(cc_df, nan_as_category=False)
    cc_df.rename(columns={'AMT_RECIVABLE':'AMT_RECEIVABLE'}, inplace=True)
    
    cc_df['LIMIT_USE'] = cc_df['AMT_BALANCE'] / cc_df['AMT_CREDIT_LIMIT_ACTUAL']
    cc_df['PAYMENT_DIV_MIN'] = cc_df['AMT_PAYMENT_CURRENT'] / cc_df['AMT_INST_MIN_REGULARITY']
    cc_df['LATE_PAYMENT'] = cc_df['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    cc_df['DRAWING_LIMIT_RATIO'] = cc_df['AMT_DRAWINGS_ATM_CURRENT'] / cc_df['AMT_CREDIT_LIMIT_ACTUAL']
    
    cc_agg = cc_df.groupby('SK_ID_CURR').agg(CREDIT_CARD_AGG)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg.reset_index(inplace=True)
    
    last_ids = cc_df.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    last_month_df = cc_df[cc_df.index.isin(last_ids)]
    cc_agg = group_and_merge(last_month_df, cc_agg, 'CC_LAST_', {'AMT_BALANCE':['mean','max']})

    for months_ in [12, 24, 48]:
        cc_prev_id = cc_df[cc_df['MONTHS_BALANCE'] >= -months_]['SK_ID_PREV'].unique()
        cc_recent = cc_df[cc_df['SK_ID_PREV'].isin(cc_prev_id)]
        prefix_ = f'INS_{months_}M_'
        cc_agg = group_and_merge(cc_recent, cc_agg, prefix_, CREDIT_CARD_TIME_AGG)
    
    return cc_agg

###############################################################################
#            [Topic: Utility Functions for Aggregation & Encoding]           #
###############################################################################
@contextmanager
def time_tracker(name):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))

def group_data(df_to_agg, prefix, aggregations, aggregate_by='SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()

def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by='SK_ID_CURR'):
    agg_df = group_data(df_to_agg, prefix, aggregations, aggregate_by=aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on=aggregate_by)

def compute_mean(df_, group_cols, counted, agg_name):
    gp_ = df_[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted: agg_name})
    df_ = df_.merge(gp_, on=group_cols, how='left')
    del gp_
    gc.collect()
    return df_

def compute_median(df_, group_cols, counted, agg_name):
    gp_ = df_[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(columns={counted: agg_name})
    df_ = df_.merge(gp_, on=group_cols, how='left')
    del gp_
    gc.collect()
    return df_

def compute_std(df_, group_cols, counted, agg_name):
    gp_ = df_[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(columns={counted: agg_name})
    df_ = df_.merge(gp_, on=group_cols, how='left')
    del gp_
    gc.collect()
    return df_

def compute_sum(df_, group_cols, counted, agg_name):
    gp_ = df_[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(columns={counted: agg_name})
    df_ = df_.merge(gp_, on=group_cols, how='left')
    del gp_
    gc.collect()
    return df_

def one_hot_encoder(df_, categorical_columns=None, nan_as_category=True):
    original_cols = list(df_.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df_.columns if df_[col].dtype == 'object']
    df_ = pd.get_dummies(df_, columns=categorical_columns, dummy_na=nan_as_category)
    new_cols = [c for c in df_.columns if c not in original_cols]
    return df_, new_cols

def label_encode(df_, categorical_columns=None):
    if not categorical_columns:
        categorical_columns = [col for col in df_.columns if df_[col].dtype == 'object']
    for col in categorical_columns:
        df_[col], _ = pd.factorize(df_[col])
    return df_, categorical_columns

def add_features_in_group(features_dict, group_data_, feature_name, agg_list, prefix):
    for agg in agg_list:
        if agg == 'sum':
            features_dict[f'{prefix}{feature_name}_sum'] = group_data_[feature_name].sum()
        elif agg == 'mean':
            features_dict[f'{prefix}{feature_name}_mean'] = group_data_[feature_name].mean()
        elif agg == 'max':
            features_dict[f'{prefix}{feature_name}_max'] = group_data_[feature_name].max()
        elif agg == 'min':
            features_dict[f'{prefix}{feature_name}_min'] = group_data_[feature_name].min()
        elif agg == 'std':
            features_dict[f'{prefix}{feature_name}_std'] = group_data_[feature_name].std()
        elif agg == 'count':
            features_dict[f'{prefix}{feature_name}_count'] = group_data_[feature_name].count()
        elif agg == 'skew':
            features_dict[f'{prefix}{feature_name}_skew'] = skew(group_data_[feature_name])
        elif agg == 'kurt':
            features_dict[f'{prefix}{feature_name}_kurt'] = kurtosis(group_data_[feature_name])
        elif agg == 'iqr':
            features_dict[f'{prefix}{feature_name}_iqr'] = iqr(group_data_[feature_name])
        elif agg == 'median':
            features_dict[f'{prefix}{feature_name}_median'] = group_data_[feature_name].median()
    return features_dict

def add_trend_feature(features_dict, grp_, feature_name, prefix):
    y_vals = grp_[feature_name].values
    try:
        x_vals = np.arange(0, len(y_vals)).reshape(-1, 1)
        lr_model = LinearRegression()
        lr_model.fit(x_vals, y_vals)
        trend_slope = lr_model.coef_[0]
    except:
        trend_slope = np.nan
    features_dict[f'{prefix}{feature_name}'] = trend_slope
    return features_dict

###############################################################################
#            [Topic: Parallelization Helpers for Large GroupBy]              #
###############################################################################
def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    if num_workers <= 0:
        num_workers = NUM_THREADS
    indices, feats = [], []
    for idx_chunk, group_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            feats_chunk = executor.map(func, group_chunk)
        feats.extend(feats_chunk)
        indices.extend(idx_chunk)

    feats_df = pd.DataFrame(feats)
    feats_df.index = indices
    feats_df.index.name = index_name
    return feats_df

def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    grp_chunk, idx_chunk = [], []
    for i, (idx, df_) in enumerate(groupby_object):
        grp_chunk.append(df_)
        idx_chunk.append(idx)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            yield idx_chunk.copy(), grp_chunk.copy()
            grp_chunk, idx_chunk = [], []

###############################################################################
#              [Topic: Memory Optimization & Configuration]                   #
###############################################################################
def reduce_memory_usage(df_):
    start_mem = df_.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df_.columns)))

    for col in df_.columns:
        col_type = df_[col].dtypes
        if col_type != object:
            cmin = df_[col].min()
            cmax = df_[col].max()
            if str(col_type)[:3] == 'int':
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df_[col] = df_[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df_[col] = df_[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df_[col] = df_[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df_[col] = df_[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df_[col] = df_[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df_[col] = df_[col].astype(np.float32)
                else:
                    df_[col] = df_[col].astype(np.float64)
    end_mem = df_.memory_usage().sum() / 1024 ** 2
    mem_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, mem_reduction))
    return df_

###############################################################################
#                        [Topic: Global Configurations]                       #
###############################################################################
NUM_THREADS = 4
DATA_DIR = "../input/home-credit-default-risk/"
SUBMISSION_SUFIX = "_model2_04"
INSTALLMENTS_LAST_K_TREND_PERIODS = [12, 24, 60, 120]
GENERATE_SUBMISSION_FILES = True
STRATIFIED_KFOLD = False
RANDOM_SEED = 737851
NUM_FOLDS = 10
EARLY_STOPPING = 100

# Modify LIGHTGBM_PARAMS for GPU training:
LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',       # Changed from 'goss' to 'gbdt' for GPU support
    'device_type': 'gpu',          # Enable GPU usage
    'n_estimators': 10000,
    'learning_rate': 0.005134,
    'num_leaves': 54,
    'max_depth': 10,
    'subsample_for_bin': 240000,
    'reg_alpha': 0.436193,
    'reg_lambda': 0.479169,
    'colsample_bytree': 0.508716,
    'min_split_gain': 0.024766,
    'subsample': 1,
    'is_unbalance': False,
    'silent': -1,
    'verbose': -1
}

# Aggregation dictionaries (please adjust as needed)
BUREAU_AGG = {
    'SK_ID_BUREAU': ['nunique'],
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['sum']  # Added for ratio calculation
}
BUREAU_ACTIVE_AGG = {
    'DAYS_CREDIT': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['sum'],           # Add this aggregation
    'AMT_CREDIT_SUM_DEBT': ['sum']
}
BUREAU_CLOSED_AGG = {'DAYS_CREDIT': ['max']}
BUREAU_LOAN_TYPE_AGG = {'DAYS_CREDIT': ['mean', 'max']}
BUREAU_TIME_AGG = {'AMT_CREDIT_SUM': ['max', 'sum']}

PREVIOUS_AGG = {'SK_ID_PREV': ['nunique'], 'AMT_APPLICATION': ['max', 'mean']}
PREVIOUS_APPROVED_AGG = {
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'AMT_ANNUITY': ['max', 'mean']  # This will create APPROVED_AMT_ANNUITY_MAX and APPROVED_AMT_ANNUITY_MEAN.
}
PREVIOUS_APPROVED_AGG = {'AMT_CREDIT': ['min', 'max', 'mean']}
PREVIOUS_REFUSED_AGG = {'AMT_APPLICATION': ['max', 'mean']}
PREVIOUS_LATE_PAYMENTS_AGG = {'DAYS_LAST_DUE': ['min', 'max', 'mean']}
PREVIOUS_LOAN_TYPE_AGG = {'AMT_CREDIT': ['sum']}
PREVIOUS_TIME_AGG = {'DAYS_DECISION': ['min', 'mean']}

POS_CASH_AGG = {'MONTHS_BALANCE': ['min', 'max', 'size']}
INSTALLMENTS_AGG = {'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']}
INSTALLMENTS_TIME_AGG = {'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum']}
CREDIT_CARD_AGG = {'AMT_BALANCE': ['max', 'mean']}
CREDIT_CARD_TIME_AGG = {'AMT_BALANCE': ['mean', 'max']}

###############################################################################
#                        [Topic: Run the Entire Pipeline]                     #
###############################################################################
if __name__ == '__main__':
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns', 100)
    with time_tracker('Pipeline total time'):
        execute_main_pipeline(debug_mode=False)





