device = 'cpu' # 'cpu' or 'gpu'

if device=='gpu':
    %load_ext cudf.pandas


import numpy as np, pandas as pd, gc
from typing import Optional, List
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')
orig['y'] = orig['y'].map({'yes': 1, 'no': 0})


NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fe_func,
    lgbm_params: dict,
    categorical_cols: Optional[List[str]] = None,
    orig: Optional[pd.DataFrame] = None,
    target_col: str = 'y',
    n_splits: int = 5,
    prediction_method: str = 'fold_average'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trains a LightGBM model using cross-validation with robust categorical feature handling.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
        fe_func (function): The function for feature engineering.
        lgbm_params (dict): Hyperparameters for the LightGBM model.
        categorical_cols (Optional[List[str]]): List of column names to be treated as categorical.
        orig (Optional[pd.DataFrame]): Optional external data to add to the training set.
        target_col (str): The name of the target column.
        n_splits (int): The number of folds for cross-validation.
        prediction_method (str): Method for test prediction ('fold_average' or 'refit').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - oof_df (pd.DataFrame): DataFrame with out-of-fold predictions.
            - test_pred_df (pd.DataFrame): DataFrame with test predictions.
    """
    if prediction_method not in ['fold_average', 'refit']:
        raise ValueError("prediction_method must be either 'fold_average' or 'refit'.")

    # --- 1. Combine all dataframes for consistent processing ---
    train_ids = train['id']
    test_ids = test['id']
    
    train['data_source'] = 'train'
    test['data_source'] = 'test'
    dfs_to_process = [train, test]
    if orig is not None:
        orig['data_source'] = 'orig'
        # Ensure orig has an 'id' column for consistency if it's missing
        if 'id' not in orig.columns:
            orig = orig.reset_index().rename(columns={'index': 'id'})
        dfs_to_process.append(orig)

    combined_df = pd.concat(dfs_to_process, ignore_index=True)

    # --- 2. Apply feature engineering and categorical conversion ---
    print("Starting unified feature engineering...")
    combined_processed = fe_func(combined_df)
    
    if categorical_cols:
        print(f"Converting {len(categorical_cols)} columns to 'category' type...")
        for col in categorical_cols:
            if col in combined_processed.columns:
                combined_processed[col] = combined_processed[col].astype('category')

    # --- 3. Split back into train, test, and optional orig sets ---
    train_processed = combined_processed[combined_processed['data_source'] == 'train'].drop('data_source', axis=1)
    test_processed = combined_processed[combined_processed['data_source'] == 'test'].drop('data_source', axis=1)
    orig_processed = None
    if orig is not None:
        orig_processed = combined_processed[combined_processed['data_source'] == 'orig'].drop('data_source', axis=1)

    # --- 4. Define features robustly from the training set ---
    features = [col for col in train_processed.columns if col not in ['id', target_col]]
    
    # Identify which of the final features are categorical for LGBM
    lgbm_categorical_features = [col for col in categorical_cols if col in features] if categorical_cols else []
    
    # --- 5. Cross-validation loop ---
    oof_preds = np.zeros(len(train_processed))
    test_preds = np.zeros(len(test_processed))
    feature_importance_df = pd.DataFrame(index=features)
    best_iterations = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, train_processed[target_col])):
        print(f"\n===== Fold {fold + 1} / {n_splits} =====")
        
        X_train_fold = train_processed.iloc[train_idx][features]
        y_train_fold = train_processed.iloc[train_idx][target_col]
        X_val_fold = train_processed.iloc[val_idx][features]
        y_val_fold = train_processed.iloc[val_idx][target_col]

        X_train_full, y_train_full = X_train_fold, y_train_fold
        if orig_processed is not None:
            print("Merging external data...")
            X_train_full = pd.concat([X_train_fold, orig_processed[features]], ignore_index=True)
            y_train_full = pd.concat([y_train_fold, orig_processed[target_col]], ignore_index=True)
        
        model = lgb.LGBMClassifier(**lgbm_params)
        print("Training model...")
        model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=lgbm_categorical_features
        )

        oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        if prediction_method == 'fold_average':
            test_preds += model.predict_proba(test_processed[features])[:, 1] / skf.get_n_splits()

        best_iterations.append(model.best_iteration_)
        feature_importance_df[f'fold_{fold+1}'] = model.feature_importances_
        
        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f}")
        print(f"Fold {fold + 1} Best Iteration: {model.best_iteration_}")

        del X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_train_full, y_train_full, model
        gc.collect()

    overall_auc = roc_auc_score(train_processed[target_col], oof_preds)
    print(f"\n>>> CV Overall AUC: {overall_auc:.6f}")

    # --- 6. Final predictions and feature importance ---
    if prediction_method == 'refit':
        print("\nStarting refit process...")
        refit_iterations = int(np.mean(best_iterations))
        print(f"Refitting on all data with n_estimators = {refit_iterations}")

        X_train_all, y_train_all = train_processed[features], train_processed[target_col]
        if orig_processed is not None:
             X_train_all = pd.concat([train_processed[features], orig_processed[features]], ignore_index=True)
             y_train_all = pd.concat([train_processed[target_col], orig_processed[target_col]], ignore_index=True)
        
        final_params = lgbm_params.copy()
        final_params['n_estimators'] = refit_iterations
        model_refit = lgb.LGBMClassifier(**final_params)
        model_refit.fit(
            X_train_all, y_train_all,
            categorical_feature=lgbm_categorical_features
        )

        test_preds = model_refit.predict_proba(test_processed[features])[:, 1]
        feature_importance_df['refit'] = model_refit.feature_importances_
        print("Refit and prediction complete.")

    oof_df = pd.DataFrame({'id': train_ids, 'y': oof_preds})
    test_pred_df = pd.DataFrame({'id': test_ids, 'y': test_preds})

    file_prefix = fe_func.__name__
    oof_filename = f"oof_{file_prefix}.csv"
    submission_filename = f"test_{file_prefix}.csv"
    oof_df.to_csv(oof_filename, index=False)
    test_pred_df.to_csv(submission_filename, index=False)
    print(f"\nSuccessfully saved OOF predictions to: {oof_filename}")
    print(f"Successfully saved test predictions to: {submission_filename}")

    importance_source = 'average'
    plot_title = "Feature Importance (Average over Folds)"
    if prediction_method == 'refit':
        importance_source = 'refit'
        plot_title = "Feature Importance (from Refit Model)"
    else: # fold_average
        feature_importance_df['average'] = feature_importance_df.mean(axis=1)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=feature_importance_df.sort_values(by=importance_source, ascending=False).head(20)[importance_source],
        y=feature_importance_df.sort_values(by=importance_source, ascending=False).head(20).index
    )
    plt.title(plot_title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    
    return oof_df, test_pred_df


base_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.8414060376361174,
    'n_estimators': 32767,
    'learning_rate': 0.015620157278478881,
    'max_bin': 1023,
    'min_child_samples': 4,
    'n_estimators': 32767,
    'n_jobs': -1,
    'num_leaves': 28,
    'reg_alpha': 7.995675207850095,
    'reg_lambda': 0.004127757701291098,
    'verbose': -1,
    'device': device,
}

def base(df):
    return df

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=base,
    lgbm_params=base_params,
    categorical_cols=CATS,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


allcats_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.39799914227070404,
    'n_estimators': 30000,
    'learning_rate': 0.029111846203466612,
    'max_bin': 1023,
    'min_child_samples': 19,
    'n_jobs': -1,
    'num_leaves': 60,
    'reg_alpha': 0.005514571831332959,
    'reg_lambda': 0.006279126463157728,
    'verbose': -1,
    'device': device,
}

def allcats(df):
    return df

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=allcats,
    lgbm_params=allcats_params,
    categorical_cols=NUMS+CATS,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


def digit_fe(df):
    
    df = df.copy()

    df['balance_is_negative'] = (df['balance'] < 0).astype(int)
    df['pdays_not_contacted'] = (df['pdays'] == -1).astype(int)

    df['age_digit_1s'] = df['age'] % 10
    df['age_digit_10s'] = (df['age'] // 10) % 10

    df['balance_abs'] = df['balance'].abs()
    df['balance_digit_1s'] = df['balance_abs'] % 10
    df['balance_digit_10s'] = (df['balance_abs'] // 10) % 10
    df['balance_digit_100s'] = (df['balance_abs'] // 100) % 10
    df['balance_digit_1000s'] = (df['balance_abs'] // 1000) % 10
    df['balance_digit_10000s'] = (df['balance_abs'] // 10000) % 10
    df = df.drop(columns=['balance_abs']) 

    df['day_digit_1s'] = df['day'] % 10
    df['day_digit_10s'] = (df['day'] // 10) % 10

    df['duration_digit_1s'] = df['duration'] % 10
    df['duration_digit_10s'] = (df['duration'] // 10) % 10
    df['duration_digit_100s'] = (df['duration'] // 100) % 10
    df['duration_digit_1000s'] = (df['duration'] // 1000) % 10

    df['campaign_digit_1s'] = df['campaign'] % 10
    df['campaign_digit_10s'] = (df['campaign'] // 10) % 10

    df['pdays_abs'] = df['pdays'].abs()
    df['pdays_digit_1s'] = df['pdays_abs'] % 10
    df['pdays_digit_10s'] = (df['pdays_abs'] // 10) % 10
    df['pdays_digit_100s'] = (df['pdays_abs'] // 100) % 10
    df = df.drop(columns=['pdays_abs']) 

    df['previous_digit_1s'] = df['previous'] % 10
    df['previous_digit_10s'] = (df['previous'] // 10) % 10
    df['previous_digit_100s'] = (df['previous'] // 100) % 10
    
    return df

digits = ['balance_is_negative','pdays_not_contacted','age_digit_1s','age_digit_10s','balance_digit_1s','balance_digit_10s','balance_digit_100s','balance_digit_1000s','balance_digit_10000s','day_digit_1s','day_digit_10s','duration_digit_1s','duration_digit_10s','duration_digit_100s','duration_digit_1000s','campaign_digit_1s','campaign_digit_10s','pdays_digit_1s','pdays_digit_10s','pdays_digit_100s','previous_digit_1s','previous_digit_10s','previous_digit_100s']


digit_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.8466993787926143,
    'n_estimators': 32767,
    'learning_rate': 0.01842227851409286,
    'max_bin': 1023,
    'min_child_samples': 4,
    'n_estimators': 15000,
    'n_jobs': -1,
    'num_leaves': 146,
    'reg_alpha': 0.26399329896096496,
    'reg_lambda': 0.029855733289437917,
    'verbose': -1,
    'device': device,
}

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=digit_fe,
    lgbm_params=digit_params,
    categorical_cols=CATS+digits,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


def arithmetic_fe(df) -> pd.DataFrame:
    df = df.copy()
    epsilon = 1e-6
    for col in NUMS:
        df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min())

        if df[col].min() >= 0:
            df[f'{col}_sqrt'] = np.sqrt(df[col])

    df['balance_per_age'] = df['balance'] / (df['age'] + epsilon)    
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + epsilon)    
    contacted_mask = df['pdays'] != -1
    df['avg_days_between_contacts'] = np.nan
    df.loc[contacted_mask, 'avg_days_between_contacts'] = df.loc[contacted_mask, 'pdays'] / (df.loc[contacted_mask, 'previous'] + epsilon)
    
    df['campaign_to_previous_ratio'] = df['campaign'] / (df['previous'] + 1)
    
    for col1, col2 in combinations(NUMS, 2):
        df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + epsilon)
        df[f'{col2}_div_{col1}'] = df[col2] / (df[col1] + epsilon)
        
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / df['day'].max())
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / df['day'].max())
    
    month_to_num_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_to_num_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / df['month_num'].max())
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / df['month_num'].max())
    df.drop('month_num', axis=1, inplace=True)

    return df


arithmetic_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.7153704456124521,
    'learning_rate': 0.0322090954919683,
    'max_bin': 1023,
    'min_child_samples': 10,
    'n_estimators': 30000,
    'n_jobs': -1,
    'num_leaves': 98,
    'reg_alpha': 0.16690885603796932,
    'reg_lambda': 92.81700080837966,
    'verbose': -1,
    'device': device,
}

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=arithmetic_fe,
    lgbm_params=arithmetic_params,
    categorical_cols=CATS,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


def binning_fe(df):
    df['age_binned'] = pd.cut(df['age'], bins=[17, 29, 39, 49, 59, 96], labels=False, right=True)
    df['balance_binned'] = pd.cut(df['balance'], bins=[-8020, -1, 0, 634, 1390, 99718], labels=False, right=True)
    df['day_binned'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=False, right=True)
    df['duration_binned'] = pd.cut(df['duration'], bins=[0, 91, 133, 361, 4919], labels=False, right=True)
    df['campaign_binned'] = pd.cut(df['campaign'], bins=[0, 1, 2, 3, 64], labels=False, right=True)
    df['pdays_binned'] = pd.cut(df['pdays'], bins=[-2, -1, 90, 180, 872], labels=False, right=True)
    df['previous_binned'] = pd.cut(df['previous'], bins=[-1, 0, 1, 5, 201], labels=False, right=True)
    
    return df

binned_features = [f'{col}_binned' for col in NUMS]


binned_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.3720426652581872,
    'n_estimators': 30000,
    'learning_rate': 0.005828593792696136,
    'max_bin': 1023,
    'min_child_samples': 7,
    'n_jobs': -1,
    'num_leaves': 443,
    'reg_alpha': 0.0019450865311753044,
    'reg_lambda': 0.07280263155386243,
    'verbose': -1,
    'device': device,
}

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=binning_fe,
    lgbm_params=binned_params,
    categorical_cols=CATS+binned_features,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


from tqdm import tqdm

target_col = 'y'
base_features = NUMS + CATS
combination_rank = 2

def orig_TE(df):
    df = df.copy()

    global_mean = orig[target_col].mean()
        
    print("Generating features from single columns...")
    for col in tqdm(base_features):
        new_col_name_mean = f'orig_mean_y_on_{col}'
        mean_map = orig.groupby(col, observed=True)[target_col].mean()
        df[new_col_name_mean] = df[col].map(mean_map).fillna(global_mean).astype('float32')
        
        new_col_name_freq = f'orig_freq_on_{col}'
        freq_map = orig[col].value_counts(normalize=True)
        df[new_col_name_freq] = df[col].map(freq_map).fillna(0).astype('float32')

    print(f"Generating features from {combination_rank}-way combination columns...")
    temp_combo_cols_to_drop = []
    
    for cols in tqdm(list(combinations(base_features, combination_rank))):
        interaction_col_name = '-'.join(map(str, cols))
        temp_combo_cols_to_drop.append(interaction_col_name)

        orig[interaction_col_name] = orig[cols[0]].astype(str) + '_' + orig[cols[1]].astype(str)
        df[interaction_col_name] = df[cols[0]].astype(str) + '_' + df[cols[1]].astype(str)
        
        new_col_name_mean = f'orig_mean_y_on_{interaction_col_name}'
        mean_map = orig.groupby(interaction_col_name, observed=True)[target_col].mean()
        df[new_col_name_mean] = df[interaction_col_name].map(mean_map).fillna(global_mean).astype('float32')

        new_col_name_freq = f'orig_freq_on_{interaction_col_name}'
        freq_map = orig[interaction_col_name].value_counts(normalize=True)
        df[new_col_name_freq] = df[interaction_col_name].map(freq_map).fillna(0).astype('float32')

    df = df.drop(columns=temp_combo_cols_to_drop)
    gc.collect()

    return df


orig_TE_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.631656706473612,
    'n_estimators': 30000,
    'learning_rate': 0.00430380500299762,
    'max_bin': 1023,
    'min_child_samples': 7,
    'n_jobs': -1,
    'num_leaves': 349,
    'reg_alpha': 0.17479551596071996,
    'reg_lambda': 0.5529684837915299,
    'verbose': -1,
    'device': device,
}

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=orig_TE,
    lgbm_params=orig_TE_params,
    categorical_cols=CATS,
    # orig=orig,
    target_col='y',
    prediction_method='fold_average'
)


orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')
orig['y'] = orig['y'].map({'yes': 1, 'no': 0})


def group_interaction(df):
    df = df.copy()
    
    df['age_group'] = pd.cut(df['age'], bins=[17, 33, 39, 48, 96], labels=['young', 'adult', 'middle-aged', 'senior'])
    df['balance_group'] = pd.cut(df['balance'], bins=[-8020, 0, 1390, 100000], labels=['neg_zero', 'low_pos', 'high_pos'])
    df['duration_group'] = pd.cut(df['duration'], bins=[-1, 91, 361, 5000], labels=['short', 'medium', 'long'])
    df['campaign_group'] = pd.cut(df['campaign'], bins=[-1, 1, 3, 64], labels=['once', 'few', 'many'])
    df['pdays_group'] = pd.cut(df['pdays'], bins=[-2, 0, 180, 872], labels=['not_contacted', 'recent', 'old'])

    # --- 2. Categorical Interaction Features ---
    df['age_balance_group'] = df['age_group'].astype(str) + '_' + df['balance_group'].astype(str)
    df['age_duration_group'] = df['age_group'].astype(str) + '_' + df['duration_group'].astype(str)
    df['job_marital'] = df['job'].astype(str) + '_' + df['marital'].astype(str)
    df['poutcome_age_group'] = df['poutcome'].astype(str) + '_' + df['age_group'].astype(str)
    df['education_loan'] = df['education'].astype(str) + '_' + df['loan'].astype(str)

    # --- 3. Groupby Aggregation Features
    group_features = ['job', 'marital', 'education', 'poutcome', 'age_group', 'balance_group']
    numeric_features = ['balance', 'duration', 'campaign', 'age']

    for group_col in group_features:
        for num_col in numeric_features:
            if 'age' in num_col and 'age' in group_col: continue
            if 'balance' in num_col and 'balance' in group_col: continue

            df[f'{num_col}_mean_by_{group_col}'] = df.groupby(group_col)[num_col].transform('mean')
            df[f'{num_col}_std_by_{group_col}'] = df.groupby(group_col)[num_col].transform('std')
            df[f'{num_col}_max_by_{group_col}'] = df.groupby(group_col)[num_col].transform('max')

    # --- 4. Numerical Transformations and Ratios ---
    df['balance_x_housing'] = df['balance'] * (df['housing'] == 'yes').astype(int)
    df['campaign_x_previous'] = df['campaign'] * (df['pdays'] > -1).astype(int)
    df['log_duration'] = np.log1p(df['duration'])
    df['log_balance'] = np.log1p(df['balance'] - df['balance'].min())
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1e-6)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1e-6)

    return df

group_inter = ['age_group', 'balance_group', 'duration_group', 'campaign_group', 'pdays_group', 'age_balance_group', 'age_duration_group', 'job_marital', 'poutcome_age_group', 'education_loan', 'balance_mean_by_job', 'balance_std_by_job', 'balance_max_by_job', 'duration_mean_by_job', 'duration_std_by_job', 'duration_max_by_job', 'campaign_mean_by_job', 'campaign_std_by_job', 'campaign_max_by_job', 'age_mean_by_job', 'age_std_by_job', 'age_max_by_job', 'balance_mean_by_marital', 'balance_std_by_marital', 'balance_max_by_marital', 'duration_mean_by_marital', 'duration_std_by_marital', 'duration_max_by_marital', 'campaign_mean_by_marital', 'campaign_std_by_marital', 'campaign_max_by_marital', 'age_mean_by_marital', 'age_std_by_marital', 'age_max_by_marital', 'balance_mean_by_education', 'balance_std_by_education', 'balance_max_by_education', 'duration_mean_by_education', 'duration_std_by_education', 'duration_max_by_education', 'campaign_mean_by_education', 'campaign_std_by_education', 'campaign_max_by_education', 'age_mean_by_education', 'age_std_by_education', 'age_max_by_education', 'balance_mean_by_poutcome', 'balance_std_by_poutcome', 'balance_max_by_poutcome', 'duration_mean_by_poutcome', 'duration_std_by_poutcome', 'duration_max_by_poutcome', 'campaign_mean_by_poutcome', 'campaign_std_by_poutcome', 'campaign_max_by_poutcome', 'age_mean_by_poutcome', 'age_std_by_poutcome', 'age_max_by_poutcome', 'balance_mean_by_age_group', 'balance_std_by_age_group', 'balance_max_by_age_group', 'duration_mean_by_age_group', 'duration_std_by_age_group', 'duration_max_by_age_group', 'campaign_mean_by_age_group', 'campaign_std_by_age_group', 'campaign_max_by_age_group', 'duration_mean_by_balance_group', 'duration_std_by_balance_group', 'duration_max_by_balance_group', 'campaign_mean_by_balance_group', 'campaign_std_by_balance_group', 'campaign_max_by_balance_group', 'age_mean_by_balance_group', 'age_std_by_balance_group', 'age_max_by_balance_group', 'balance_x_housing', 'campaign_x_previous', 'log_duration', 'log_balance', 'duration_per_campaign', 'balance_per_age']


group_interaction_params = {
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.592607969651623,
    'n_estimators': 32767,
    'learning_rate': 0.013124618614290944,
    'max_bin': 1023,
    'min_child_samples': 9,
    'n_estimators': 30000,
    'n_jobs': -1,
    'num_leaves': 56,
    'reg_alpha': 0.001975258376030875,
    'reg_lambda': 14.420859949293076,
    'verbose': -1,
    'device': device,
}

oof_df, test_pred_df = train_lgbm(
    train=train,
    test=test,
    fe_func=group_interaction,
    lgbm_params=group_interaction_params,
    categorical_cols=CATS+group_inter,
    orig=orig,
    target_col='y',
    prediction_method='fold_average'
)

