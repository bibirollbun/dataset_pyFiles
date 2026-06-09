# Home Credit Default Risk Solution
# Simplified version to ensure the code runs smoothly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import os
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    """Timer, used to record the execution time of each step"""
    t0 = time.time()
    yield
    print("{} - Completed in: {:.0f} seconds".format(title, time.time() - t0))

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# 1. Read data
print("Reading datasets...")
with timer("Read data"):
    app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
    app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
    print('Number of training samples:', len(app_train))
    print('Number of test samples:', len(app_test))

# 2. Data preprocessing
with timer("Data preprocessing"):
    # Check target variable distribution
    print('Target variable distribution:')
    print(app_train['TARGET'].value_counts())
    print('Positive sample ratio: {:.2%}'.format(app_train['TARGET'].mean()))
    
    # Handle outliers
    app_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    app_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Handle categorical variables
    categorical_features = [col for col in app_train.columns if app_train[col].dtype == 'object']
    
    print('Number of categorical features:', len(categorical_features))
    
    # Label encoding
    for col in categorical_features:
        le = LabelEncoder()
        le.fit(list(app_train[col].astype(str).values) + list(app_test[col].astype(str).values))
        app_train[col] = le.transform(list(app_train[col].astype(str).values))
        app_test[col] = le.transform(list(app_test[col].astype(str).values))

# 3. Create basic features
with timer("Create basic features"):
    # Create basic ratio features
    app_train['CREDIT_INCOME_RATIO'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
    app_train['ANNUITY_INCOME_RATIO'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['CREDIT_TERM'] = app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY']
    app_train['DAYS_EMPLOYED_RATIO'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
    
    app_test['CREDIT_INCOME_RATIO'] = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']
    app_test['ANNUITY_INCOME_RATIO'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']
    app_test['CREDIT_TERM'] = app_test['AMT_CREDIT'] / app_test['AMT_ANNUITY']
    app_test['DAYS_EMPLOYED_RATIO'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']

# 4. Process Bureau data
with timer("Process Bureau data"):
    bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
    bb = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
    
    # Count the number of loans for each customer
    bureau_counts = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
    bureau_counts.rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'}, inplace=True)
    
    # Process the average loan amount for each customer
    bureau_avg_loan = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean().reset_index()
    bureau_avg_loan.rename(columns={'AMT_CREDIT_SUM': 'BUREAU_AVG_LOAN'}, inplace=True)
    
    # Process overdue day statistics in bureau data
    bureau_overdue = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max().reset_index()
    bureau_overdue.rename(columns={'CREDIT_DAY_OVERDUE': 'BUREAU_MAX_OVERDUE'}, inplace=True)
    
    # Merge features
    app_train = app_train.merge(bureau_counts, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(bureau_avg_loan, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(bureau_overdue, on='SK_ID_CURR', how='left')
    
    app_test = app_test.merge(bureau_counts, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(bureau_avg_loan, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(bureau_overdue, on='SK_ID_CURR', how='left')
    
    del bureau, bb, bureau_counts, bureau_avg_loan, bureau_overdue
    gc.collect()

# 5. Process previous_application data
with timer("Process Previous_application data"):
    prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
    
    # Count the number of applications for each customer
    prev_app_counts = prev.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index()
    prev_app_counts.rename(columns={'SK_ID_PREV': 'PREV_APP_COUNT'}, inplace=True)
    
    # Calculate the average loan amount for each customer
    prev_app_amt = prev.groupby('SK_ID_CURR')['AMT_CREDIT'].mean().reset_index()
    prev_app_amt.rename(columns={'AMT_CREDIT': 'PREV_APP_AVG_AMOUNT'}, inplace=True)
    
    # Calculate the rejection ratio of applications
    prev_app_rejected = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].apply(
        lambda x: sum(x == 'Refused') / len(x)
    ).reset_index()
    prev_app_rejected.rename(columns={'NAME_CONTRACT_STATUS': 'PREV_APP_REJECTION_RATIO'}, inplace=True)
    
    # Merge features
    app_train = app_train.merge(prev_app_counts, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(prev_app_amt, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(prev_app_rejected, on='SK_ID_CURR', how='left')
    
    app_test = app_test.merge(prev_app_counts, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(prev_app_amt, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(prev_app_rejected, on='SK_ID_CURR', how='left')
    
    del prev, prev_app_counts, prev_app_amt, prev_app_rejected
    gc.collect()

# 6. Process installments_payments data
with timer("Process Installments_payments data"):
    ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
    
    # Calculate average days late
    ins['DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_LATE'].apply(lambda x: max(0, x))
    
    # Calculate average days late
    avg_late_days = ins.groupby('SK_ID_CURR')['DAYS_LATE'].mean().reset_index()
    avg_late_days.rename(columns={'DAYS_LATE': 'AVG_LATE_DAYS'}, inplace=True)
    
    # Calculate payment ratio
    ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    avg_payment_ratio = ins.groupby('SK_ID_CURR')['PAYMENT_RATIO'].mean().reset_index()
    avg_payment_ratio.rename(columns={'PAYMENT_RATIO': 'AVG_PAYMENT_RATIO'}, inplace=True)
    
    # Merge features
    app_train = app_train.merge(avg_late_days, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(avg_payment_ratio, on='SK_ID_CURR', how='left')
    
    app_test = app_test.merge(avg_late_days, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(avg_payment_ratio, on='SK_ID_CURR', how='left')
    
    del ins, avg_late_days, avg_payment_ratio
    gc.collect()

# 7. Process POS_CASH_balance data
with timer("Process POS_CASH_balance data"):
    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
    
    # Calculate average DPD
    avg_pos_dpd = pos.groupby('SK_ID_CURR')['SK_DPD'].mean().reset_index()
    avg_pos_dpd.rename(columns={'SK_DPD': 'AVG_POS_DPD'}, inplace=True)
    
    # Calculate max DPD
    max_pos_dpd = pos.groupby('SK_ID_CURR')['SK_DPD'].max().reset_index()
    max_pos_dpd.rename(columns={'SK_DPD': 'MAX_POS_DPD'}, inplace=True)
    
    # Calculate the number of POS transactions for each customer
    pos_counts = pos.groupby('SK_ID_CURR').size().reset_index()
    pos_counts.rename(columns={0: 'POS_COUNT'}, inplace=True)
    
    # Merge features
    app_train = app_train.merge(avg_pos_dpd, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(max_pos_dpd, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    app_test = app_test.merge(avg_pos_dpd, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(max_pos_dpd, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    del pos, avg_pos_dpd, max_pos_dpd, pos_counts
    gc.collect()

# 8. Process credit_card_balance data
with timer("Process Credit_card_balance data"):
    cc = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')
    
    # Calculate the number of credit cards for each customer
    cc_counts = cc.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().reset_index()
    cc_counts.rename(columns={'SK_ID_PREV': 'CC_COUNT'}, inplace=True)
    
    # Calculate average balance
    avg_cc_balance = cc.groupby('SK_ID_CURR')['AMT_BALANCE'].mean().reset_index()
    avg_cc_balance.rename(columns={'AMT_BALANCE': 'AVG_CC_BALANCE'}, inplace=True)
    
    # Calculate max DPD
    max_cc_dpd = cc.groupby('SK_ID_CURR')['SK_DPD'].max().reset_index()
    max_cc_dpd.rename(columns={'SK_DPD': 'MAX_CC_DPD'}, inplace=True)
    
    # Merge features
    app_train = app_train.merge(cc_counts, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(avg_cc_balance, on='SK_ID_CURR', how='left')
    app_train = app_train.merge(max_cc_dpd, on='SK_ID_CURR', how='left')
    
    app_test = app_test.merge(cc_counts, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(avg_cc_balance, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(max_cc_dpd, on='SK_ID_CURR', how='left')
    
    del cc, cc_counts, avg_cc_balance, max_cc_dpd
    gc.collect()

# 9. Fill missing values
with timer("Fill missing values"):
    # Fill missing values for numerical features
    app_train = app_train.fillna(-999)
    app_test = app_test.fillna(-999)
    
    print('Training set shape:', app_train.shape)
    print('Test set shape:', app_test.shape)

# 10. Train LightGBM model
with timer("Train LightGBM model"):
    # Prepare data
    features = [col for col in app_train.columns if col not in ['TARGET', 'SK_ID_CURR']]
    X = app_train[features]
    y = app_train['TARGET']
    X_test = app_test[features]
    
    print('Number of features:', len(features))
    
    # Check data
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('X_test shape:', X_test.shape)
    
    # Define cross-validation
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'learning_rate': 0.02,
        'num_leaves': 34,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'max_depth': 8,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'min_split_gain': 0.0222415,
        'min_child_weight': 39.3259775,
        'metric': 'auc',
        'verbose': -1
    }
    
    # Initialize prediction variables
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    feature_importance_df = pd.DataFrame()
    
    # Cross-validation training
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f'Fold {fold_}')
        
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
        
        # Use callbacks instead of verbose_eval
        callbacks = [
            lgb.callback.log_evaluation(period=200),
            lgb.callback.early_stopping(stopping_rounds=200)
        ]
        
        clf = lgb.train(
            params,
            trn_data,
            num_boost_round=10000,
            valid_sets=[trn_data, val_data],
            callbacks=callbacks
        )
        
        oof_preds[val_idx] = clf.predict(X.iloc[val_idx])
        test_preds += clf.predict(X_test) / n_folds
        
        # Record feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print(f'Fold {fold_} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])}')
        del clf, trn_data, val_data
        gc.collect()
    
    # Calculate cross-validation AUC
    cv_auc = roc_auc_score(y, oof_preds)
    print(f'Full AUC score: {cv_auc}')
    
    # Visualize feature importance
    plt.figure(figsize=(10, 20))
    feature_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_features = feature_importance.head(30).index
    
    sns.barplot(y=top_features, x=feature_importance[top_features], orient='h')
    plt.title('LightGBM Features (Top 30 by importance)')
    plt.tight_layout()
    plt.savefig('lgbm_feature_importance.png')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, oof_preds)
    plt.plot(fpr, tpr, label=f'CV AUC: {cv_auc:.4f}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve.png')

# 11. Generate prediction results
with timer("Generate prediction results"):
    submission = pd.DataFrame({
        'SK_ID_CURR': app_test['SK_ID_CURR'],
        'TARGET': test_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print('Prediction results have been saved as submission.csv')

print('Done! Final CV score:', cv_auc)

