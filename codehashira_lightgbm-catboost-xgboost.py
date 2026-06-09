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


#pip install optuna-integration[xgboost]


import optuna
#from optuna.integration import XGBoostPruningCallback
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
raw_train = train.copy()
raw_test = test.copy()
test_ids = test.pop('id')
train = train.drop('id', axis = 1)

def engineer_features(df):
    df_eng = df.copy()

    # --- CLEANUP ---
    # Drop gender to avoid bias/legal issues
    cols_to_drop = ['gender','marital_status']
    for col in cols_to_drop:
        df_eng = df_eng.drop(columns=[col], axis=1)

    # --- FINANCIAL REALITY CHECKS ---

    # 1. Estimated Monthly Disposable Income
    # Logic: (Annual Income / 12) * (Percentage of income NOT going to current debt)
    df_eng['monthly_disposable_cash'] = (df_eng['annual_income'] / 12) * (1 - df_eng['debt_to_income_ratio'])

    # 2. Loan Burden (LTI)
    # Logic: How many years of salary does this loan represent?
    # We add a small epsilon (1e-6) to avoid division by zero errors if income is 0.
    df_eng['loan_to_income'] = df_eng['loan_amount'] / (df_eng['annual_income'] + 1e-6)

    # 3. Market Skepticism (Rate per Credit Score Unit)
    # Logic: High rate + High Score = The lender knows something we don't.
    df_eng['rate_per_score_unit'] = df_eng['interest_rate'] / df_eng['credit_score']

    return df_eng

train, test = engineer_features(train), engineer_features (test)


cols = train.select_dtypes(include=['object', 'category'])
for col in cols:
    print(col, train[col].nunique
          ())


ct = pd.crosstab(train['grade_subgrade'], train['loan_paid_back'])
ct['Ratio_unpaid_to_paid'] = (ct[0]/ ct[1]) * 100
cr  = pd.crosstab(train['employment_status'], train['loan_paid_back'])
cr['Ratio_unpaid_to_paid'] = (cr[0]/ cr[1]) * 100
cr


'''kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

m_smoothing = 75
train['subgrade_mean_target_smooth'] = 0
global_mean = train['loan_paid_back'].mean()

for train_idx, test_idx in kf.split(train, train['loan_paid_back']):
    train_fold = train.iloc[train_idx]
    agg_map = train.groupby('grade_subgrade')['loan_paid_back'].agg(count='count', mean_p='mean')
    
    #Apply the Smoothing Formula
    agg_map['smooth_mean'] = (
        (agg_map['count'] * agg_map['mean_p']) + (m_smoothing * global_mean)
    ) / (agg_map['count'] + m_smoothing)
    mean_map = agg_map['smooth_mean']
    train.iloc[test_idx, train.columns.get_loc('subgrade_mean_target_smooth')] = \
        train.iloc[test_idx]['grade_subgrade'].map(mean_map)
train['subgrade_mean_target_smooth'] = train['subgrade_mean_target_smooth'].fillna(global_mean)
test['subgrade_mean_target_smooth'] = test['grade_subgrade'].map(mean_map)'''

y = train['loan_paid_back']
X = train.drop('loan_paid_back', axis = 1)
le = LabelEncoder()

cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
for col in cat_cols:
     X[col] = le.fit_transform(X[col])
     test[col] = le.transform(test[col])


X.head()


test.head()


'''X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state = 42)
def objective(trial):

    param = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators":trial.suggest_int("n_estimators", 100, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 100, log=True),
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-auc")

    model = XGBClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[pruning_callback],
        verbose=False
    )

    preds = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, preds)
    return auc_score
study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials =150)
print(f'Best score is {study.best_value}')
print(f'Best parameters are: {study.best_params}') '''


'''kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state =42)

param = {
    'n_estimators': 2002,
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 10,
    'eta': 0.02957806129572468,
    'gamma': 2.3585236766908477,
    'min_child_weight': 3.9635388764853836,
    'subsample': 0.9274662540352472,
    'colsample_bytree': 0.5441057712304507,
    'lambda': 4.439628259573784,
    'alpha': 2.1233780945010965,
    'scale_pos_weight': 1.8299853854618957}

model = XGBClassifier(**param)
scores = []

for fold, (train_idx,test_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    print(f'Running fold {fold + 1} out of {fold + 1}')
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        early_stopping_rounds = 100,
        verbose=False
    )
    y_pred = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, y_pred)
    print(f'Score for this fold {score}')
    scores.append(score)
pd.Series(scores).mean()'''


'''import shap 
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.show()
shap_abs_mean = np.abs(shap_values).mean(axis=0)

# Create a clean DataFrame for sorting
feature_importance = pd.DataFrame({
    'Feature': X_train.columns, 
    'SHAP_Value': shap_abs_mean
})

# Sort features by importance
feature_importance = feature_importance.sort_values(by='SHAP_Value', ascending=False).reset_index(drop=True)

print("\n--- SHAP Feature Ranking ---")
print(feature_importance.head(10))'''


'''import lightgbm as lgb
params = {'objective': 'binary',
          'metric': 'auc',
          'learning_rate': 0.07517425053533487,
          'num_leaves': 16, 'max_depth': 4,
          'min_child_samples': 41, 'subsample': 0.7301422811599733,
          'colsample_bytree': 0.7218067019593544, 'reg_alpha': 0.34738712520704734,
          'reg_lambda': 0.023068484168629146}
    
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_list = []
scores = []    
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold + 1}/{5}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        tr_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            tr_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)[:,]
        score = roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(score)
        print(f"Fold {fold + 1} AUC: {score:.6f}")
    
oof_score = roc_auc_score(y, oof_preds)
print(f"\nLightGBM OOF AUC: {oof_score:.6f} (+/- {np.std(scores):.6f})") '''



'''import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 32, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'verbosity': -1
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for tr_idx, val_idx in kf.split(X, y):
        train_data = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        valid_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=5000,
            callbacks=[
        lgb.early_stopping(50),
        optuna.integration.LightGBMPruningCallback(trial, "auc")
    ]
        )

        oof_preds[val_idx] = model.predict(X.iloc[val_idx])

    return roc_auc_score(y, oof_preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)'''



'''import shap 
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.show()
shap_abs_mean = np.abs(shap_values).mean(axis=0)

# Create a clean DataFrame for sorting
feature_importance = pd.DataFrame({
    'Feature': X_train.columns, 
    'SHAP_Value': shap_abs_mean
})

# Sort features by importance
feature_importance = feature_importance.sort_values(by='SHAP_Value', ascending=False).reset_index(drop=True)

print("\n--- SHAP Feature Ranking ---")
print(feature_importance.head(10)) '''


'''import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Example: your dataset
# X = pd.DataFrame(...)  # features
# y = pd.Series(...)     # target
# cat_features = ['employment_status', 'education_level', 'grade_subgrade']  # your categorical columns

n_splits = 2
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def objective(trial):

    # Optuna searches these hyperparameters
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_seed': 42,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'task_type': 'GPU'  # or 'CPU' if no GPU
    }

    oof_preds = np.zeros(len(y))

    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Use Pool to handle categorical features
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=100
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y, oof_preds)
    return auc

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # increase trials for better search

print("Best parameters found:", study.best_params)'''



def create_features(df, target_col=None):
    """
    Create derived features for loan prediction
    Adaptable to any column structure
    """
    df = df.copy()
    
    # Get column information
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID and target
    cols_to_remove = ['id']
    if target_col:
        cols_to_remove.append(target_col)
    
    for col in cols_to_remove:
        if col in numerical_cols:
            numerical_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    print(f"\nCreating features...")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Ratio features - common in loan datasets
    ratio_pairs = [
        ('loan_amnt', 'annual_inc'),
        ('installment', 'annual_inc'),
        ('revol_bal', 'annual_inc'),
        ('loan_amnt', 'installment'),
        ('revol_bal', 'revol_util')
    ]
    
    for col1, col2 in ratio_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_to_{col2}'] = df[col1] / (df[col2] + 1)
            print(f"Created: {col1}_to_{col2}")
    
    # Log transformations for monetary columns
    log_cols = ['loan_amnt', 'annual_inc', 'revol_bal', 'installment']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            print(f"Created: {col}_log")
    
    # Square root transformations
    sqrt_cols = ['loan_amnt', 'annual_inc']
    for col in sqrt_cols:
        if col in df.columns:
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
    
    # DTI related features
    if 'dti' in df.columns:
        df['dti_squared'] = df['dti'] ** 2
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['total_debt_burden'] = df['dti'] + (df['loan_amnt'] / (df['annual_inc'] + 1))
    
    # Employment length extraction
    if 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].fillna('0').astype(str).str.extract('(\d+)', expand=False).fillna(0).astype(float)
        df['emp_length_missing'] = df['emp_length'].isna().astype(int)
    
    # Credit history features
    if 'earliest_cr_line' in df.columns:
        try:
            df['credit_history_years'] = 2024 - pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce').dt.year
            df['credit_history_years'] = df['credit_history_years'].fillna(df['credit_history_years'].median())
        except:
            pass
    
    # Statistical features for numerical columns
    if len(numerical_cols) >= 3:
        num_data = df[numerical_cols].fillna(0)
        df['num_mean'] = num_data.mean(axis=1)
        df['num_std'] = num_data.std(axis=1)
        df['num_max'] = num_data.max(axis=1)
        df['num_min'] = num_data.min(axis=1)
    
    # Missing value indicators
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[f'{col}_missing'] = df[col].isna().astype(int)
    
    return df

def frequency_encode(train, test, col):
    """Frequency encoding for categorical variables"""
    freq = train[col].value_counts(normalize=True).to_dict()
    train[f'{col}_freq'] = train[col].map(freq)
    test[f'{col}_freq'] = test[col].map(freq).fillna(0)
    return train, test

def target_encode_cv(train, test, col, target, n_splits=5):
    """Target encoding with CV to prevent overfitting"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train[f'{col}_target'] = 0
    
    # CV encoding for train
    for tr_idx, val_idx in kf.split(train, train[target]):
        tr_mean = train.iloc[tr_idx].groupby(col)[target].mean()
        train.loc[val_idx, f'{col}_target'] = train.iloc[val_idx][col].map(tr_mean)
    
    # Fill remaining with global mean
    global_mean = train[target].mean()
    train[f'{col}_target'].fillna(global_mean, inplace=True)
    
    # Encode test with full train
    test_mean = train.groupby(col)[target].mean()
    test[f'{col}_target'] = test[col].map(test_mean).fillna(global_mean)
    
    return train, test

def prepare_features(train, test):
    """Complete feature preparation pipeline"""
    # Auto-detect target column
    possible_targets = ['loan_status', 'loan_paid_back', 'target', 'label']
    target_col = None
    for col in possible_targets:
        if col in train.columns:
            target_col = col
            break
    
    if target_col is None:
        # If none found, use the last column as target
        target_col = train.columns[-1]
    
    print(f"\nDetected target column: '{target_col}'")
    id_col = 'id'
    
    # Save IDs and target
    train_ids = train[id_col].values if id_col in train.columns else None
    test_ids = test[id_col].values if id_col in test.columns else None
    y = train[target_col].values if target_col in train.columns else None
    
    # Create features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    train = create_features(train, target_col)
    test = create_features(test, target_col)
    
    # Get categorical columns
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    for col in [id_col, target_col]:
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    print(f"\nCategorical columns to encode: {categorical_cols}")
    
    # Apply encodings
    for col in categorical_cols:
        if train[col].nunique() > 50:
            # High cardinality - use target encoding
            print(f"Target encoding: {col} (cardinality: {train[col].nunique()})")
            train, test = target_encode_cv(train, test, col, target_col)
        else:
            # Lower cardinality - use frequency encoding
            print(f"Frequency encoding: {col} (cardinality: {train[col].nunique()})")
            train, test = frequency_encode(train, test, col)
        
        # Label encode original column
        le = LabelEncoder()
        combined = pd.concat([train[col].fillna('missing'), test[col].fillna('missing')])
        le.fit(combined)
        train[col] = le.transform(train[col].fillna('missing'))
        test[col] = le.transform(test[col].fillna('missing'))
    
    # Select features
    feature_cols = [col for col in train.columns if col not in [id_col, target_col]]
    X_train = train[feature_cols].fillna(-999)
    X_test = test[feature_cols].fillna(-999)
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, y, X_test, train_ids, test_ids


def train_and_predict(X_train, y_train, X_test, model_name='lightgbm', n_folds=5):
    """Train model and generate test predictions"""
    params_dict = {
        'lightgbm': 
            {'objective': 'binary',
              'metric': 'auc',
              'learning_rate': 0.07517425053533487,
              'num_leaves': 16, 'max_depth': 4,
              'min_child_samples': 41, 'subsample': 0.7301422811599733,
              'colsample_bytree': 0.7218067019593544, 'reg_alpha': 0.34738712520704734,
              'reg_lambda': 0.023068484168629146},
    'xgboost': {
        'objective': 'binary:logistic',
        'n_estimators': 2002,
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 10,
        'eta': 0.02957806129572468,
        'gamma': 2.3585236766908477,
        'min_child_weight': 3.9635388764853836,
        'subsample': 0.9274662540352472,
        'colsample_bytree': 0.5441057712304507,
        'lambda': 4.439628259573784,
        'alpha': 2.1233780945010965,
        'scale_pos_weight': 1.8299853854618957},
        
    'catboost': {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 7,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 200,
            'early_stopping_rounds': 100
        },
    }
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n{model_name.upper()} - Fold {fold + 1}/{n_folds}")
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        if model_name == 'lightgbm':
            tr_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(params_dict[model_name], tr_data, num_boost_round=3000,
                            valid_sets=[val_data], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / n_folds
            
        elif model_name == 'xgboost':
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)
            model = xgb.train(params_dict[model_name], dtr, num_boost_round=3000,
                            evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=200)
            oof_preds[val_idx] = model.predict(dval)
            test_preds += model.predict(dtest) / n_folds
            
        elif model_name == 'catboost':
            model = CatBoostClassifier(**params_dict[model_name])
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      cat_features = cat_cols,
                      use_best_model=True, verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        score = roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(score)
        print(f"Fold {fold + 1} AUC: {score:.6f}")
    
    oof_score = roc_auc_score(y_train, oof_preds)
    print(f"\n{model_name.upper()} OOF AUC: {oof_score:.6f} (+/- {np.std(scores):.6f})")
    
    return oof_preds, test_preds, oof_score






# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

"""Main execution pipeline"""
print("="*80)
print("LOAN PAYBACK PREDICTION - KAGGLE COMPETITION")
print("="*80)


# Train models and get predictions
print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

print("\n[1/3] Training LightGBM...")
lgb_oof, lgb_test, lgb_score = train_and_predict(X, y, test, 'lightgbm', n_folds=5)

print("\n[2/3] Training XGBoost...")
xgb_oof, xgb_test, xgb_score = train_and_predict(X, y, test, 'xgboost', n_folds=5)




from catboost import CatBoostClassifier
print("\n[3/3] Training CatBoost...")
X_raw, y_raw, raw_test, d,x = prepare_features(raw_train, raw_test)
cat_oof, cat_test, cat_score = train_and_predict(X_raw, y_raw, raw_test, 'catboost', n_folds=5)


from sklearn.linear_model import LogisticRegression
X_meta_train = pd.DataFrame({'lgb_oof': lgb_oof,
                                'xgb_oof': xgb_oof,
                                 'cat_oof': cat_oof
                                })
X_meta_test = pd.DataFrame({'lgb_oof': lgb_test,
                           'xgb_oof': xgb_test,
                           'cat_oof': cat_test})
meta_model_base = RidgeClassifier(random_state=42)
meta_model = LogisticRegression 
meta_model.fit(X_meta_train, y)
meta_pred = meta_model.predict_proba(X_meta_test)[:,1]

'''# Ensemble predictions
print("\n" + "="*80)
print("ENSEMBLE PREDICTIONS")
print("="*80)

# Weight by CV scores
total = lgb_score + xgb_score + cat_score
w_lgb = lgb_score / total
w_xgb = xgb_score / total
w_cat = cat_score / total

print(f"\nWeights based on CV scores:")
print(f"  LightGBM: {w_lgb:.4f} (AUC: {lgb_score:.6f})")
print(f"  XGBoost:  {w_xgb:.4f} (AUC: {xgb_score:.6f})")
print(f"  CatBoost: {w_cat:.4f} (AUC: {cat_score:.6f})")

ensemble_test = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test

# Create submissions'''
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': meta_pred
})
submission.to_csv('submission.csv', index=False)

# Individual model submissions
pd.DataFrame({'id': test_ids, 'loan_status': lgb_test}).to_csv('submission_lgb.csv', index=False)
pd.DataFrame({'id': test_ids, 'loan_status': xgb_test}).to_csv('submission_xgb.csv', index=False)
pd.DataFrame({'id': test_ids, 'loan_status': cat_test}).to_csv('submission_cat.csv', index=False)

print("\n" + "="*80)
print("COMPLETE! Submission files created:")
print("  ✓ submission.csv (weighted ensemble)")
print("  ✓ submission_lgb.csv")
print("  ✓ submission_xgb.csv")
print("  ✓ submission_cat.csv")
print("="*80)

