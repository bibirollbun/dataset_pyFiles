import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import gc


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print(f"\nTrain: {train.shape}")
print(f"Test:  {test.shape}")
print(f"submission:  {submission.shape}")
print(f"Orig:  {orig.shape}")




# Combine all data
target = 'loan_paid_back'

CATS_BASE = ['gender', 'marital_status', 'education_level', 'employment_status', 
             'loan_purpose', 'grade_subgrade']
NUMS_BASE = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
             'loan_amount', 'interest_rate']


test[target] = -1
combine = pd.concat([train, test, orig], axis=0, ignore_index=True)

print(f"\nCombined data: {combine.shape}")


train.head()


test.head()


train.info()


train.dtypes


print("Target column statistics (loan_paid_back):")

train['loan_paid_back'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train.describe().T


def remove_outliers(train_df, test_df=None):
    
    train_df = train_df.copy()
    

    zscore_features = ['credit_score', 'interest_rate']
    limits = {}
    
    for feature in zscore_features:
        if feature in train_df.columns:
            mean = train_df[feature].mean()
            std = train_df[feature].std()
            limits[f'{feature}_upper'] = mean + 3 * std
            limits[f'{feature}_lower'] = mean - 3 * std
            train_df[feature] = np.clip(train_df[feature], limits[f'{feature}_lower'], limits[f'{feature}_upper'])
    

    iqr_features = [
        'annual_income', 
        'debt_to_income_ratio', 
        'loan_amount',
        'total_debt',
        'available_income',
        'monthly_payment',
        'payment_to_income',
        'monthly_income',
        'monthly_debt',
        'remaining_income',
        'income_credit_interaction',
        'debt_credit_interaction'
    ]
    
    for feature in iqr_features:
        if feature in train_df.columns:
            Q1 = train_df[feature].quantile(0.25)
            Q3 = train_df[feature].quantile(0.75)
            IQR = Q3 - Q1
            limits[f'{feature}_lower'] = Q1 - 1.5 * IQR
            limits[f'{feature}_upper'] = Q3 + 1.5 * IQR
            train_df[feature] = np.clip(train_df[feature], limits[f'{feature}_lower'], limits[f'{feature}_upper'])
    

    ratio_features = {
        'debt_to_income_ratio': (0, 0.7),  
        'payment_to_income_ratio': (0, 0.5),  
        'income_loan_ratio': (0, 100), 
        'loan_to_income': (0, 5),  
        'affordability': (0, 10), 
        'default_risk': (0, 1), 
        'risk_score': (0, 0.5),  
    }
    
    for feature, (lower, upper) in ratio_features.items():
        if feature in train_df.columns:
            train_df[feature] = np.clip(train_df[feature], lower, upper)
            limits[f'{feature}_lower'] = lower
            limits[f'{feature}_upper'] = upper
    
   
    if test_df is not None:
        test_df = test_df.copy()
        
        
        for feature in zscore_features:
            if feature in test_df.columns:
                test_df[feature] = np.clip(
                    test_df[feature], 
                    limits[f'{feature}_lower'], 
                    limits[f'{feature}_upper']
                )
        
       
        for feature in iqr_features:
            if feature in test_df.columns:
                test_df[feature] = np.clip(
                    test_df[feature], 
                    limits[f'{feature}_lower'], 
                    limits[f'{feature}_upper']
                )
        
        
        for feature, (lower, upper) in ratio_features.items():
            if feature in test_df.columns:
                test_df[feature] = np.clip(test_df[feature], lower, upper)
        
    return train_df, test_df
    


def engineer_features(df):
    df = df.copy()

  
    # Basic Ratios
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Debt Metrics
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    df['debt_burden'] = df['debt_to_income_ratio'] * df['loan_amount']
    
    # Payment Analysis
    df['monthly_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12 + 1)
    df['affordability'] = df['available_income'] / (df['loan_amount'] + 1)
    
    # Risk Scoring
    df['default_risk'] = (
        df['debt_to_income_ratio'] * 0.40 +
        (850 - df['credit_score']) / 850 * 0.35 +
        df['interest_rate'] / 100 * 0.25
    )
    
    # Credit Analysis
    df['credit_utilization'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['credit_interest_product'] = df['credit_score'] * df['interest_rate'] / 100
    
    # Log Transformations
    df['annual_income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    
    # Grade Parsing
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map)
    
    # Monthly Profiles
    df['monthly_income'] = df['annual_income'] / 12
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    df['monthly_debt'] = df['total_debt'] / 12
    df['remaining_income'] = df['monthly_income'] - df['monthly_debt']
    
    # Efficiency & Interactions
    df['credit_efficiency'] = df['credit_score'] / (df['debt_to_income_ratio'] + 0.001)
    df['risk_score'] = (df['debt_to_income_ratio'] * df['interest_rate']) / (df['credit_score'] + 1)
    df['income_credit_interaction'] = df['annual_income'] * df['credit_score']
    df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_score']
    
    # Polynomial Features
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['debt_ratio_squared'] = df['debt_to_income_ratio'] ** 2
    
    # Combined Categorical Features
    df['gender_marital'] = (df['gender'] + '_' + df['marital_status']).astype('category')
    df['education_employment'] = (df['education_level'] + '_' + df['employment_status']).astype('category')
    
    # Risk Flags
    df['high_risk_flag'] = (
        (df['debt_to_income_ratio'] > 0.4) |
        (df['credit_score'] < 650) |
        (df['interest_rate'] > 15)
    ).astype(int)
    df['excellent_credit_flag'] = (df['credit_score'] >= 750).astype(int)
    df['high_income_flag'] = (df['annual_income'] >= 50000).astype(int)
    df['has_advanced_degree'] = df['education_level'].isin(["Master's", "PhD"]).astype(int)

  
    
    # Debt-to-Credit Ratio
    df['debt_to_credit_score'] = df['debt_to_income_ratio'] / (df['credit_score'] / 850)
    
    # Payment Burden Score (combines multiple risk factors)
    df['payment_burden_score'] = (
        df['payment_to_income_ratio'] * 0.4 +
        df['debt_to_income_ratio'] * 0.3 +
        (df['interest_rate'] / 20) * 0.3
    )
    
    # Credit Score Bins (categorical)
    df['credit_score_bin'] = (pd.cut(
        df['credit_score'], 
        bins=[0, 580, 670, 740, 800, 850],
        labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent']
    )).astype('category')
    
    # DTI Bins
    df['dti_bin'] = (pd.cut(
        df['debt_to_income_ratio'],
        bins=[0, 0.1, 0.2, 0.35, 1.0],
        labels=['Low', 'Medium', 'High', 'Very_High']
    )).astype('category')
    
    # Income Stability Score (combines income and employment)
    df['income_stability'] = df['annual_income'] * (
        df['employment_status'].map({
            'Employed': 1.0,
            'Self-employed': 0.8,
            'Unemployed': 0.3
        }).fillna(0.5)
    )
    
    # Loan Affordability Index
    df['loan_affordability_index'] = (
        df['available_income'] / (df['monthly_payment'] * 12 + 1)
    )
    
    # Interest Rate vs Credit Score Mismatch
   
    df['rate_credit_mismatch'] = df['interest_rate'] - (20 - df['credit_score'] / 50)
    
    # Relative Loan Size
    df['relative_loan_size'] = df['loan_amount'] / df['annual_income'].median()
    
    # Risk Tier 
    df['risk_tier'] = 'Medium'
    df.loc[
        (df['credit_score'] >= 720) & 
        (df['debt_to_income_ratio'] <= 0.3) & 
        (df['interest_rate'] <= 12), 
        'risk_tier'
    ] = 'Low'
    df.loc[
        (df['credit_score'] < 640) | 
        (df['debt_to_income_ratio'] > 0.4) | 
        (df['interest_rate'] > 15), 
        'risk_tier'
    ] = 'High'
    df['risk_tier'] = df['risk_tier'].astype('category')
    
    # Grade-Income Interaction
    df['grade_income_ratio'] = df['grade_rank'] / (df['annual_income_log'] + 1)
    
    # Total Financial Pressure
    df['total_financial_pressure'] = (
        df['debt_to_income_ratio'] * 0.3 +
        df['payment_to_income_ratio'] * 0.3 +
        (df['loan_amount'] / df['annual_income']) * 0.2 +
        ((20 - df['interest_rate']) / 20) * 0.2
    )
    
    # Credit Score Momentum 
    expected_rate = 20 - (df['credit_score'] - 300) / 30
    df['credit_rate_deviation'] = df['interest_rate'] - expected_rate

    flags = ['high_risk_flag', 'excellent_credit_flag', 'high_income_flag', 'has_advanced_degree']
    for f in flags:
        df[f] = df[f].astype('int8')

    
    return df



combine = engineer_features(combine)


NEW_FEATURES = [  
    'income_loan_ratio', 'loan_to_income', 'total_debt', 'available_income', 'debt_burden',
    'monthly_payment', 'payment_to_income', 'affordability', 'payment_to_income_ratio',
    'monthly_income', 'monthly_debt', 'remaining_income', 'default_risk', 'credit_utilization',
    'credit_interest_product', 'credit_efficiency', 'risk_score', 'annual_income_log',
    'loan_amount_log', 'grade_letter', 'grade_number', 'grade_rank', 'income_credit_interaction',
    'debt_credit_interaction', 'credit_score_squared', 'debt_ratio_squared', 'gender_marital',
    'education_employment', 'high_risk_flag', 'excellent_credit_flag', 'high_income_flag',
    'has_advanced_degree','debt_to_credit_score', 'payment_burden_score', 'credit_score_bin', 'dti_bin',
    'income_stability', 'loan_affordability_index', 'rate_credit_mismatch', 'relative_loan_size',
    'risk_tier', 'grade_income_ratio', 'total_financial_pressure', 'credit_rate_deviation'
]

RAW_NUMS = [
    'age', 'loan_term', 'installment', 'num_of_open_accounts', 
    'total_credit_limit', 'current_balance', 'delinquency_history', 
    'num_of_delinquencies', 'public_records'
]


print(f"Created {len(NEW_FEATURES)} new features")





NUMS = NUMS_BASE + [f for f in NEW_FEATURES if f not in ['grade_letter']] + RAW_NUMS
CATS = CATS_BASE.copy()
CATS.append('grade_letter')

# Factorize numeric columns to create categorical numeric features
CATS_NUM = []
SIZES = {}
for c in NUMS:
    if combine[c].dtype.name.startswith(('int', 'float')):
        n = f"{c}_cat"
        combine[n], _ = combine[c].factorize()
        combine[n] = combine[n].astype('int32')
        CATS_NUM.append(n)
        SIZES[n] = combine[n].max() + 1
print(f"Created {len(CATS_NUM)} categorical numeric features")

# --- 2-way interactions ---
important_pairs = [
    ('employment_status', 'grade_subgrade'),
    ('employment_status', 'education_level'),
    ('employment_status', 'loan_purpose'),
    ('grade_subgrade', 'loan_purpose'),
    ('grade_subgrade', 'education_level'),
    ('marital_status', 'employment_status'),
]

# Add numeric-cat interactions
for num_cat in ['credit_score_cat', 'debt_to_income_ratio_cat', 'interest_rate_cat']:
    for cat in ['employment_status', 'grade_subgrade']:
        important_pairs.append((num_cat, cat))

CATS_INTER = []
for c1, c2 in important_pairs:
    if c1 in combine.columns and c2 in combine.columns:
        name = f"{c1}_{c2}"
        combine[name] = combine[c1].astype(str) + '_' + combine[c2].astype(str)
        combine[name] = combine[name].astype('category')
        CATS_INTER.append(name)
print(f"Created {len(CATS_INTER)} strategic interactions")

# --- Count encoding for all categorical features ---
CE = []
ALL_CATS = CATS + CATS_NUM + CATS_INTER
for c in ALL_CATS:
    tmp = combine.groupby(c)[target].count()
    tmp.name = f"CE_{c}"
    if f"CE_{c}" not in combine.columns:
        combine = combine.merge(tmp, on=c, how='left')
    CE.append(f"CE_{c}")
print(f"Created {len(CE)} count encodings")

# --- Split back ---
train = combine.iloc[:len(train)].copy()
test = combine.iloc[len(train):len(train) + len(test)].copy()
orig = combine.iloc[-len(orig):].copy()
print(f"\nTrain: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

# --- Build correct numeric & categorical lists ---
categorical_cols = [c for c in train.columns if str(train[c].dtype) in ['category', 'object']]
numeric_cols = [c for c in train.columns if c not in categorical_cols and c not in ['id', target]]


print(f"Total categorical features: {len(categorical_cols)}")
print(f"Total numeric features: {len(numeric_cols)}")
print(f"Total features: {len(categorical_cols) + len(numeric_cols)}")



FEATURES = NUMS + CATS + CATS_NUM + CATS_INTER + CE
print(f"\n Total Features: {len(FEATURES)}")


train.head()


train = train.drop(columns=['id'], errors='ignore')
test = test.drop(columns=['id'], errors='ignore')


train, test = remove_outliers(train, test)


train.columns


train.info()


def reduce_memory(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            
            if df[col].nunique() < df[col].count() * 0.5:
                df[col] = df[col].astype('category')

        elif str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")

        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory reduced from {start_mem:.2f} MB → {end_mem:.2f} MB "
              f"({(start_mem - end_mem)/start_mem*100:.1f}% saved)")
    
    return df



train = reduce_memory(train)
test = reduce_memory(test)


train.info()


y_train = train['loan_paid_back']
X_train = train.drop('loan_paid_back', axis=1)

X_test = test.copy()


X_train.shape


preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore'),
         categorical_cols),
        
        ('scale', MinMaxScaler(),
         numeric_cols)
    ],
    sparse_threshold=1.0
)


def target_encode(X_train_fold, y_train_fold, X_valid_fold, X_test_fold, cat_cols, smoothing=10):
    
  
    X_train_enc = X_train_fold.copy(deep=True)
    X_valid_enc = X_valid_fold.copy(deep=True)
    X_test_enc = X_test_fold.copy(deep=True)
    
    global_mean = y_train_fold.mean()
    
    for col in cat_cols:
 
        train_col_values = X_train_fold[col].astype(str)
        valid_col_values = X_valid_fold[col].astype(str)
        test_col_values = X_test_fold[col].astype(str)
        
        
        stats = pd.DataFrame({
            'target': y_train_fold.values,
            'category': train_col_values.values
        }).groupby('category')['target'].agg(['sum', 'count'])
        
        
        stats['smoothed_target'] = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
     
        target_encode_map = stats['smoothed_target'].to_dict()
        
   
        X_train_enc[f'{col}_target_enc'] = train_col_values.map(target_encode_map).fillna(global_mean).astype('float32')
        X_valid_enc[f'{col}_target_enc'] = valid_col_values.map(target_encode_map).fillna(global_mean).astype('float32')
        X_test_enc[f'{col}_target_enc'] = test_col_values.map(target_encode_map).fillna(global_mean).astype('float32')
    
    return X_train_enc, X_valid_enc, X_test_enc



FOLDS = 8 
SEED = 42 

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.008,  
    "max_depth": 0,
    "subsample": 0.85,  
    "colsample_bytree": 0.75,  
    "colsample_bylevel": 0.75,  
    "colsample_bynode": 0.8,  
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide",
    "max_leaves": 40,  
    'scale_pos_weight': 0.8, 
    "min_child_weight": 3,  
    'lambda': 5.0,  
    'alpha': 2.5,  
    'max_bin': 256,
    'gamma': 0.1, 
}


print(f"\nTraining {FOLDS}-Fold STRATIFIED Cross-Validation")
print("=" * 80)


oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []
best_iterations = []
feature_importance_list = []


skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


print(f"\n Feature Summary:")
print(f"   Total Features: {len(X_train.columns)}")
print(f"   Categorical: {len(categorical_cols)}")
print(f"   Numeric: {len(numeric_cols)}")
print(f"   Target Encoding will be applied to {len(categorical_cols)} categorical features")
print("=" * 80)


for fold, (train_idx, val_idx) in enumerate(tqdm(
    skf.split(X_train, y_train), 
    total=FOLDS, 
    desc="Training Folds",
    position=0
)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{FOLDS}")
    print('='*50)
    
   
    X_train_fold = X_train.iloc[train_idx].copy()
    y_train_fold = y_train.iloc[train_idx]
    X_valid_fold = X_train.iloc[val_idx].copy()
    y_valid = y_train.iloc[val_idx]
    X_test_fold = X_test.copy()
    
  
    print(f"\n Class Distribution:")
    train_dist = y_train_fold.value_counts(normalize=True).sort_index()
    valid_dist = y_valid.value_counts(normalize=True).sort_index()
    print(f"   Train: {dict(train_dist)}")
    print(f"   Valid: {dict(valid_dist)}")
    
 
    print(f"\n  Fitting preprocessor on original features...")
    preprocessor.fit(X_train_fold)
    
   
    if len(categorical_cols) > 0:
        print(f"\n Applying target encoding to {len(categorical_cols)} categorical features...")
        X_train_fold, X_valid_fold, X_test_fold = target_encode(
            X_train_fold,
            y_train_fold,
            X_valid_fold,
            X_test_fold,
            categorical_cols,
            smoothing=10
        )
        print(f"Target encoding complete. Added {len(categorical_cols)} new features.")
    
 
    print(f"\n Applying preprocessing transformations...")
    X_train_prep = preprocessor.transform(X_train_fold)
    X_valid_prep = preprocessor.transform(X_valid_fold)
    X_test_prep = preprocessor.transform(X_test_fold)
    

    dtrain = xgb.DMatrix(X_train_prep, label=y_train_fold)
    dval = xgb.DMatrix(X_valid_prep, label=y_valid)
    dtest = xgb.DMatrix(X_test_prep)
    

    print(f"\n Training XGBoost model...")
    evals_result = {}
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=12000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=350,
        verbose_eval=500,
        evals_result=evals_result
    )
    
    best_iter = model.best_iteration
    best_iterations.append(best_iter)
    
  
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, best_iter + 1))
    test_preds += model.predict(dtest, iteration_range=(0, best_iter + 1)) / FOLDS
    
   
    fold_auc = roc_auc_score(y_valid, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    
 
    importance = model.get_score(importance_type='gain')
    feature_importance_list.append(importance)
    
  
    print(f"\n✅ Fold {fold+1} Results:")
    print(f"   AUC Score: {fold_auc:.5f}")
    print(f"   Best Iteration: {best_iter}")
    print(f"   Train Size: {len(train_idx):,}")
    print(f"   Valid Size: {len(val_idx):,}")
    
 
    del dtrain, dval, dtest, model, X_train_prep, X_valid_prep, X_test_prep
    del X_train_fold, X_valid_fold, X_test_fold
    gc.collect()







print("\n" + "="*80)
print(" CROSS-VALIDATION RESULTS")
print("="*80)

overall_auc = roc_auc_score(train[target], oof_preds)

print(f"\n  Fold-wise Performance:")
for i, (score, best_iter) in enumerate(zip(fold_scores, best_iterations), 1):
    print(f"   Fold {i}: AUC={score:.5f} | Best Iter={best_iter}")

print(f"\n Summary Statistics:")
print(f"   Mean CV AUC:     {np.mean(fold_scores):.5f}")
print(f"   Std CV AUC:      {np.std(fold_scores):.5f}")
print(f"   Overall OOF AUC: {overall_auc:.5f}")
print(f"   Avg Best Iter:   {int(np.mean(best_iterations))}")

print(f"\n Prediction Statistics:")
print(f"   OOF Min:  {oof_preds.min():.5f}")
print(f"   OOF Max:  {oof_preds.max():.5f}")
print(f"   OOF Mean: {oof_preds.mean():.5f}")
print(f"   Test Min:  {test_preds.min():.5f}")
print(f"   Test Max:  {test_preds.max():.5f}")
print(f"   Test Mean: {test_preds.mean():.5f}")


print("\n" + "="*80)
print(" FEATURE IMPORTANCE ANALYSIS")
print("="*80)


all_features = set()
for imp_dict in feature_importance_list:
    all_features.update(imp_dict.keys())


avg_importance = {}
for feature in all_features:
    importances = [imp_dict.get(feature, 0) for imp_dict in feature_importance_list]
    avg_importance[feature] = np.mean(importances)


importance_df = pd.DataFrame({
    'feature': list(avg_importance.keys()),
    'importance': list(avg_importance.values())
}).sort_values('importance', ascending=False)

print(f"\n TOP 20 FEATURES BY IMPORTANCE (Average across {FOLDS} folds):")
print("-" * 80)
top_20 = importance_df.head(20).reset_index(drop=True)
top_20.index = top_20.index + 1
print(top_20.to_string())

print(f"\n BOTTOM 10 FEATURES BY IMPORTANCE:")
print("-" * 80)
bottom_10 = importance_df.tail(10).reset_index(drop=True)
bottom_10.index = range(len(importance_df) - 9, len(importance_df) + 1)
print(bottom_10.to_string())


importance_df.to_csv('feature_importance.csv', index=False)
print(f"\n Feature importance saved to: feature_importance.csv")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)



fig, axes = plt.subplots(1, 2, figsize=(16, 5))


axes[0].bar(range(1, FOLDS+1), fold_scores, color='steelblue', edgecolor='black')
axes[0].axhline(overall_auc, color='red', linestyle='--', label=f'Overall: {overall_auc:.5f}')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('ROC AUC')
axes[0].set_title('Cross-Validation Fold Scores', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)


sns.histplot(oof_preds, bins=50, kde=True, color='coral', ax=axes[1])
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Frequency')
axes[1].set_title('OOF Prediction Distribution', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()




submission['loan_paid_back'] = test_preds
submission.to_csv('submission.csv', index=False)

print("Submission saved to 'submission.csv'")

print(f"Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")


import pickle


with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved to 'preprocessor.pkl'")


model.save_model('xgboost_model.json')
print("XGBoost model saved to 'xgboost_model.json'")


np.save('xgb_test_predictions.npy', test_preds)
print("XGBoost test predictions saved")


np.save('xgb_oof_predictions.npy', oof_preds)
print("XGBoost OOF predictions saved")

