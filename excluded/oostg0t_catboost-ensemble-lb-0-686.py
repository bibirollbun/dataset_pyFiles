!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz -q
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl -q


import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# File paths
TRAIN_PATH = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
TEST_PATH = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv'
DATA_DICT_PATH = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'

# Load data
train_df, test_df = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
sample_sub, data_dict = pd.read_csv(SAMPLE_SUB_PATH), pd.read_csv(DATA_DICT_PATH).iloc[:57]

# Extract categorical and numerical features
categorical_features = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
numerical_features = data_dict[data_dict['type'] == 'Numerical']['variable'].tolist()

# Initialize storage for global imputation values
imputation_values = {
    'categorical': {},
    'numerical': {},
    'encoder': {}
}

class FE:
    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def load_data(self, path):
        return pl.read_csv(path)

    def recalculate_hla_sums(self, df):
        df = df.with_columns(
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_nmdp_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_low").fill_null(0)).alias("hla_low_res_6"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_high_res_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0)
            ).alias("hla_low_res_8"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_high_res_8"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0) +
             pl.col("hla_match_dqb1_low").fill_null(0)).alias("hla_low_res_10"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0) +
             pl.col("hla_match_dqb1_high").fill_null(0)).alias("hla_high_res_10"),
        )
        return df

    def cast_datatypes(self, df):
        num_cols = [
            'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6',
            'hla_low_res_6', 'hla_high_res_10', 'hla_low_res_10',
            'hla_match_dqb1_high', 'hla_match_dqb1_low',
            'hla_match_drb1_high', 'hla_match_drb1_low',
            'hla_nmdp_6', 'year_hct', 'hla_match_a_high',
            'hla_match_a_low', 'hla_match_b_high', 'hla_match_b_low',
            'hla_match_c_high', 'hla_match_c_low', 'donor_age',
            'age_at_hct', 'comorbidity_score', 'karnofsky_score',
            'efs', 'efs_time'
        ]

        for col in df.columns:
            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))  
            else:
                df = df.with_columns(pl.col(col).fill_null('Unknown').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))

    def info(self, df):
        print(f'\nShape of dataframe: {df.shape}')    
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        display(df.head())

    def apply_fe(self, path):
        df = self.load_data(path)
        df = self.recalculate_hla_sums(df)
        df = self.cast_datatypes(df)
        df = df.to_pandas()
        self.info(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        return df, cat_cols

def preprocess_data(df, train=True):
    df = df.copy()
    nan_indicators = []
    
    # Process categorical features
    for col in categorical_features:
        nan_indicator = f"IS_NAN_{col}"
        df[nan_indicator] = df[col].isna().astype(int)
        nan_indicators.append(nan_indicator)
        
        if train:
            mode_val = df[col].mode()[0]
            imputation_values['categorical'][col] = mode_val
            df[col] = df[col].fillna('Missing')
        else:
            df[col] = df[col].fillna(imputation_values['categorical'].get(col, 'Missing'))

    # Process numerical features
    for col in numerical_features:
        nan_indicator = f"IS_NAN_{col}"
        df[nan_indicator] = df[col].isna().astype(int)
        nan_indicators.append(nan_indicator)
        
        if train:
            median_val = df[col].median()
            imputation_values['numerical'][col] = median_val
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna(imputation_values['numerical'].get(col, 0))

    # Label encode categorical features
    if train:
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            imputation_values['encoder'][col] = le
    else:
        for col in categorical_features:
            le = imputation_values['encoder'].get(col)
            if le:
                df[col] = df[col].map(lambda x: x if x in le.classes_ else 'Missing')
                df[col] = le.transform(df[col].astype(str))

    return df, nan_indicators

# Apply feature engineering
fe = FE()
train_df, train_cat_cols = fe.apply_fe(TRAIN_PATH)
test_df, test_cat_cols = fe.apply_fe(TEST_PATH)

# Apply preprocessing
train_df, train_nan_indicators = preprocess_data(train_df, train=True)
test_df, test_nan_indicators = preprocess_data(test_df, train=False)

# Update feature lists
final_categorical = categorical_features.copy()
final_numerical = numerical_features + train_nan_indicators

# Verify feature types
print(f"Categorical features ({len(final_categorical)}): {final_categorical}")
print(f"Numerical features ({len(final_numerical)}): {final_numerical[:5]}...")


from lifelines import NelsonAalenFitter, KaplanMeierFitter, WeibullFitter, LogNormalFitter, CoxPHFitter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def remove_highly_correlated_features(df, threshold=0.9):
    """Remove features with correlation greater than threshold"""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def add_target_transformations_oof(train_df, n_splits=5, random_state=42):
    """
    Adds survival analysis target transformations using out-of-fold estimates to prevent data leakage.
    - Nelson-Aalen cumulative hazard
    - Kaplan-Meier survival probability
    - Weibull survival probability
    - Log-Normal survival probability
    - CoxPH survival probability
    - Mathematical transforms
    """
    df = train_df.copy()
    
    # Initialize new columns
    df['na_cumulative_hazard'] = np.nan
    df['km_survival_prob'] = np.nan
    df['weibull_survival'] = np.nan
    df['lognormal_survival'] = np.nan
    
    # Identify features for CoxPH (exclude time, event, and new target columns)
    original_columns = train_df.columns
    new_target_columns = ['na_cumulative_hazard', 'km_survival_prob', 
                          'weibull_survival', 'lognormal_survival']
    features = original_columns.drop(['efs_time', 'efs'] + new_target_columns, errors='ignore')
    
    # Split into folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in tqdm(kf.split(df)):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        # 1. Nelson-Aalen Cumulative Hazard
        naf = NelsonAalenFitter()
        naf.fit(train_fold['efs_time'], event_observed=train_fold['efs'])
        na_pred = naf.cumulative_hazard_at_times(val_fold['efs_time'].values)
        df.loc[val_idx, 'na_cumulative_hazard'] = na_pred.values
        
        # 2. Kaplan-Meier Survival Probability
        kmf = KaplanMeierFitter()
        kmf.fit(train_fold['efs_time'], event_observed=train_fold['efs'])
        km_pred = kmf.survival_function_at_times(val_fold['efs_time'].values)
        df.loc[val_idx, 'km_survival_prob'] = km_pred.values
        
        # 3. Weibull Survival Probability
        wf = WeibullFitter()
        wf.fit(train_fold['efs_time'], event_observed=train_fold['efs'])
        weibull_pred = wf.survival_function_at_times(val_fold['efs_time'].values)
        df.loc[val_idx, 'weibull_survival'] = weibull_pred.values
        
        # 4. Log-Normal Survival Probability
        lnf = LogNormalFitter()
        lnf.fit(train_fold['efs_time'], event_observed=train_fold['efs'])
        ln_pred = lnf.survival_function_at_times(val_fold['efs_time'].values)
        df.loc[val_idx, 'lognormal_survival'] = ln_pred.values
    
    # Additional transformations based on OOF features
    df['cloglog_survival'] = np.log(-np.log(df['km_survival_prob'] + 1e-8))
    df['target_for_cox'] = df['efs_time'] * df['efs'].apply(lambda x: 1 if x == 1 else -1)
    df['combined_risk'] = df['na_cumulative_hazard'] * (1 - df['km_survival_prob'])
    
    return df

# Apply transformations
train_df = add_target_transformations_oof(train_df)

# Verify new features
train_df[['efs_time', 'efs', 'na_cumulative_hazard', 
          'km_survival_prob', 'weibull_survival', 
          'lognormal_survival', 
          'combined_risk', 'target_for_cox', 
          'cloglog_survival']].head()


MODELS_CFG = [
    {
        'model_name': 'catboost_cox_loss',
        'target': 'target_for_cox',
        'needs_minus': True,
        'params': {
            'bootstrap_type': 'Bernoulli',
            'grow_policy': 'Lossguide',
            'iterations': 1795,
            'learning_rate': 0.05235183813255633,
            'depth': 10,
            'l2_leaf_reg': 1.1718548801748443,
            'max_bin': 352,
            'random_strength': 6.520306660612896,
            'subsample': 0.5937132193141078,
            'max_leaves': 26,
            'verbose': False,
            'loss_function': 'Cox',
        }
    },
    {
        'model_name': 'catboost_km',
        'target': 'km_survival_prob',
        'needs_minus': True,
        'params': {
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'Depthwise',
            'iterations': 2596,
            'learning_rate': 0.021542867970092913,
            'depth': 6,
            'l2_leaf_reg': 9.826923593795929,
            'min_data_in_leaf': 63,
            'max_bin': 384,
            'random_strength': 0.7644605525172539,
            'bagging_temperature': 0.4878153260468965,
            'verbose': False,
        }
    },
    {
        'model_name': 'catboost_na',
        'target': 'na_cumulative_hazard',
        'needs_minus': False,
        'params': {
            'bootstrap_type': 'Bernoulli',
            'grow_policy': 'Lossguide',
            'iterations': 1875,
            'learning_rate': 0.01979088008683825,
            'depth': 4,
            'l2_leaf_reg': 4.804530282234728e-07,
            'min_data_in_leaf': 36,
            'max_bin': 338,
            'random_strength': 8.553709762079974,
            'subsample': 0.7707144814535483,
            'max_leaves': 37,
            'verbose': False,
            'early_stopping_rounds': 50,
        }
    },
    # --------------------- LGB -------------------------
]



TRAINING_CFG = {
    'n_splits': 5,
    'random_state': 42,
    'shuffle': True,
    'use_test': True,
    'test_size': 0.15
}

if TRAINING_CFG['use_test']:
    from sklearn.model_selection import train_test_split
    
    train_df, test_df_1 = train_test_split(train_df, test_size=TRAINING_CFG['test_size'], shuffle=True, random_state=TRAINING_CFG['random_state'])


from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from catboost import CatBoostRegressor

from scipy.stats import rankdata
import lightgbm as lgb

def calculate_score(efs_time, preds, efs):
    return concordance_index(efs_time, preds, efs)

cols2drop = ['na_cumulative_hazard', 'km_survival_prob', 'target_for_cox', 'combined_risk', 'efs', 'efs_time',
             'weibull_survival', 'lognormal_survival', 'cloglog_survival', 'ID']

kf = KFold(n_splits=TRAINING_CFG['n_splits'], shuffle=TRAINING_CFG['shuffle'], random_state=TRAINING_CFG['random_state'])

all_trained_models = {i: list() for i in range(1, kf.get_n_splits()+1)}

for i, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    
    print(f"----- FOLD {i+1} out of {kf.get_n_splits()} ------")
    
    X = train_df.drop(columns=cols2drop)
        
    efs = train_df['efs']
    efs_time = train_df['efs_time']
    
    val_efs = efs.iloc[val_idx]
    val_efs_time = efs_time.iloc[val_idx]
    
    if TRAINING_CFG['use_test']:
        X_test = test_df_1.drop(columns=cols2drop)
        test_efs = test_df_1['efs']
        test_efs_time = test_df_1['efs_time']
    
    for model_cfg in MODELS_CFG:
        
        y = train_df[model_cfg['target']]
        X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        
        if 'catboost' in model_cfg['model_name']:
            reg = CatBoostRegressor(**model_cfg['params'])
            reg.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            val_preds = reg.predict(X_val)
            val_preds = -1 * val_preds if model_cfg['needs_minus'] else val_preds
            val_score = calculate_score(val_efs_time, val_preds, val_efs)
            
            if TRAINING_CFG['use_test']:
                test_preds = reg.predict(X_test)
                test_preds = -1 * test_preds if model_cfg['needs_minus'] else test_preds
                test_score = calculate_score(test_efs_time, test_preds, test_efs)
                
        elif 'lgb' in model_cfg['model_name']:
            reg = lgb.LGBMRegressor(**model_cfg['params'])
            reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

            val_preds = reg.predict(X_val)
            val_preds = -1 * val_preds if model_cfg['needs_minus'] else val_preds
            val_score = calculate_score(val_efs_time, val_preds, val_efs)

            if TRAINING_CFG['use_test']:
                test_preds = reg.predict(X_test)
                test_preds = -1 * test_preds if model_cfg['needs_minus'] else test_preds
                test_score = calculate_score(test_efs_time, test_preds, test_efs)
                
        s = f"Model: {model_cfg['model_name']}, val_score: {val_score}"
        if TRAINING_CFG['use_test']:
             print(s + f", test_score: {test_score}")
        else:
            print(s)
            
        all_trained_models[i+1].append([model_cfg, reg, val_score])
        
    print(f'Mean val score: {np.mean([x[-1] for x in all_trained_models[i+1]])}')


import numpy as np
from scipy.stats import rankdata

def predict_test(test_df, custom_weights=None, use_scaling=False):
    # Drop the specified columns to get the feature matrix for test data
    X_test = test_df.drop(columns=['ID'])
    
    # Collect all predictions and corresponding validation scores
    all_preds = []
    default_weights = []  # Collects validation scores unless custom_weights are provided
    
    # Iterate over each fold and each model within the fold
    for fold_models in all_trained_models.values():
        for model_info in fold_models:
            model_cfg, model, val_score = model_info
            # Generate predictions
            preds = model.predict(X_test)
            # Apply the 'needs_minus' transformation if required
            if model_cfg['needs_minus']:
                preds = -1 * preds
            all_preds.append(preds)
            default_weights.append(val_score)
    
    # Process each model's predictions (either rank or scale)
    processed_preds = []
    for preds in all_preds:
        if use_scaling:
            # Apply min-max scaling
            min_val = np.min(preds)
            max_val = np.max(preds)
            if max_val == min_val:
                # Handle case where all predictions are identical
                scaled = np.zeros_like(preds)
            else:
                scaled = (preds - min_val) / (max_val - min_val)
            processed_preds.append(scaled)
        else:
            # Convert predictions to ranks
            ranks = rankdata(preds)
            processed_preds.append(ranks)
    
    # Convert to numpy array for efficient computation
    processed_preds = np.array(processed_preds)
    
    # Determine which weights to use (custom or default)
    if custom_weights is not None:
        if len(custom_weights) != len(default_weights):
            raise ValueError(f"custom_weights length ({len(custom_weights)}) must match the number of models ({len(default_weights)})")
        weights = np.array(custom_weights)
    else:
        weights = np.array(default_weights)
    
    # Compute the weighted average of processed predictions
    ensemble_preds = np.average(processed_preds, axis=0, weights=weights)
    
    return ensemble_preds

#test_preds = predict_test(test_df, use_scaling=True)


from sklearn.linear_model import LinearRegression

cb_preds = pd.DataFrame()

for i, fold_models in tqdm(enumerate(all_trained_models.values(), 1)):
    for model_info in fold_models:
        model_cfg, model, val_score = model_info
        preds = model.predict(test_df_1.drop(columns=cols2drop))
        cb_preds[f'{model_cfg["model_name"]}_fold_{i}'] = preds
        
cb_preds['target'] = test_df_1['na_cumulative_hazard'].values




meta_reg = LinearRegression()
meta_reg.fit(cb_preds.drop(columns=['target']), cb_preds['target'])


cb_test_preds = pd.DataFrame()

for i, fold_models in tqdm(enumerate(all_trained_models.values(), 1)):
    for model_info in fold_models:
        model_cfg, model, val_score = model_info
        preds = model.predict(test_df.drop(columns='ID'))
        cb_test_preds[f'{model_cfg["model_name"]}_fold_{i}'] = preds
        
test_preds = meta_reg.predict(cb_test_preds)


sub_df = pd.DataFrame({
    'ID': test_df['ID'],
    'prediction': -test_preds
})

sub_df.to_csv('submission.csv', index=False)


sub_df




