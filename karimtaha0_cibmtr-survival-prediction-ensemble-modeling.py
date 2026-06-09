# Install required packages
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# Suppress warnings
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import polars as pl
import pandas as pd

# Visualization
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import rankdata, uniform, randint
from lifelines.utils import concordance_index
from sklearn.experimental import enable_iterative_imputer  # Enables IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# Settings
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None


cfg = {
    'train_path': Path('/kaggle/input/equity-post-HCT-survival-predictions/train.csv'),
    'test_path': Path('/kaggle/input/equity-post-HCT-survival-predictions/test.csv'),
    'subm_path': Path('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv'),
    'colorscale': 'Sunset',
    'color': '#EADDCA',
    'batch_size': 32768,
    'early_stop': 300,
    'penalizer': 0.01,
    'n_splits': 5,
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 10,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'num_iterations': 6000,
        'early_stopping_rounds': 100,
        'seed': 42
    },
    'ctb_params': {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.85,
        'reg_lambda': 8.0,
        'depth': 8
    },
    'cox_params': [
        {   # Cox1
            'grow_policy': 'Depthwise',
            'min_child_samples': 8,
            'loss_function': 'Cox',
            'learning_rate': 0.03,
            'random_state': 42,
            'task_type': 'CPU',
            'num_trees': 6000,
            'subsample': 0.6,
            'reg_lambda': 8.0,
            'depth': 8,
        },
        {   # Cox2
            'grow_policy': 'Lossguide',
            'loss_function': 'Cox',
            'learning_rate': 0.03,
            'random_state': 42,
            'task_type': 'CPU',
            'num_trees': 6000,
            'subsample': 0.6,
            'reg_lambda': 8.0,
            'num_leaves': 32,
            'depth': 8,
        },
        {   # Cox3
            'grow_policy': 'Depthwise',
            'min_child_samples': 16,
            'loss_function': 'Cox',
            'learning_rate': 0.02,
            'random_state': 42,
            'task_type': 'CPU',
            'num_trees': 7000,
            'subsample': 0.5,
            'reg_lambda': 6.0,
            'depth': 10,
        }
    ],
    'lgb_ensemble_configs': [
        {   # GBDT configuration
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'random_state': 42
        },
        {   # DART configuration
            'boosting_type': 'dart',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 15,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_estimators': 1000,
            'max_drop': 50,
            'random_state': 42
        },
        {   # GOSS configuration
            'boosting_type': 'goss',
            'learning_rate': 0.1,
            'num_leaves': 127,
            'max_depth': 10,
            'subsample': 1.0,
            'colsample_bytree': 0.9,
            'n_estimators': 1000,
            'top_rate': 0.2,
            'other_rate': 0.1,
            'random_state': 42
        }
    ],
    'lgb_param_space': {
        'num_leaves': randint(15, 128),
        'max_depth': randint(5, 30),
        'learning_rate': uniform(0.01, 0.19),
        'feature_fraction': uniform(0.6, 0.4),
        'bagging_fraction': uniform(0.6, 0.4),
        'bagging_freq': randint(1, 7),
        'min_child_samples': randint(5, 50),
        'min_child_weight': uniform(1e-5, 1.0),
        'reg_alpha': uniform(0.0, 1.0),
        'reg_lambda': uniform(0.0, 1.0),
    }
}


def load_data(path, batch_size):
    """Load data using polars with batch processing."""
    df = pl.read_csv(path, batch_size=batch_size)
    return df


# Load the data
train_df = load_data(cfg['train_path'], cfg['batch_size'])
test_df = load_data(cfg['test_path'], cfg['batch_size'])
print("Data loaded successfully.")


train_df.head()


def remove_high_missing_cols(df, threshold=0.6):
    """Remove columns with more than threshold percentage of missing values."""
    df_pd = df.to_pandas()
    missing_percent = df_pd.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    df_pd.drop(columns=cols_to_drop, inplace=True)
    print(f"Columns dropped due to high missing values: {cols_to_drop}")
    return pl.from_pandas(df_pd)


# Remove columns with more than 60% missing values
#train_df = remove_high_missing_cols(train_df, threshold=0.6)
#test_df = remove_high_missing_cols(test_df, threshold=0.6)


def cast_datatypes(df):
    """Cast columns to appropriate data types without filling missing values."""
    from polars import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8

    numeric_types = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64]

    for col, dtype in zip(df.columns, df.dtypes):
        if col == 'ID':
            df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif dtype in numeric_types:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        else:
            # First cast to Utf8
            df = df.with_columns(pl.col(col).cast(pl.Utf8))
            # Then cast to Categorical
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df


# Cast data types
train_df = cast_datatypes(train_df)
test_df = cast_datatypes(test_df)
print("Data types cast successfully.")


def add_features(df):
    """Create advanced feature interactions and transformations."""
    # Interactions
    df = df.with_columns([
        (pl.col('age_at_hct') * pl.col('karnofsky_score')).alias('age_karnofsky'),
        (pl.col('age_at_hct') * pl.col('comorbidity_score')).alias('age_comorbidity'),
        (pl.col('donor_age') - pl.col('age_at_hct')).abs().alias('donor_recipient_age_diff')
    ])

    # Time-based features
    df = df.with_columns([
        (pl.col('year_hct') - 2000).alias('years_since_2000')
    ])

    # HLA match ratios
    df = df.with_columns([
        ((pl.col('hla_high_res_8') + pl.col('hla_low_res_8')) / 16).alias('hla_match_ratio')
    ])

    # Polynomial features
    df = df.with_columns([
        (pl.col('age_at_hct') ** 2).alias('age_squared'),
        (pl.col('karnofsky_score') ** 2).alias('karnofsky_squared')
    ])

    # Risk stratification
    df = df.with_columns([
        ((pl.col('age_comorbidity') + pl.col('age_karnofsky')) / 2).alias('combined_risk_score')
    ])

    # Additional medical score combinations
    df = df.with_columns([
        (pl.col('comorbidity_score') * pl.col('karnofsky_score')).alias('comorbidity_karnofsky'),
        (pl.col('hla_match_ratio') * pl.col('karnofsky_score')).alias('match_karnofsky')
    ])

    return df


# Add features
train_df = add_features(train_df)
test_df = add_features(test_df)
print("Features added successfully.")


def impute_missing_values(df):
    """Impute missing values using IterativeImputer."""
    df_pd = df.to_pandas()

    # Separate numerical and categorical columns
    num_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_pd.select_dtypes(include=['category', 'object']).columns.tolist()

    # For categorical columns, use Mode imputation (most frequent)
    for col in cat_cols:
        df_pd[col] = df_pd[col].fillna(df_pd[col].mode()[0])
        df_pd[col] = df_pd[col].astype('category')

    # For numerical columns, use IterativeImputer
    imp_num = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        max_iter=10,
        random_state=0
    )
    df_pd[num_cols] = imp_num.fit_transform(df_pd[num_cols])

    # Convert back to Polars DataFrame
    df = pl.from_pandas(df_pd)
    return df


# Impute missing values
train_df = impute_missing_values(train_df)
test_df = impute_missing_values(test_df)
print("Missing values imputed using IterativeImputer.")


def encode_categorical(df, encoder, train=True):
    """Encode categorical variables, excluding 'race_group'."""
    # Exclude 'race_group' from encoding
    cat_cols = [col for col in df.columns if df[col].dtype == pl.Categorical and col != 'race_group']
    
    df_pd = df.to_pandas()
    
    # Keep 'race_group' as a separate column
    race_group = df_pd['race_group']
    
    # All other categorical columns are already of 'category' dtype
    if train:
        encoded = encoder.fit_transform(df_pd[cat_cols])
    else:
        encoded = encoder.transform(df_pd[cat_cols])
    
    cat_encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df_pd.index
    )
    
    # Combine the data, keeping 'race_group' intact
    df_encoded = pd.concat([df_pd.drop(columns=cat_cols + ['race_group']), cat_encoded_df, race_group], axis=1)
    
    # Convert back to Polars DataFrame
    df = pl.from_pandas(df_encoded)
    return df


# Initialize the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Encode categorical variables
train_df = encode_categorical(train_df, encoder, train=True)
test_df = encode_categorical(test_df, encoder, train=False)
print("Categorical encoding completed.")


# Convert Polars DataFrames to Pandas DataFrames
train_df_pd = train_df.to_pandas()
test_df_pd = test_df.to_pandas()

# Identify categorical columns
cat_cols = [col for col in train_df_pd.columns if train_df_pd[col].dtype.name == 'category']
print(f"Categorical columns: {cat_cols}")


def normalize_features(df, scaler, train=True):
    """Normalize numerical features."""
    df_pd = df.to_pandas()
    num_cols = [col for col in df_pd.columns 
                if pd.api.types.is_numeric_dtype(df_pd[col])
                and col not in ['ID', 'efs', 'efs_time']]

    if train:
        df_pd[num_cols] = scaler.fit_transform(df_pd[num_cols])
    else:
        df_pd[num_cols] = scaler.transform(df_pd[num_cols])

    return pl.from_pandas(df_pd)


# Initialize the scaler
scaler = StandardScaler()

# Normalize features
train_df = normalize_features(train_df, scaler, train=True)
test_df = normalize_features(test_df, scaler, train=False)
print("Feature normalization completed.")


def clean_feature_names(X):
    """Clean feature names to be compatible with LightGBM."""
    clean_columns = {col: col.replace(' ', '_').replace('-', '_').replace('/', '_')
                     .replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                     .replace('{', '').replace('}', '').replace(':', '_').replace(';', '_')
                     .replace(',', '_').replace('.', '_').replace('&', 'and')
                     .replace('%', 'pct').replace('#', 'num').replace('*', 'star')
                     .replace('@', 'at').replace('!', '').replace('?', '')
                     .replace('=', 'eq').replace('+', 'plus').replace('>', 'gt')
                     .replace('<', 'lt').replace('|', '_')
                     for col in X.columns}

    seen_names = set()
    for old_name, new_name in clean_columns.items():
        if new_name in seen_names:
            i = 1
            while f"{new_name}_{i}" in seen_names:
                i += 1
            clean_columns[old_name] = f"{new_name}_{i}"
        seen_names.add(clean_columns[old_name])

    X.rename(columns=clean_columns, inplace=True)
    return X

def prepare_features(data, cat_cols, model_type='lgb'):
    """Prepare features for specific model type."""
    X = data.copy()
    
    if model_type == 'lgb':
        # Clean feature names
        X = clean_feature_names(X)
        
        # Ensure categorical columns have the correct data types
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
    else:  # CatBoost
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        # Ensure numerical columns are correctly typed
        for col in X.columns:
            if col not in cat_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
    
    return X


def prepare_data_for_cox(data):
    """Prepare data for Cox model by handling categorical variables."""
    cox_data = data.copy()

    # Identify categorical columns (excluding encoded ones)
    categorical_cols = [col for col in cox_data.columns 
                        if col not in ['ID', 'efs', 'efs_time'] 
                        and not col.startswith('enc_')
                        and not pd.api.types.is_numeric_dtype(cox_data[col])]

    # One-hot encode categorical columns
    cox_data = pd.get_dummies(cox_data, columns=categorical_cols, drop_first=True)

    # Ensure all columns are numeric
    for col in cox_data.columns:
        if col not in ['ID', 'efs', 'efs_time']:
            cox_data[col] = pd.to_numeric(cox_data[col], errors='coerce')

    return cox_data

def create_targets(data, penalizer):
    """Create multiple target variables for different modeling approaches."""
    print("Creating target variables...")

    # Create a copy of the data
    data = data.copy()

    try:
        # Cox target
        print("Creating Cox target...")
        cox_data = prepare_data_for_cox(data)
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_data, duration_col='efs_time', event_col='efs')
        data['target1'] = cph.predict_partial_hazard(cox_data)
    except Exception as e:
        print(f"Warning: Error in Cox model fitting: {e}")
        print("Using fallback target calculation...")
        data['target1'] = data['efs_time'] * data['efs']

    # Kaplan-Meier target
    print("Creating Kaplan-Meier target...")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data['efs_time'], event_observed=data['efs'])
    data['target2'] = kmf.survival_function_at_times(data['efs_time']).values

    # Nelson-Aalen target
    print("Creating Nelson-Aalen target...")
    naf = NelsonAalenFitter()
    naf.fit(durations=data['efs_time'], event_observed=data['efs'])
    data['target3'] = -naf.cumulative_hazard_at_times(data['efs_time']).values

    # Cox loss target
    print("Creating Cox loss target...")
    data['target4'] = data['efs_time'].copy()
    data.loc[data['efs'] == 0, 'target4'] *= -1

    print("Target creation completed.")
    return data


# Since we need to create targets only for the training data
train_df_pd = train_df.to_pandas()
train_df_pd = create_targets(train_df_pd, cfg['penalizer'])
print("Target variables created.")


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculate the competition metric (Stratified Concordance Index).
    """
    # Remove ID column
    solution = solution.copy()
    submission = submission.copy()
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Merge solution and submission
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True, drop=True)

    # Calculate C-index for each race group
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []

    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]

        # Calculate concordance index
        c_index_race = concordance_index(
            merged_df_race['efs_time'],
            -merged_df_race['prediction'],
            merged_df_race['efs']
        )
        metric_list.append(c_index_race)

    # Return mean - std of C-indices
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))


def calculate_fold_score(data, valid_idx, oof_preds):
    """Calculate score for a single fold."""
    y_true = data.iloc[valid_idx][['ID', 'efs', 'efs_time', 'race_group']].copy()
    y_pred = pd.DataFrame({
        'ID': data.iloc[valid_idx]['ID'],
        'prediction': oof_preds[valid_idx]
    })
    return score(y_true, y_pred, 'ID')

def train_model(data, cat_cols, params, target, model_type, n_splits, early_stop):
    print(f"\nTraining {model_type} model for {target}...")
    feature_cols = [col for col in data.columns 
                    if col not in ['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4']]
    X = data[feature_cols].copy()
    y = data[target].copy()

    X = prepare_features(X, cat_cols, model_type)
    print(f"Number of features: {X.shape[1]}")

    models = []
    oof_preds = np.zeros(len(X))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X)):
        print(f"Training fold {fold + 1}/{n_splits}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(early_stop)]
            )
        else:
            model = CatBoostRegressor(**params, cat_features=cat_cols)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=early_stop,
                verbose=False
            )

        models.append(model)
        oof_preds[valid_idx] = model.predict(X_valid)

        # Optionally calculate and print fold score
        fold_score = calculate_fold_score(data, valid_idx, oof_preds)
        print(f"Fold {fold + 1} Score: {fold_score:.4f}")

    return models, oof_preds


def predict(models, test_data, cat_cols, model_type='lgb'):
    """Make predictions using trained models."""
    feature_cols = [col for col in test_data.columns 
                    if col not in ['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4']]
    X_test = test_data[feature_cols].copy()

    X_test = prepare_features(X_test, cat_cols, model_type)

    preds = np.mean([model.predict(X_test) for model in models], axis=0)
    return preds


# Load original training data to check columns
original_train_df = pd.read_csv(cfg['train_path'])
print("Columns in original training data:")
print(original_train_df.columns.tolist())


print("Columns in train_df_pd after preprocessing:")
print(train_df_pd.columns.tolist())


# Convert Polars DataFrames to Pandas DataFrames
train_df_pd = train_df.to_pandas()
test_df_pd = test_df.to_pandas()

# Identify categorical columns
cat_cols = [col for col in train_df_pd.columns if train_df_pd[col].dtype.name == 'category']
print(f"Categorical columns: {cat_cols}")

# Proceed to create target variables for the training data
train_df_pd = create_targets(train_df_pd, cfg['penalizer'])

# Now you can use `cat_cols` in your model training
all_models = []
all_preds = []

# Training LightGBM Models
for target in ['target1', 'target2', 'target3']:
    models, _ = train_model(
        train_df_pd, cat_cols, cfg['lgb_params'], target, 'lgb',
        cfg['n_splits'], cfg['early_stop'])
    all_models.extend(models)
    preds = predict(models, test_df_pd, cat_cols, 'lgb')
    all_preds.append(preds)


# Training CatBoost Models
for target in ['target1', 'target2', 'target3']:
    models, _ = train_model(
        train_df_pd, cat_cols, cfg['ctb_params'], target, 'ctb',
        cfg['n_splits'], cfg['early_stop'])
    all_models.extend(models)
    preds = predict(models, test_df.to_pandas(), cat_cols, 'ctb')
    all_preds.append(preds)


# Training Cox Models
for params in cfg['cox_params']:
    models, _ = train_model(
        train_df_pd, cat_cols, params, 'target4', 'ctb',
        cfg['n_splits'], cfg['early_stop'])
    all_models.extend(models)
    preds = predict(models, test_df.to_pandas(), cat_cols, 'ctb')
    all_preds.append(preds)


# Generate ensemble predictions
final_predictions = np.mean(all_preds, axis=0)


# Create submission
submission = pd.DataFrame({
    'ID': test_df.to_pandas()['ID'],
    'prediction': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")


display(submission.head())


import random

# --- Visualization 1: Feature Importance from a LightGBM Model ---

# Identify LightGBM models from the ensemble (assuming LGBMRegressor is used)
lgb_models = [m for m in all_models if m.__class__.__name__ == 'LGBMRegressor']
if lgb_models:
    # Use the first LightGBM model for visualization
    lgb_model = lgb_models[0]
    
    # Prepare features for visualization: use the same features as used in training.
    # Exclude columns not used for training.
    X_train_features = train_df_pd.drop(columns=['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'])
    # Use your helper to prepare features (for LightGBM, as in your training loop)
    X_prepared = prepare_features(X_train_features.copy(), cat_cols, model_type='lgb')
    
    # Extract feature importances and select top 20 features
    importances = lgb_model.feature_importances_
    indices = np.argsort(importances)[-20:]
    top_features = [X_prepared.columns[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.show()
else:
    print("No LightGBM model found in the ensemble.")

# --- Visualization 2: Actual vs Predicted Scatter Plot ---

# For demonstration, sample 100 random rows from your training data.
# We'll use target1 as an example target.
sample_indices = random.sample(range(len(train_df_pd)), 100)
X_sample = train_df_pd.iloc[sample_indices].drop(columns=['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'])
# Prepare these features similar to training
X_sample_prepared = prepare_features(X_sample.copy(), cat_cols, model_type='lgb')
y_true_sample = train_df_pd.iloc[sample_indices]['target1']

# Use the same LightGBM model for prediction (if available)
if lgb_models:
    y_pred_sample = lgb_model.predict(X_sample_prepared)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true_sample, y=y_pred_sample, alpha=0.7)
    plt.plot([y_true_sample.min(), y_true_sample.max()], [y_true_sample.min(), y_true_sample.max()], color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Sample of 100)")
    plt.tight_layout()
    plt.show()
else:
    print("No LightGBM model available for prediction.")




