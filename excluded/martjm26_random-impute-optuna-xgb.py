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


# load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_df = pd.concat([train_df, train_extra_df], axis=0, ignore_index=True)


# initial overview -- check dtypes, shape, number of NaNs, and range of values/weird min or max values
display(train_df.head(15).T, train_df.info(), train_df.describe())


# deal with numeric columns' missing values
numeric = train_df.drop(columns=["id", "Price"]).select_dtypes(include=['number']).columns
train_df[numeric] = train_df[numeric].fillna(train_df[numeric].mean())
test_df[numeric] = test_df[numeric].fillna(test_df[numeric].mean())
display(train_df.isnull().sum(), test_df.isnull().sum())


# view the distribution of values in each object column
import seaborn as sns
import matplotlib.pyplot as plt
categorical = train_df.select_dtypes(include='object').columns
plt.figure(figsize=(15, len(categorical) * 5))
for c in categorical:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=c, data=train_df, order=train_df[c].value_counts().index)
    plt.title(f'Distribution of Values in {c}')
    plt.xlabel(c)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# deal with object columns' missing values, using Random Imputer for more accurate imputation
# just using something such as the mode would ruin the balance already present
from sklearn.preprocessing import LabelEncoder

def process_data_with_distribution_imputer(train_df, test_df, cat_columns):
    """
    Process categorical data by:
    1. Encoding non-null values
    2. Imputing missing values based on observed distributions
    """
    # encoders
    label_encoders = {col: LabelEncoder() for col in cat_columns}
    
    # copies for processing
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    for col in cat_columns:
        # fit encoder on non-null training values
        label_encoders[col].fit(train_df[col].dropna())
        
        # value distribution from training data
        train_dist = train_df[col].value_counts(normalize=True)
        
        # process training data
        # encode non-null values
        train_processed.loc[train_df[col].notna(), col] = label_encoders[col].transform(
            train_df[col][train_df[col].notna()]
        )
        
        # impute missing values based on distribution
        missing_train = train_df[col].isna()
        if missing_train.any():
            train_processed.loc[missing_train, col] = np.random.choice(
                label_encoders[col].transform(train_dist.index),
                size=missing_train.sum(),
                p=train_dist.values
            )
            
        # process test data
        # encode non-null values
        test_processed.loc[test_df[col].notna(), col] = label_encoders[col].transform(
            test_df[col][test_df[col].notna()]
        )
        
        # impute missing values based on training distribution
        missing_test = test_df[col].isna()
        if missing_test.any():
            test_processed.loc[missing_test, col] = np.random.choice(
                label_encoders[col].transform(train_dist.index),
                size=missing_test.sum(),
                p=train_dist.values
            )
    
    return train_processed, test_processed, label_encoders


display(train_df.head(15).T, test_df.head(15).T)


# run imputing function to properly handle NaNs in categorical features
cat_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
               'Waterproof', 'Style', 'Color']

train_processed, test_processed, label_encoders = process_data_with_distribution_imputer(
    train_df=train_df,
    test_df=test_df,
    cat_columns=cat_columns
)

# make sure relative balance of unique values was preserved
for c in categorical:
    print(f"\nColumn: {c}")
    print("Original value counts:")
    print(train_df[c].value_counts(normalize=True).head())
    print("\nImputed value counts:")
    print(train_processed[c].value_counts(normalize=True).head())


# convert back to strings
for col in cat_columns:
    train_processed[col] = label_encoders[col].inverse_transform(train_processed[col].astype(int))
    test_processed[col] = label_encoders[col].inverse_transform(test_processed[col].astype(int))


display(train_processed.head(15).T, test_processed.head(15).T)


from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

def fit_engineer_backpack_features(train_df, test_df=None, target_col='Price', random_state=42):
    """
    Engineer features for price prediction.
    If test_df is provided, applies the same transformations learned from train_df.
    """
    # copy to avoid modifying original
    train_engineered = train_df.copy()
    test_engineered = test_df.copy() if test_df is not None else None
    encoders = {}
    
    # categorical combinations
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df['Material_Size'] = df['Material'] + '_' + df['Size']
            df['Brand_Style'] = df['Brand'] + '_' + df['Style']
    
    # binary features
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df['Has_Laptop_Compartment'] = (df['Laptop Compartment'] == 'Yes').astype(int)
            df['Is_Waterproof'] = (df['Waterproof'] == 'Yes').astype(int)
            df['Capacity_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
    
    # size-based groupings
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df['Size_Numeric'] = df['Size'].map(size_map)
    
    # material quality tiers
    material_tiers = {
        'Canvas': 1,
        'Polyester': 2,
        'Nylon': 3,
        'Leather': 4
    }
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df['Material_Tier'] = df['Material'].map(material_tiers)
    
    # brand-based statistics
    weight_stats = train_engineered.groupby('Brand')['Weight Capacity (kg)'].agg(['mean', 'std']).fillna(0)
    weight_stats.columns = ['Brand_Weight_mean', 'Brand_Weight_std']
    
    if target_col in train_engineered.columns:
        price_stats = train_engineered.groupby('Brand')[target_col].agg(['mean', 'median', 'std']).fillna(0)
        price_stats.columns = ['Brand_Price_mean', 'Brand_Price_median', 'Brand_Price_std']
        brand_stats = pd.concat([weight_stats, price_stats], axis=1)
    else:
        brand_stats = weight_stats
    
    # merge stats to both dataframes
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df = df.merge(brand_stats, left_on='Brand', right_index=True, how='left')
    
    # style statistics if target column exists
    if target_col in train_engineered.columns:
        style_stats = train_engineered.groupby('Style')[target_col].agg(['mean', 'median', 'std']).fillna(0)
        style_stats.columns = ['Style_Price_mean', 'Style_Price_median', 'Style_Price_std']
        
        for df in [train_engineered, test_engineered]:
            if df is not None:
                df = df.merge(style_stats, left_on='Style', right_index=True, how='left')
    
    # material statistics if target column exists
    if target_col in train_engineered.columns:
        material_stats = train_engineered.groupby('Material')[target_col].agg(['mean', 'median', 'std']).fillna(0)
        material_stats.columns = ['Material_Price_mean', 'Material_Price_median', 'Material_Price_std']
        
        for df in [train_engineered, test_engineered]:
            if df is not None:
                df = df.merge(material_stats, left_on='Material', right_index=True, how='left')
    
    # handle categorical encoding
    # target encoding for high-cardinality features if target exists
    target_encode_cols = ['Brand', 'Material', 'Color', 'Brand_Style', 'Material_Size']
    
    if target_col in train_engineered.columns:
        for c in target_encode_cols:
            encoders[f'{c}_target'] = TargetEncoder(
                cols=[c],
                min_samples_leaf=1,
                smoothing=10,
                handle_missing='value',
                handle_unknown='value',
            )
            # fit on train
            encoders[f'{c}_target'].fit(train_engineered[[c]], train_engineered[target_col])
            
            # transform train and test
            train_engineered[f'{c}_encoded'] = encoders[f'{c}_target'].transform(train_engineered[[c]])
            if test_engineered is not None:
                test_engineered[f'{c}_encoded'] = encoders[f'{c}_target'].transform(test_engineered[[c]])
    
    # label encoding for low-cardinality features
    label_encode_cols = ['Size', 'Style']
    
    for c in label_encode_cols:
        encoders[f'{c}_label'] = LabelEncoder()
        # fit on train
        encoders[f'{c}_label'].fit(train_engineered[c])
        
        # transform train and test
        train_engineered[f'{c}_encoded'] = encoders[f'{c}_label'].transform(train_engineered[c])
        if test_engineered is not None:
            test_engineered[f'{c}_encoded'] = encoders[f'{c}_label'].transform(test_engineered[c])
    
    # interaction features
    for df in [train_engineered, test_engineered]:
        if df is not None:
            df['Capacity_Size_Interaction'] = df['Weight Capacity (kg)'] * df['Size_Numeric']
            df['Material_Capacity_Interaction'] = df['Material_Tier'] * df['Weight Capacity (kg)']
    
    if test_engineered is not None:
        return train_engineered, test_engineered, encoders
    else:
        return train_engineered, encoders


# create new features on train and test dataframes
train_engineered, test_engineered, encoders = fit_engineer_backpack_features(
    train_df=train_processed,
    test_df=test_processed,
    target_col='Price'
)


display(train_engineered.info(), test_engineered.info())


# make sure data is in the right format
# use Optuna to find optimal hyperparams
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def prepare_data_for_xgb(df_engineered, modeling_features):
    """
    Prepare data for XGBoost by ensuring all features are numeric.
    """
    # copy to avoid modifying the original
    df_processed = df_engineered[modeling_features].copy()
    
    # check for object columns
    object_columns = df_processed.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        raise ValueError(f"Found object columns that need to be encoded: {object_columns}")
    
    return df_processed

def objective(trial, X_train, X_valid, y_train, y_valid):
    """
    Optuna objective function for XGBoost optimization.
    """
    param = {
        'device': 'cuda',  # use GPU
        'tree_method': 'hist',  # GPU accelerated histogram algorithm
        
        # parameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        
        # fixed parameters
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42
    }
    
    # DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # train with early stopping
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # predict and evaluate
    preds = model.predict(dvalid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse


def optimize_xgb(X, y, n_trials=100):
    """
    Optimize XGBoost hyperparameters using Optuna.
    """
    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # create study object
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # optimize
    study.optimize(
        lambda trial: objective(trial, X_train, X_valid, y_train, y_valid),
        n_trials=n_trials,
        n_jobs=1  # 1 for GPU to avoid conflicts
    )
    
    return study.best_params, study


def train_final_model(X_train, y_train, X_valid, y_valid, best_params):
    """
    Train final XGBoost model with best parameters.
    """
    # add fixed parameters
    final_params = {
        **best_params,
        'device': 'cuda',
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42
    }
    
    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # train final model
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=100
    )
    
    # evaluate
    train_preds = model.predict(dtrain)
    valid_preds = model.predict(dvalid)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, valid_preds)),
        'train_r2': r2_score(y_train, train_preds),
        'valid_r2': r2_score(y_valid, valid_preds)
    }
    
    return model, metrics


modeling_features = [
    # numeric features
    'Compartments', 'Weight Capacity (kg)',
    'Capacity_per_Compartment', 'Size_Numeric', 'Material_Tier',
    
    # binary features
    'Has_Laptop_Compartment', 'Is_Waterproof',
    
    # target encoded features
    'Brand_encoded', 'Material_encoded', 'Color_encoded',
    'Brand_Style_encoded', 'Material_Size_encoded',
    
    # label encoded features
    'Size_encoded', 'Style_encoded',
    
    # interaction features
    'Capacity_Size_Interaction', 'Material_Capacity_Interaction'
]

# prepare data for XGBoost
X = prepare_data_for_xgb(train_engineered, modeling_features)
y = train_engineered['Price']

# optimize hyperparameters
#best_params, study = optimize_xgb(X, y, n_trials=100)


# best parameters found by Optuna
# hardcoded now becuase Optuna took a very long time to run trials
best_params = {
    'learning_rate': 0.018046432068412444, 
    'max_depth': 4, 
    'min_child_weight': 7, 
    'subsample': 0.8671407998973113, 
    'colsample_bytree': 0.9452379423473304, 
    'gamma': 0.0075721601186984795, 
    'lambda': 0.1216874945507346, 
    'alpha': 4.787701508000442e-07
}

# train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# train final model
model, metrics = train_final_model(X_train, y_train, X_valid, y_valid, best_params)


print("Model performance:", metrics)

# optimization history
#optuna.visualization.plot_optimization_history(study)


# need to run Optuna to do this
#optuna.visualization.plot_param_importances(study)


from xgboost import plot_importance
plot_importance(model)
plt.title('Feature Importance')
plt.show()


def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    """
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    return predictions


# predictions
predictions = make_predictions(model, test_engineered[modeling_features])
print("\nPrediction Statistics:")
print(pd.Series(predictions).describe())


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sample_sub


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': predictions
})

submission


# Save predictions
submission.to_csv('predictions.csv', index=False)




