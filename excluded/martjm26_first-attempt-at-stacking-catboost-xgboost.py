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
    train_engineered = train_engineered.merge(brand_stats, left_on='Brand', right_index=True, how='left')
    if test_engineered is not None:
        test_engineered = test_engineered.merge(brand_stats, left_on='Brand', right_index=True, how='left')
    
    # style statistics if target column exists
    if target_col in train_engineered.columns:
        style_stats = train_engineered.groupby('Style')[target_col].agg(['mean', 'median', 'std']).fillna(0)
        style_stats.columns = ['Style_Price_mean', 'Style_Price_median', 'Style_Price_std']
        
        train_engineered = train_engineered.merge(style_stats, left_on='Style', right_index=True, how='left')
        if test_engineered is not None:
            test_engineered = test_engineered.merge(style_stats, left_on='Style', right_index=True, how='left')
    
    # material statistics if target column exists
    if target_col in train_engineered.columns:
        material_stats = train_engineered.groupby('Material')[target_col].agg(['mean', 'median', 'std']).fillna(0)
        material_stats.columns = ['Material_Price_mean', 'Material_Price_median', 'Material_Price_std']
        
        train_engineered = train_engineered.merge(material_stats, left_on='Material', right_index=True, how='left')
        if test_engineered is not None:
            test_engineered = test_engineered.merge(material_stats, left_on='Material', right_index=True, how='left')
    
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


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import optuna
import xgboost as xgb

def prepare_data_for_catboost(df_engineered):
    """
    Prepare data for CatBoost, keeping categorical columns as is.
    """
    return df_engineered.copy().drop(["Price"], axis=1)

def optimize_catboost(X, y, categorical_features, n_trials=100):
    """
    Optimized CatBoost hyperparameter tuning focusing on key parameters
    and using more focused search spaces.
    """
    def objective(trial):
        # larger validation set to speed up training
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        train_pool = Pool(
            X_train, 
            y_train,
            cat_features=categorical_features
        )
        valid_pool = Pool(
            X_valid, 
            y_valid,
            cat_features=categorical_features
        )
        
        # base parameters 
        param = {
            'task_type': 'GPU',
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'bootstrap_type': 'Bernoulli',  # Fix bootstrap type to reduce search space
            'subsample': 0.8,               # Fixed reasonable value
            'random_strength': 1.0,         # Fixed reasonable value
            'max_bin': 254,                 # Speeds up training on GPU
            'min_data_in_leaf': 50,         # Fixed to prevent overfitting
            'grow_policy': 'Depthwise'      # Usually faster than Lossguide
        }
        
        # focus on most impactful parameters with narrower ranges since Optuna slow
        param.update({
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'depth': trial.suggest_int('depth', 5, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0)
        })
        
        model = CatBoostRegressor(**param)
        
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = model.predict(valid_pool)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        
        return rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    return study.best_params, study


def train_final_catboost(X_train, y_train, X_valid, y_valid, categorical_features, best_params):
    """
    Train final CatBoost model with best parameters.
    """
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
    
    final_params = {
        **best_params,
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42
    }
    
    model = CatBoostRegressor(**final_params)
    
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=200,
        verbose=100
    )
    
    # evaluate
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, valid_preds)),
        'train_r2': r2_score(y_train, train_preds),
        'valid_r2': r2_score(y_valid, valid_preds)
    }
    
    return model, metrics


def create_stacked_predictions(X, y, X_test, cat_model, modeling_features, n_folds=5):
    """
    Create stacked predictions using both CatBoost and XGBoost with optimized weights.
    """
    # arrays for predictions
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    
    # arrays to store individual model predictions for weight optimization
    oof_cat_predictions = np.zeros(len(X))
    oof_xgb_predictions = np.zeros(len(X))
    test_cat_predictions = np.zeros(len(X_test))
    test_xgb_predictions = np.zeros(len(X_test))

    # KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # numerical-only features for XGBoost
    X_xgb = X[modeling_features]
    X_test_xgb = X_test[modeling_features]
    
    # XGBoost params
    xgb_params = {
        'learning_rate': 0.018046432068412444, 
        'max_depth': 4, 
        'min_child_weight': 7, 
        'subsample': 0.8671407998973113, 
        'colsample_bytree': 0.9452379423473304, 
        'gamma': 0.0075721601186984795, 
        'lambda': 0.1216874945507346, 
        'alpha': 4.787701508000442e-07,
        'device': 'cuda',
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}")
        
        # split
        X_train_full, X_val_full = X.iloc[train_idx], X.iloc[val_idx]
        X_train_xgb = X_xgb.iloc[train_idx]
        X_val_xgb = X_xgb.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # CatBoost predictions
        cat_train_pred = cat_model.predict(X_train_full)
        cat_val_pred = cat_model.predict(X_val_full)
        
        # store CatBoost validation predictions
        oof_cat_predictions[val_idx] = cat_val_pred
        
        # train XGBoost on residuals
        dtrain = xgb.DMatrix(X_train_xgb, label=y_train - cat_train_pred)
        dval = xgb.DMatrix(X_val_xgb, label=y_val - cat_val_pred)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # XGBoost predictions
        xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val_xgb))
        
        # store XGBoost validation predictions
        oof_xgb_predictions[val_idx] = xgb_val_pred
        
        # test predictions for both models
        cat_test_pred = cat_model.predict(X_test)
        xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test_xgb))
        
        test_cat_predictions += cat_test_pred / n_folds
        test_xgb_predictions += xgb_test_pred / n_folds
    
    # optimize weights
    def find_optimal_weights(cat_preds, xgb_preds, true_values):
        def objective(weight):
            weighted_preds = cat_preds + weight[0] * xgb_preds
            return np.sqrt(mean_squared_error(true_values, weighted_preds))
        
        # weights from 0 to 1
        weights = np.linspace(0, 1, 101)
        best_rmse = float('inf')
        best_weight = 0
        
        for w in weights:
            rmse = objective([w])
            if rmse < best_rmse:
                best_rmse = rmse
                best_weight = w
        
        return best_weight, best_rmse
    
    # find optimal weight using out-of-fold predictions
    optimal_weight, best_rmse = find_optimal_weights(oof_cat_predictions, oof_xgb_predictions, y)
    print(f"\nOptimal XGBoost weight: {optimal_weight:.3f}")
    print(f"Best RMSE: {best_rmse:.3f}")
    
    # final predictions using optimal weight
    oof_predictions = oof_cat_predictions + optimal_weight * oof_xgb_predictions
    test_predictions = test_cat_predictions + optimal_weight * test_xgb_predictions
    
    return oof_predictions, test_predictions, optimal_weight, best_rmse


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
    'Capacity_Size_Interaction', 'Material_Capacity_Interaction',
    
    # brand aggregation features
    'Brand_Weight_mean', 'Brand_Weight_std',
    'Brand_Price_mean', 'Brand_Price_median', 'Brand_Price_std',
    
    # style aggregation features
    'Style_Price_mean', 'Style_Price_median', 'Style_Price_std',
    
    # material aggregation features
    'Material_Price_mean', 'Material_Price_median', 'Material_Price_std'
]

# categorical features for CatBoost
categorical_features = [
    'Brand', 'Material', 'Size', 'Laptop Compartment',
    'Waterproof', 'Style', 'Color', 'Brand_Style', 'Material_Size'
]

# prepare data
X = prepare_data_for_catboost(train_engineered)
y = train_engineered['Price']

#best_params, study = optimize_catboost(X, y, categorical_features, n_trials=100)


# OLD: [I 2025-02-20 17:08:59,580] Trial 40 finished with value: 38.87184073009811 and parameters: {'learning_rate': 0.06965987621996231, 'depth': 5, 'l2_leaf_reg': 4.984901537620981}.

# NEW: [I 2025-02-24 15:57:07,361] Trial 73 finished with value: 38.871310500227494 and parameters: {'learning_rate': 0.11147221370680077, 'depth': 5, 'l2_leaf_reg': 4.865071915855894}.


# hardcoded params after running Optuna to reduce runtime
cat_params = {
    'bootstrap_type': 'Bernoulli',  
    'subsample': 0.8,                        
    'grow_policy': 'SymmetricTree',
    'learning_rate': 0.11147221370680077,
    'depth': 5,
    'l2_leaf_reg': 4.984901537620981
}

# split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# train
cat_model, cat_metrics = train_final_catboost(
    X_train, y_train, X_valid, y_valid, categorical_features, cat_params
)

# bestTest = 38.87988859
# bestIteration = 360
# Shrink model to first 361 iterations.


print("Model performance:", cat_metrics)

# optimization history
#optuna.visualization.plot_optimization_history(study)


# need to run Optuna to do this
#optuna.visualization.plot_param_importances(study)


import matplotlib.pyplot as plt

# get feature importance
feature_importance = cat_model.get_feature_importance()
feature_names = X_train.columns

# plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()


# prep test data
def prepare_test_data_for_catboost(df_engineered):
    """
    Prepare test data for CatBoost, keeping categorical columns as is.
    """
    return df_engineered.copy()


X_test = prepare_test_data_for_catboost(test_engineered)
X_test.info()


# stacked predictions
oof_predictions, final_predictions, optimal_weight, best_rmse = create_stacked_predictions(
    X, y, X_test, cat_model, modeling_features
)


# evaluate stacked models
stack_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
stack_r2 = r2_score(y, oof_predictions)

print("\nStacked Model Performance:")
print(f"RMSE: {stack_rmse:.3f}")
print(f"R2 Score: {stack_r2:.4f}")

print("\nStacked Prediction Statistics:")
print(pd.Series(final_predictions).describe())


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sample_sub


# submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': final_predictions
})

submission


# save
submission.to_csv('predictions.csv', index=False)




