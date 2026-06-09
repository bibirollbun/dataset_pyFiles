import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tqdm import tqdm
from colorama import Fore, Style, init
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_df.drop('id', axis = 1, inplace = True)
test_df.drop('id', axis = 1, inplace = True)


train_df


def display_info(df):
    print(Style.BRIGHT+Fore.BLUE+f'\n Length of Train : {len(df)} \n' + Style.RESET_ALL)
    print(Style.BRIGHT+Fore.GREEN+f'\n Train head\n' + Style.RESET_ALL)
    display(train_df.head())
                   
    print(Style.BRIGHT+Fore.GREEN+f'\n Train info\n'+Style.RESET_ALL)               
    display(train_df.info())
    
    print(Style.BRIGHT+Fore.GREEN+f'\n Train missing values\n'+Style.RESET_ALL)               
    display(train_df.isna().sum())


display_info(train_df)


def get_val_counts(df, column_name, sort_by_column_name=False):
    value_count = df[column_name].value_counts().reset_index()
    value_count["Percentage"] = (value_count['count'] / value_count['count'].sum()) * 100
    value_count = value_count.reset_index(drop = True)
    if sort_by_column_name:
        value_count = value_count.sort_values(column_name)
    return value_count
    

def plot_value_counts_bar(df, column_name, ax=None, sort_by_column_name = False):
    val_count_df = get_val_counts(df, column_name, sort_by_column_name)
    
    if ax == None :
        f,ax = plt.subplots(figsize=(12,6))
    
    sns.barplot(data = val_count_df, y='count', x=column_name )

    for index, row in val_count_df.iterrows():
        count = row["count"]
        percentage = row["Percentage"]
        ax.text(
            x=index, 
            y=count + max(val_count_df["count"])*0.02,  # Adjust position slightly above the bar
            s=f'{count} ({percentage:.2f}%)', 
            ha='center', 
            va='bottom'
        )

def cat_feature_distribution(df):
    cat_fe = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # fig, axes = plt.subplots(nrows = len(cat_fe), ncols=1, figsize=(15, 2*len(cat_fe)))
    for i, fe in enumerate(cat_fe):
        plot_value_counts_bar(train_df, fe)



cat_feature_distribution(train_df)


SEED = 42
target = 'Price'
n_splits = 10
n_repeats = 1


X = train_df.drop(target, axis = 1)
y = train_df[target]


# making a pipeline for imputation of numerical and categorical columns
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


'''
Threshold for skewness : 

-0.5 to 0.5 => approx. symmetric
-1 to -0.5 or 0.5 to 1 => Moderetly skewed
<-1 or >1 => Highly skewed

'''

# For numerical columns we use mean if skewness is b/w -1 to 1 else we use median
# For categorical columns we will use most frequent

def data_imputation_pipeline(df : pd.DataFrame):

    # seperate numerical and categorical columns
    numerical_cols = df.select_dtypes(include = ["number"]).columns
    categorical_cols = df.select_dtypes(include = ["object", "category"]).columns

    # define cols to use mean and those on which to use median
    mean_numerical_cols = [col for col in numerical_cols if abs(train_df[col].skew()) <= 1]
    median_numerical_cols = [col for col in numerical_cols if abs(train_df[col].skew()) > 1]

    # define transformers for numerical and categorical data
    mean_numerical_transformer = SimpleImputer(strategy = "mean")
    median_numerical_transformer = SimpleImputer(strategy = "median")
    categorical_transformer = SimpleImputer(strategy = "most_frequent")

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num1", mean_numerical_transformer, mean_numerical_cols),
            ("num2", median_numerical_transformer, median_numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # create a pipeline
    pipeline = Pipeline(steps = [("preprocessor", preprocessor)])

    return pipeline, mean_numerical_cols, median_numerical_cols, categorical_cols

def update_df(train_df, X, test_df):
    
    pipeline, mean_cols, median_cols, cat_cols = data_imputation_pipeline(X)

    # Fit-transform the training data
    transformed_X = pipeline.fit_transform(X)
    transformed_test_df = pipeline.fit_transform(test_df)
    
    # Convert back to DataFrame with proper column names
    column_order = mean_cols + median_cols + list(cat_cols)
    
    X = pd.DataFrame(transformed_X, columns=column_order)
    test_df = pd.DataFrame(transformed_test_df, columns=column_order)
    
    # Restore original data types
    for col in mean_cols + median_cols:
        X[col] = pd.to_numeric(X[col])
        test_df[col] = pd.to_numeric(test_df[col])
    
    for col in cat_cols:
        X[col] = X[col].astype(train_df[col].dtype)
        test_df[col] = test_df[col].astype(train_df[col].dtype)
        
    # Convert object to category type
    X = X.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)
    test_df = test_df.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)

    
    return X, test_df


X, test_data = update_df(train_df, X, test_df)


import lightgbm as lgb
from lightgbm import early_stopping 
from tqdm import tqdm
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, train_test_split


# defining the error
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_function(X, y, test_data, model_name, n_splits, n_repeats, params = None, e_stop = 50):
    
    cat_c = list(X.select_dtypes(include = ["category"]).columns)
    kFold = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = SEED)

    train_rmsLe_scores = []
    val_rmsLe_scores = []
    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kFold.split(X, y), desc = "Training Folds", total = n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # y_train_log = np.log1p(y_train)
        # y_val_log = np.log1p(y_val)


        if model_name == "LGBM":
            callbacks = [lgb.early_stopping(stopping_rounds = e_stop, verbose = False)]
            model = lgb.LGBMRegressor(**params, random_state = SEED, verbose = -1, njobs = -1, device = 'cpu')
            model.fit(X_train, y_train, eval_set = [(X_val, y_val)],
                      eval_metric = 'rmse', callbacks = callbacks)

        elif model_name == "CAT":
            model = CatBoostRegressor(**params, random_state = SEED, verbose = 0)
            model.fit(X_train, y_train, 
                      eval_set = (X_val, y_val),
                      cat_features = cat_c,
                      early_stopping_rounds=100,
                      verbose = 0)
            
        elif model_name == "XGB":
            model = XGBRegressor(**params,random_state = SEED, objective= "reg:squarederror", verbosity = 0)
            model.fit(X_train, y_train,
                     eval_set = (X_val, y_val),
                     early_stopping_rounds = 100,
                     verbose = 0)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        test_pred = model.predict(test_data)

        # Exponentiate them to get predictions.
        # y_train_pred = np.expm1(y_train_log_pred)
        # y_val_pred = np.expm1(y_val_log_pred)
        # test_pred = np.expm1(test_log_pred)

        fold_test_preds.append(test_pred)

        train_rmsLe = rmse(y_train, y_train_pred)
        val_rmsLe = rmse(y_val, y_val_pred)

        train_rmsLe_scores.append(train_rmsLe)
        val_rmsLe_scores.append(val_rmsLe)

        print(f"Fold {fold+1} - Train RMSE: {train_rmsLe:.4f}, Validation RMSE: {val_rmsLe:.4f}")
    
    mean_test_preds = np.mean(fold_test_preds, axis=0) # mean of prediction from all folds...final prediction array
    mean_val_scores = f'{np.mean(val_rmsLe_scores):.4f}' # final validation score

    

    results_df = pd.DataFrame({
        "Fold" : np.arange(1, n_splits*n_repeats + 1),
        "Train rmse" : train_rmsLe_scores,
        "Validation rmse" : val_rmsLe_scores
    })

    print("\n--- Final Mean Scores ---")
    print(f"Mean Train RMSLE: {np.mean(train_rmsLe_scores):.4f}")
    print(f"Mean Validation RMSLE: {np.mean(val_rmsLe_scores):.4f}")

    print("\n=== KFold RMSLE Results ===")
    print(results_df)
    print("\n---------------------------------------------------------------------")
    
    return mean_test_preds, mean_val_scores


params1 = {
    'n_estimators' : 200
}

params2 = {
    'learning_rate': 0.08692991511139551, 'num_leaves': 85, 'max_depth': 15, 'min_data_in_leaf': 95,
     'feature_fraction': 0.7567559292276751, 'bagging_fraction': 0.9472874885021447, 'bagging_freq': 1,
     'max_bin': 305, 'min_child_weight': 1, 'scale_pos_weight': 4,'n_estimators':200
}


mp1 = train_function(
    X = X,
    y = y,
    test_data = test_data,
    model_name = "CAT", 
    n_splits = 10, 
    n_repeats = 1, 
    params = params1, 
    e_stop = 50
)


sample[target] = mp1[0]

sample.to_csv("submission.csv", index = False)
sample.head()

