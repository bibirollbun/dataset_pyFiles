#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# Boosting
import lightgbm as lgb
import xgboost as xgb


# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.base import *

# Setting
pd.set_option('max_colwidth',None)
seed = 855
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


#
# Data
#

# dataset
train = pd.read_csv(data_path[2])
test = pd.read_csv(data_path[1])
sub = pd.read_csv(data_path[0])


# view
print(f"Training shape: {train.shape} | Testing shape: {test.shape}\n")
train.head()


#
# Custom Function -- Imputation
#

def impute(df):
    """
    Impute numerical columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with imputed numerical columns.
    """
    
    impute_cols = []
    
    # find & extract columns with NaN
    res = df.isna().sum()

    for i, j in zip(res,list(df.columns)):
        if i > 0:  # no. of rows with NaN > 0
            impute_cols.append(j)
        else:
            pass

    print(f"Found \" {', '.join(impute_cols)} \" columns in the dataset \n")
    
    # loop through each column & impute based on the column type 
    for c in impute_cols:
        # check if its numerical
        if df[c].dtype == int or df[c].dtype == float:
            qnt = df[c].quantile(0.75)   # 75% quantile
            df[[c]] = df[[c]].fillna(value=qnt,axis=1)
        # check if its object
        else:
            df[c].fillna(df[c].mode()[0], inplace=True)
       
    print(f"** Imputation Completed **\n")
    return df


#
# Custom Function -- Extract "New" Features
#

def new_features(df):
    """
    Extract new features

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new feature columns
    """
    
    # when the condition is true
    choicelist = [1]

    # convert 'sale_date' to datetime
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # get sale year & month from sale_date
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month

    # get age
    df['age'] = df['sale_year'] - df['year_built']

    # renovated
    df['renovated'] = np.select([df['year_reno'] > 0], choicelist, default=0)
    
    # year(s) since renovation
    df['years_since_reno'] = np.where(df['renovated'], df['sale_year'] - df['year_reno'], 0)

    # total bathrooms
    df['total_baths'] = df['bath_full'] + 0.75 * df['bath_3qtr'] + 0.5 * df['bath_half']

    # total valuation
    df['total_value'] = df['land_val'] + df['imp_val']

    # living area
    df['living_area'] = df['sqft'] + df['sqft_fbsmt']
    
    return df



#
# Custom Function -- Encoding
#

def encode(df, encoding='ordinal'):
    """
    Encode categorical columns

    Args:
        df (pd.DataFrame): The input DataFrame.
        encoder (string): ordinal / label

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns
    """
    # get columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if encoding == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    elif encoding == 'label':
        encoder = LabelEncoder()
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    else:
        raise ValueError("Encoding must be 'ordinal' or 'label'")

    return df


#
# Custom Function - Winkler Scorer
#

def winkler_score(y_true, lower, upper, alpha=.1):
    """
    Winkler Scorer

    Args:
        y_true: true value of target.
        lower/upper: lower & upper bound of prediction

    Returns: wrinkler score
    """
    
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)
    return width + (2/alpha) * (below + above)


#
# Preprocessing - Imputation
#

# train & test
train_df = impute(train)
test_df = impute(test)

# view
train_df.head()


#
# Preprocessing - New Features
#

# train & test
train_df = new_features(train_df)
test_df = new_features(test_df)

# drop 'sale_date' column
train_df = train_df.drop(columns=['sale_date'])
test_df = test_df.drop(columns=['sale_date'])

# view
train_df.head()


#
# Preprocessing - Encoding
#

# train & test
train_df = encode(train_df,encoding='ordinal')
test_df = encode(test_df,encoding='ordinal')

# view
train_df.head()


#
# Feature Engineering & Model Config
#

# feature & target
x = train_df.drop(columns=['sale_price'])
y = train_df['sale_price']

# hybrid model
models = {
    "lower": lgb.LGBMRegressor(
        objective="quantile", 
        alpha=0.05,
        device="gpu",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed
    ),
    "upper": xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.95,
        device='cuda',
        n_estimators=1500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=6,
        tree_method="hist",
        random_state=seed
    )
}


# Finding optimal weight for model training

# Initialize storage for OOF predictions
oof_lowers = np.zeros(len(x))
oof_uppers = np.zeros(len(x))

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y)), total=5, desc="Folds"):
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    # Train lower quantile model
    lower_model = clone(models["lower"])
    lower_model.fit(x_train, y_train)
    lower_pred = lower_model.predict(x_val)
        
    # Train upper quantile model
    upper_model = clone(models["upper"])
    upper_model.fit(x_train, y_train)
    upper_pred = upper_model.predict(x_val)
        
    oof_lowers[val_idx] = lower_pred
    oof_uppers[val_idx] = upper_pred

print("\nModel Performance Evaluation:", oof_lowers, oof_uppers)

# Ensure valid intervals
lower = np.minimum(oof_lowers, oof_uppers)
upper = np.maximum(oof_lowers, oof_uppers)
    
# Calculate MWIS
wis = winkler_score(y, lower, upper)
mwis = np.mean(wis)
model_score = mwis
    
# Calculate coverage
coverage = np.mean((y >= lower) & (y <= upper)) * 100
print(f"Result: MWIS = {mwis:.2f}, Coverage = {coverage:.2f}%")


# Hill Climbing Optimization
print("\nStarting Hill Climbing Optimization...")
current_weights = .4
best_score = 100000000000000000.0
    
# Calculate initial combined score
combined_lower = current_weights * oof_lowers
combined_upper = current_weights * oof_uppers
current_score = np.mean(winkler_score(y, combined_lower, combined_upper))
    
print(f"Initial MWIS: {current_score:.4f}")
    
# Optimization loop
for step in tqdm(range(100), desc="Hill Climbing"):
    candidate_weights = current_weights
        
    # Generate candidate weights
    perturbation = np.random.dirichlet([9])[0] - .9
    candidate_weights = candidate_weights + .1 * perturbation
    candidate_weights = np.maximum(candidate_weights, 0)

    # Calculate combined predictions
    combined_lower = candidate_weights * oof_lowers
    combined_upper = candidate_weights * oof_uppers
        
    # Calculate MWIS
    candidate_score = np.mean(winkler_score(y, combined_lower, combined_upper))

    # Update if improvement
    if candidate_score < best_score:
        best_score = candidate_score
        best_weights = candidate_weights
        current_weights = candidate_weights
        print(f"Step {step}: New best MWIS = {best_score:.4f}, Weights = {best_weights}")


# Final model training
print("\nTraining Models...")
test_preds = {}

# Train lower quantile model
lower_model = models["lower"]
lower_model.fit(x, y)
test_preds["lower"] = lower_model.predict(test_df)

# Train upper quantile model
upper_model = models["upper"]
upper_model.fit(x, y)
test_preds["upper"] = upper_model.predict(test_df)

# Combine test predictions
final_lower = best_weights * test_preds["lower"]
final_upper = best_weights * test_preds["upper"]

# Ensure valid intervals
final_lower, final_upper = np.minimum(final_lower, final_upper), np.maximum(final_lower, final_upper)
final_lower = np.maximum(final_lower, 0)


#
# Submission
#

test_ids = test_df["id"]

submission = pd.DataFrame({
    "id": test_ids,
    "pi_lower": final_lower,
    "pi_upper": final_upper
})

submission.to_csv("submission.csv", index=False)
print("Submission saved successfully\n")

submission = pd.read_csv("submission.csv")

# view
submission.head()

