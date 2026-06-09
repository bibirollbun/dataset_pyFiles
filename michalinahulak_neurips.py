!pip install /kaggle/input/rdkit-2025-3-3-cp311-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import seaborn as sns
import warnings

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import re

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train.head(2)


train.info()


desc_names = [desc[0] for desc in Descriptors.descList]

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

def preprocess_data(df):
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES']]
    descriptors_df = pd.DataFrame(descriptors, columns=desc_names, index=df.index)
    df = pd.concat([df, descriptors_df], axis=1)
    df.drop(columns='SMILES')
    return df


def show_distribution(df, column):
    """
    Display descriptive statistics and a histogram for a given column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The name of the column to analyze.
    """
    print(f"Descriptive statistics for the '{column}' column:")
    print(df[column].describe())
    
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {column} Values')
    plt.xlabel(column)
    plt.ylabel('Number of Samples')
    plt.grid(True)
    plt.show()


def val_loss_function(actual, predicted):
    return mean_absolute_error(actual, predicted)

def cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5):
    print(f"Model: {model.__class__.__name__}")

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_score = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        cur_val_score = val_loss_function(y_val, val_preds)
        print(f"Current validation score: {cur_val_score:.4f}")
        
        val_score += cur_val_score / n_splits

        test_preds += model.predict(X_test) / n_splits

    print(f"Average validation score: {val_score:.4f}")
    return oof_preds, test_preds, val_score


train = preprocess_data(train)
train.head(2)


print(train.isna().sum()[train.isna().sum() > 0])


df_test = preprocess_data(test)
df_test.head(2)


cols_to_drop = [
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'
]

train = train.drop(columns=cols_to_drop, errors='ignore')
df_test = df_test.drop(columns=cols_to_drop, errors='ignore')


NUNIQUE1=[c for c in train.columns if train[c].nunique()==1]
print(len(NUNIQUE1))
NUNIQUE1


train = train.drop(columns=NUNIQUE1)
df_test = df_test.drop(columns=NUNIQUE1)


ffv_train = train[train['FFV'].notna()].drop(columns = ['Tg', 'Tc', 'Density', 'Rg'])
print(ffv_train.shape)


show_distribution(ffv_train, 'FFV')


label_col = 'FFV'
features = ffv_train.drop(columns=['id', 'SMILES', 'FFV'], errors='ignore').columns
X_train = ffv_train[features]
y_train = ffv_train[label_col]
X_test = df_test[features]


models = [
    # LGBMRegressor(
    #     boosting_type='gbdt',
    #     device='gpu'  
    # ),
    # XGBRegressor(
    #     tree_method='gpu_hist', 
    #     predictor='gpu_predictor'
    # ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['CatBoostRegressor']['test'])
sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
sub['FFV'] = y_pred
sub


tg_train = train[train['Tg'].notna()].drop(columns = ['FFV', 'Tc', 'Density', 'Rg'])
print(tg_train.shape)


show_distribution(tg_train, 'Tg')


numeric_cols = tg_train.select_dtypes(include='number').columns

correlations = tg_train[numeric_cols].corr()['Tg'].drop('Tg').drop('id').sort_values(ascending=False).abs()

selected_features = correlations[correlations.abs() > 0.01].index.tolist()


label_col = 'Tg'
# features = tg_train.drop(columns=['id', 'SMILES', 'Tg'], errors='ignore').columns
# features
X_train = tg_train[selected_features]
y_train = tg_train[label_col]
X_test = df_test[selected_features]


models = [
    # LGBMRegressor(
    #     boosting_type='gbdt',
    #     device='gpu'  
    # ),
    # XGBRegressor(
    #     tree_method='gpu_hist', 
    #     predictor='gpu_predictor'
    # ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['CatBoostRegressor']['test'])
sub['Tg'] = y_pred
sub


tc_train = train[train['Tc'].notna()].drop(columns = ['FFV', 'Tg', 'Density', 'Rg'])

print(tc_train.shape)


show_distribution(tc_train, 'Tc')


# numeric_cols = tc_train.select_dtypes(include='number').columns

# correlations = tc_train[numeric_cols].corr()['Tc'].drop('Tc').sort_values(ascending=False).abs()

# selected_features = correlations[correlations.abs() > 0.001].index.tolist()


label_col = 'Tc'
features = tc_train.drop(columns=['id', 'SMILES', 'Tc'], errors='ignore').columns

X_train = tc_train[features]
y_train = tc_train[label_col]

X_test = df_test[features]


models = [
    # LGBMRegressor(
    #     boosting_type='gbdt',
    #     device='gpu'  
    # ),
    # XGBRegressor(
    #     tree_method='gpu_hist', 
    #     predictor='gpu_predictor'
    # ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['CatBoostRegressor']['test'])
sub['Tc'] = y_pred
sub


den_train = train[train['Density'].notna()].drop(columns = ['FFV', 'Tg', 'Tc', 'Rg'])

print(den_train.shape)


show_distribution(den_train, 'Density')


# numeric_cols = den_train.select_dtypes(include='number').columns

# correlations = den_train[numeric_cols].corr()['Density'].drop('Density').sort_values(ascending=False).abs()

# selected_features = correlations[correlations.abs() > 0.01].index.tolist()


label_col = 'Density'
features = den_train.drop(columns=['id', 'SMILES', 'Density'], errors='ignore').columns

X_train = den_train[features]
y_train = den_train[label_col]

X_test = df_test[features]


models = [
    # LGBMRegressor(
    #     boosting_type='gbdt',
    #     device='gpu'  
    # ),
    # XGBRegressor(
    #     tree_method='gpu_hist', 
    #     predictor='gpu_predictor'
    # ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['CatBoostRegressor']['test'])
sub['Density'] = y_pred
sub


rg_train = train[train['Rg'].notna()].drop(columns = ['FFV', 'Tg', 'Tc', 'Density'])

print(rg_train.shape)


show_distribution(rg_train, 'Rg')


# numeric_cols = rg_train.select_dtypes(include='number').columns

# correlations = rg_train[numeric_cols].corr()['Rg'].drop('Rg').sort_values(ascending=False).abs()
# selected_features = correlations[correlations.abs() > 0.0001].index.tolist()


label_col = 'Rg'
features = rg_train.drop(columns=['id', 'SMILES', 'Rg'], errors='ignore').columns

X_train = rg_train[features]
y_train = rg_train[label_col]

X_test = df_test[features]


models = [
    # LGBMRegressor(
    #     boosting_type='gbdt',
    #     device='gpu'  
    # ),
    # XGBRegressor(
    #     tree_method='gpu_hist', 
    #     predictor='gpu_predictor'
    # ),
    CatBoostRegressor(
        verbose=0,
        task_type='GPU',  
        devices='0'       
    )
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = (results['CatBoostRegressor']['test'])
sub['Rg'] = y_pred
sub.to_csv('submission.csv', index = False)
sub

