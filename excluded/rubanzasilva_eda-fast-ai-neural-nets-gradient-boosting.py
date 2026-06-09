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


%%time
%pip install catboost
%pip install optuna
%pip install optuna_distributed
#%pip install openfe
%pip install seaborn
%pip install xgboost
%pip install lightgbm
%pip install fastkaggle
#%pip install h2o
%pip install -Uqq fastbook
#%pip install polars
%pip install -q -U autogluon.tabular
%pip install autogluon
%pip install --upgrade pip
%pip install tqdm
#%pip install wandb
#%pip install sweetviz
%pip install xlearn


#%pip install -U autogluon > /dev/null
# In your terminal/command prompt
#pip install numpy==1.24.3
# or
%pip install numpy==1.23.5


# Update to compatible versions
!pip install autogluon==1.1.1 xgboost==2.0.3
# Restart kernel after installation


#%pip freeze > requirements.txt


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import random
from tqdm import tqdm

#import fastbook
#fastbook.setup_book()
#from fastbook import *
from fastai.tabular.all import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import random
from tqdm import tqdm
from ipywidgets import interact

from fastai.imports import *
np.set_printoptions(linewidth=130)


from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score,train_test_split,GridSearchCV


from pathlib import Path
import os

import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier,CatBoostRegressor,Pool, metrics, cv


import warnings


#from openfe import OpenFE, transform
from autogluon.tabular import TabularDataset, TabularPredictor

#import h2o
#from h2o.automl import H2OAutoML

import gc

import optuna
from optuna.samplers import TPESampler

import pickle
from joblib import dump, load
#import sweetviz as sv
#from IPython.display import FileLink

#import h2o
#from h2o.automl import H2OAutoML
import xlearn as xl


path = Path('/kaggle/input/playground-series-s5e8/')
path


train_df = pd.read_csv(path/'train.csv',index_col='id')
test_df = pd.read_csv(path/'test.csv',index_col='id')
sub_df = pd.read_csv(path/'sample_submission.csv')
original_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')


!ls /kaggle/input/playground-series-s5e8


train_df.shape, original_df.shape


train_df.shape, original_df.shape


train_df.head()


original_df.head()


# Map 'yes'/'no' to 1/0 in the target variable 'y'
original_df['y'] = original_df['y'].map({'yes': 1, 'no': 0})
original_df


train_df = pd.concat([train_df, original_df], ignore_index=True)
train_df


# Read first few lines as text to see the actual format
with open('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', 'r') as f:
    for i in range(3):
        print(repr(f.readline()))


train_df.info()


train_df.columns


missing_values_count = train_df.isnull().sum()
sorted_missing_values = missing_values_count.sort_values(ascending=False)
print(sorted_missing_values)


#Missing values returned as percentages.
missing_percentages = train_df.isnull().mean() * 100
sorted_missing_percentages = missing_percentages.sort_values(ascending=False)
print(sorted_missing_percentages)


train_df.hist(figsize=(20,15),edgecolor='black');


train_df.describe().T


splits = RandomSplitter(valid_pct=0.2)(range_of(original_df))


#train_df = pd.concat([train_df, original_df], ignore_index=True)


cont_names,cat_names = cont_cat_split(train_df, dep_var='y')
#splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
#to = TabularPandas(train_df, procs=[Categorify,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='y',
                   y_block=CategoryBlock(),
                   splits=splits)
dls = to.dataloaders(bs=64)
#dls = to.dataloaders(bs=1024)
test_dl = dls.test_dl(test_df)

X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()


cont_names,len(cont_names)


cat_names, len(cat_names)


X_train


def cross_val_predict(model, X_train, y_train, X_test, n_splits=5, random_state=42, 
                       return_proba=True, scoring='roc_auc', verbose=True):
    """
    Generic cross-validation function that works with ANY sklearn-compatible model.
    
    Parameters:
    - model: INSTANTIATED model object (e.g., xgb.XGBClassifier(n_estimators=100))
    - X_train, y_train: training data and labels
    - X_test: test data for final predictions
    - n_splits: number of CV folds
    - random_state: random seed for reproducibility
    - return_proba: if True, return probabilities; if False, return class predictions
    - scoring: metric to use ('roc_auc', 'accuracy')
    - verbose: print progress
    
    Returns:
    - oof_predictions: out-of-fold predictions on training set
    - test_predictions: predictions on test set (averaged across folds)
    - mean_score: mean score across folds
    - fold_scores: list of scores for each fold
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.base import clone
    import numpy as np
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results
    if return_proba:
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
    else:
        oof_predictions = np.zeros(len(X_train), dtype=int)
        test_predictions = np.zeros(len(X_test))
    
    fold_scores = []
    
    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
        if verbose:
            print(f"Training fold {fold}/{n_splits}...")
        
        # Split data for current fold
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        # Clone the model to avoid interference between folds
        model_fold = clone(model)
        
        # Train model on current fold
        model_fold.fit(X_fold_train, y_fold_train)
        
        # Get predictions based on return_proba setting
        if return_proba:
            # Get probabilities for positive class
            y_pred_proba = model_fold.predict_proba(X_fold_val)[:, 1]
            oof_predictions[val_index] = y_pred_proba
            test_predictions += model_fold.predict_proba(X_test)[:, 1] / n_splits
            
            # Calculate score
            if scoring == 'roc_auc':
                cv_score = roc_auc_score(y_fold_val, y_pred_proba)
            else:  # accuracy
                y_pred_fold = (y_pred_proba > 0.5).astype(int)
                cv_score = accuracy_score(y_fold_val, y_pred_fold)
        else:
            # Get class predictions
            y_pred_fold = model_fold.predict(X_fold_val)
            oof_predictions[val_index] = y_pred_fold
            test_predictions += model_fold.predict(X_test) / n_splits
            cv_score = accuracy_score(y_fold_val, y_pred_fold)
        
        fold_scores.append(cv_score)
        if verbose:
            print(f"Fold {fold} {scoring}: {cv_score:.6f}")
    
    # Round test predictions if using hard predictions
    if not return_proba:
        test_predictions = np.round(test_predictions).astype(int)
    
    # Calculate mean score
    mean_score = np.mean(fold_scores)
    if verbose:
        print(f"\nMean {scoring}: {mean_score:.6f}")
    
    return oof_predictions, test_predictions, mean_score, fold_scores


# =============================================================================
# USAGE EXAMPLES WITH DIFFERENT MODELS
# =============================================================================

# Example: CatBoost (if installed)
"""
cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)

cat_oof, cat_test, cat_auc, cat_scores = cross_val_predict(
    model=cat_model,
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test,
    return_proba=True,
    scoring='roc_auc'
)
"""

# =============================================================================
# ENSEMBLE ALL MODELS
# =============================================================================
"""
print("Individual Model Performance:")
print(f"XGBoost AUC: {xgb_auc:.6f}")
print(f"Random Forest AUC: {rf_auc:.6f}")
print(f"LightGBM AUC: {lgb_auc:.6f}")

# Simple ensemble
ensemble_oof = (xgb_oof + rf_oof + lgb_oof) / 3
ensemble_test = (xgb_test + rf_test + lgb_test) / 3

ensemble_auc = roc_auc_score(y_train, ensemble_oof)
print(f"\nEnsemble AUC: {ensemble_auc:.6f}")

# Create submission
submission = pd.DataFrame({
    'id': test_df.index,
    'Depression': ensemble_test
})
submission.to_csv('ensemble_submission.csv', index=False)

print(f"\nShapes:")
print(f"OOF predictions: {ensemble_oof.shape} (training data)")
print(f"Test predictions: {ensemble_test.shape} (test data)")
print("Different shapes = different datasets!")
"""


learn = tabular_learner(dls, metrics=RocAucBinary())


#learn.lr_find()
learn.lr_find(suggest_funcs=(slide,valley))


%%time
learn.fit_one_cycle(30, 1.025e-2)


dl = test_dl


learn.save('nn_pnality_tp_pd')


nn_preds = learn.get_preds(dl=dl)
nn_preds_x = learn.get_preds()[0]
a_preds, _ = learn.get_preds(dl=dl)
nn_preds_y = a_preds.squeeze(1)
nn_preds_proba = (a_preds[:, 1])


a_preds.shape


nn_preds_y.shape,sub_df.shape


nn_preds_x


nn_preds_y


a_preds


nn_preds_proba


#final_preds = torch.round(a_preds).long()
final_preds = a_preds[:,1]
final_preds


final_preds.shape


!rm submission.csv
sub_df['y'] = final_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


%%time
rf = RandomForestClassifier(100, min_samples_leaf=3)
rf_model = rf.fit(X_train, y_train);

rf_preds = tensor(rf_model.predict(test_dl.xs))
rf_preds_probs = tensor(rf_model.predict_proba(test_dl.xs))[:, 1]

rf_preds_x = tensor(rf_model.predict(X_test))
rf_preds_proba = tensor(rf_model.predict_proba(X_test))[:, 1]

#mse = mean_absolute_error(y_test, rf_preds_x)
#rmse = np.sqrt(mse)

#accuracy_score(y_test,rf_preds_x)
rf_score = roc_auc_score(y_test,rf_preds_proba)
rf_score


rf_preds_probs,rf_preds_probs.shape


rf_preds_probs 


def rf_feat_importance(m, train_subset):
    return pd.DataFrame({'cols':train_subset.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(rf_model, X_train)
#fi[:10]

fi


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

#plot_fi(fi[:30]);
plot_fi(fi);


rf = RandomForestClassifier(100, min_samples_leaf=3)
rf


rf_oof_predictions, rf_test_predictions, rf_mean_score, rf_fold_scores = cross_val_predict(
    model=rf, X_train=X_train, y_train=y_train, X_test=X_test, 
)


rf_oof_predictions.shape, rf_test_predictions.shape


rf_oof_predictions


rf_test_predictions


rf_mean_score, rf_fold_scores


!rm submission.csv
sub_df['y'] = rf_preds_probs
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


%%time
xgb_model = xgb.XGBClassifier()
xgb_model = xgb_model.fit(X_train, y_train)

xgb_preds = tensor(xgb_model.predict(test_dl.xs))
xgb_preds_proba = tensor(xgb_model.predict_proba(test_dl.xs))[:, 1]

xgb_preds_x = tensor(xgb_model.predict(X_test))
xgb_preds_x_proba = tensor(xgb_model.predict_proba(X_test))[:, 1]

xgb_score = roc_auc_score(y_test,xgb_preds_x_proba)
xgb_score


xgb_preds_proba


#plot_importance(xgb_model.fit(X_train, y_train))
plot_importance(xgb_model)


xgb_model = xgb.XGBClassifier()
xgb_model


xgb_oof_predictions, xgb_test_predictions, xgb_mean_score, xgb_fold_scores = cross_val_predict(
    model=xgb_model, X_train=X_train, y_train=y_train, X_test=X_test, 
)


xgb_oof_predictions


xgb_test_predictions


xgb_mean_score


 xgb_fold_scores


!rm submission.csv
sub_df['y'] = xgb_preds_proba
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


%%time
#ds subset
lgbm_model = lgb.LGBMClassifier()
lgbm_model = lgbm_model.fit(X_train, y_train)

#test set preds
lgbm_preds = tensor(lgbm_model.predict(test_dl.xs))
lgbm_preds_prob = tensor(lgbm_model.predict_proba(test_dl.xs))
lgbm_preds_proba = (lgbm_preds_prob[:, 1])

#validation set preds
lgbm_preds_x = tensor(lgbm_model.predict(X_test))
lgbm_preds_x_prob = tensor(lgbm_model.predict_proba(X_test))
lgbm_positive_preds_x = (lgbm_preds_x_prob[:, 1])

lgbm_score = roc_auc_score(y_test,lgbm_positive_preds_x)
lgbm_score

#lgb_preds_x_prob = tensor(lgb_model.predict_proba(X_test))

lgbm_score = roc_auc_score(y_test,lgbm_positive_preds_x)
lgbm_score


lgbm_preds_prob,lgbm_preds_prob.shape


# Plot feature importance without using Gain or split
lgb.plot_importance(lgbm_model, figsize=(7,6), title="LightGBM Feature Importance")
#plt.title('LGBM Feature Importance')
#plt.tight_layout()
plt.show()


lgbm_model = lgb.LGBMClassifier()
lgbm_model


lgbm_oof_predictions, lgbm_test_predictions, lgbm_mean_score, lgbm_fold_scores = cross_val_predict(
    model=lgbm_model, X_train=X_train, y_train=y_train, X_test=X_test,
)


lgbm_oof_predictions


lgbm_test_predictions


 lgbm_mean_score


lgbm_fold_scores


!rm submission.csv
sub_df['y'] = lgbm_preds_proba
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


%%time
#using full ds
cat_model = CatBoostClassifier()
cat_model = cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

#test set preds
cat_preds = tensor(cat_model.predict(test_dl.xs))
cat_preds_probs = tensor(cat_model.predict_proba(test_dl.xs))[:, 1]
#cat_preds_final = cat_preds.squeeze(1)

#validation set preds
cat_preds_x = tensor(cat_model.predict(X_test))
cat_preds_x_proba = tensor(cat_model.predict_proba(X_test))[:, 1]

#cat_preds_x_final = cat_preds_x.squeeze(1)

#accuracy_score(y_test,cat_preds_x)

cat_score = roc_auc_score(y_test,cat_preds_x_proba)
cat_score


feature_importance_default = cat_model.get_feature_importance()
feature_names = X_train.columns

# Create DataFrame for better visualization
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_default
}).sort_values('importance', ascending=False)

print("=== Default Feature Importance (PredictionValuesChange) ===")
print(importance_df.head(10))


cat_model = CatBoostClassifier()
cat_model


cat_oof_predictions, cat_test_predictions, cat_mean_score, cat_fold_scores = cross_val_predict(
    model=cat_model, X_train=X_train, y_train=y_train, X_test=X_test, 
)


cat_xgb_preds = (xgb_preds_proba + cat_preds_probs)/ 2
cat_xgb_preds 


cat_xgb_x_preds = (xgb_preds_x_proba + cat_preds_x_proba)/ 2
cat_xgb_x_preds 


cat_xgb_score = roc_auc_score(y_test,cat_xgb_x_preds)
cat_xgb_score


cat_lgbm_preds = (lgbm_preds_proba + cat_preds_probs)/ 2
cat_lgbm_preds 


cat_lgbm_x_preds = (lgbm_positive_preds_x + cat_preds_x_proba)/ 2
cat_lgbm_x_preds.shape


cat_lgbm_score = roc_auc_score(y_test,cat_lgbm_x_preds)
cat_lgbm_score


lgbm_xgb_preds = (lgbm_preds_proba +  xgb_preds_proba)/ 2
lgbm_xgb_preds 


lgbm_xgb_preds_x = (lgbm_positive_preds_x + xgb_preds_x_proba)/ 2
lgbm_xgb_score = roc_auc_score(y_test,lgbm_xgb_preds_x)
lgbm_xgb_score


!rm submission.csv
sub_df['y'] = lgbm_xgb_preds 
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


cat_lgbm_xgb_preds = (lgbm_preds_proba + cat_preds_probs + xgb_preds_proba)/ 3
cat_lgbm_xgb_preds 


cat_lgbm_xgb_preds_x = (lgbm_positive_preds_x + cat_preds_x_proba + xgb_preds_x_proba)/ 3
cat_lgbm_xgb_score = roc_auc_score(y_test,cat_lgbm_xgb_preds_x)
cat_lgbm_xgb_score


stacking_estimators = [
    ('cat_boost',cat_model),
    #('rf',rf_model),
    ('lgbm',lgbm_model),
    ('xgb',xgb_model),
]

#stacking_classifier_cat_xgb_lgbm= StackingClassifier(
    #estimators=stacking_estimators,
    #final_estimator=LogisticRegression(),
    #cv=5
#)
stacking_classifier_cat_lgbm= StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=lgb.LGBMClassifier(),
    cv=5
)
stacking_classifier_cat_lgbm.fit(X_train, y_train)


stacking_preds_cat_lgbm = (stacking_classifier_cat_lgbm.predict_proba(test_dl.xs))[:,1]
stacking_preds_cat_lgbm_x = (stacking_classifier_cat_lgbm.predict_proba(X_test))[:,1]

stacking_score_cat_lgbm = roc_auc_score(y_test, stacking_preds_cat_lgbm_x)

print(f"Final Stacking Classifier ROC_AUC on test set: {stacking_score_cat_lgbm}")

print(f"Final Stacking Classifier ROC_AUC on test set: {stacking_score_cat_lgbm}")


stacking_preds_cat_lgbm.shape


stacking_preds_cat_lgbm = torch.tensor(stacking_preds_cat_lgbm)


stacking_preds_cat_lgbm


!rm submission.csv
sub_df['y'] = stacking_preds_cat_lgbm 
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


target = 'y'
eval_metric = 'roc_auc'
#train_data = train
train_data = train_df
#Time_limit = 3600*10
Time_limit = 3600
problem_type='binary'


%%time
#predictor = TabularPredictor(label=target, eval_metric=eval_metric, verbosity=1).fit(
    #train_data, presets='best_quality', time_limit=Time_limit,
    #ag_args_fit={'num_gpus': 2}
    #ag_args_fit={
        #'num_gpus': 2, 
        #'stopping_metric': 'log_loss'
    
    #}
#)

#results = predictor.fit_summary()


%%time
#predictor = TabularPredictor(label=target, eval_metric=eval_metric,verbosity=1,problem_type=problem_type).fit(
    #train_data, presets='best_quality',excluded_model_types=['KNN'], time_limit=Time_limit,
    #ag_args_fit={
        #'num_gpus': 2, 
        #'stopping_metric': 'log_loss'
    #}
#)


%%time
#results = predictor.fit_summary()


%%time
#predictor.leaderboard()


%%time
#autogluon_preds = predictor.predict(test_df)
#autogluon_preds_proba = predictor.predict_proba(test_df)
#autogluon_preds_proba.head(5)  


#predictions = autogluon_preds_proba


%%time
#autogluon_preds = predictor.predict(test_df)
#autogluon_preds_proba = predictor.predict_proba(test_df, as_multiclass=False)
#autogluon_preds_proba.head(5)  


#autogluon_preds_proba.shape, cat_lgbm_xgb_preds.shape


#autogluon_preds_proba.values


#autogluon_preds_proba_values = torch.tensor(autogluon_preds_proba.values)


#autogluon_cat_lgbm_xgb_preds = (cat_lgbm_xgb_preds + autogluon_preds_proba_values)/2
#autogluon_cat_lgbm_xgb_preds,autogluon_cat_lgbm_xgb_preds.shape


#!rm submission.csv
#sub_df['y'] = autogluon_cat_lgbm_xgb_preds
#sub_df.to_csv('submission.csv', index=False)
#sub = pd.read_csv('submission.csv', index_col='id')
#sub


#!rm submission.csv
#sub_df['y'] = autogluon_preds_proba.values
#sub_df.to_csv('submission.csv', index=False)
#sub = pd.read_csv('submission.csv', index_col='id')
#sub


#!rm submission.csv
#submit = pd.read_csv(path/'sample_submission.csv')
#predictions = predictions.reset_index(drop=False)
#submit = pd.DataFrame({'id': predictions['id'], 'Personality': predictions['Personality']})
#submit.set_index('id', inplace=True)
#submit.to_csv('submission.csv', index=True)
#sub = pd.read_csv('submission.csv', index_col='id')
#sub


!rm submission.csv
sub_df['y'] = cat_lgbm_preds 
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


"""
Comprehensive GPU-optimized Optuna optimization for binary classification Kaggle competition
10-hour runtime with extensive hyperparameter search using XGBoost on GPU
Incorporates best practices from previous optimization experience
Optimizes using AUC-ROC with cross-validation for robust evaluation
"""
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time
import gc

# Load your data - modify path and target column name according to your data
def load_data():
    # Replace 'path' with your actual path variable and 'target' with your target column name
    from pathlib import Path
    path = Path('.')  # Modify this to your actual path
    
    train_df = pd.read_csv(path/'train.csv',index_col='id')
    test_df = pd.read_csv(path/'test.csv',index_col='id')
    
    # Assuming your target column is named 'target' - modify according to your actual target column name
    X = train_df.drop(['y'], axis=1)  # Remove only target column, id is already index
    y = train_df['y']
    X_test = test_df.copy()  # Test set doesn't need any columns dropped, id is already index
    
    return X, y, X_test, test_df.index

def objective(trial):
    # Load data once and reuse (global scope for efficiency)
    if not hasattr(objective, "data_loaded"):
        objective.X, objective.y, objective.X_test, objective.test_ids = load_data()
        objective.data_loaded = True
    
    X, y = objective.X, objective.y
    
    # Comprehensive hyperparameter search space combining best practices
    params = {
        # Core XGBoost parameters
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',  # GPU acceleration (your approach)
        'tree_method': 'gpu_hist',
        'random_state': 42,
        'verbosity': 0,
        
        # Tree structure - extended ranges from your approach
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Your range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # Your range
        'max_leaves': trial.suggest_int('max_leaves', 0, 1000),  # Your addition
        'max_bin': trial.suggest_int('max_bin', 200, 1000),  # Your addition
        
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),  # Your range
        'n_estimators': trial.suggest_int('n_estimators', 50, 10000),  # Your extended range
        
        # Regularization - your parameter names and ranges
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        
        # Sampling parameters - your ranges
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # Additional sampling for comprehensive search
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        
        # Advanced parameters for binary classification
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
    }
    
    # Optional advanced parameters (uncommented versions of your code)
    params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
    
    # DART-specific parameters when selected
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 0.01, 0.9)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.01, 0.9)
        params['one_drop'] = trial.suggest_categorical('one_drop', [0, 1])
    
    # Use KFold like your approach but with more folds for longer runtime
    K_FOLDS = 5  # Increased from your 3 for more robust evaluation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold_idx, (train_index, val_index) in enumerate(kfold.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_index], X.iloc[val_index]
        y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]
        
        try:
            # Use XGBClassifier for binary classification instead of Regressor
            xgb_model_fold = xgb.XGBClassifier(**params)
            
            # Fit with early stopping for efficiency
            xgb_model_fold.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Get probability predictions for AUC calculation
            y_pred_proba = xgb_model_fold.predict_proba(X_fold_val)[:, 1]
            
            # Calculate AUC score for current fold
            auc_score = roc_auc_score(y_fold_val, y_pred_proba)
            fold_scores.append(auc_score)
            
            # Clean up memory
            del xgb_model_fold
            gc.collect()
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            return 0.5  # Return baseline AUC score if error
        
        # Report intermediate results for pruning
        trial.report(np.mean(fold_scores), fold_idx)
        
        # Prune unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return mean CV AUC score
    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    
    # Log progress like your approach
    print(f"Trial {trial.number}: CV AUC = {mean_cv_score:.5f} ± {std_cv_score:.5f}")
    
    return mean_cv_score

def train_best_model_and_predict(best_params):
    """Train final model with best parameters and generate predictions"""
    # Load data
    X, y, X_test, test_ids = load_data()
    
    # Prepare final parameters
    final_params = best_params.copy()
    final_params.update({
        'device': 'cuda',
        'tree_method': 'gpu_hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'verbosity': 1,
    })
    
    print("Training final model...")
    
    # Train final model with best parameters
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        X, y,
        eval_set=[(X, y)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Generate predictions for test set
    print("Generating predictions...")
    test_preds = final_model.predict_proba(X_test)[:, 1]  # Get probabilities
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,  # test_ids is now the index from test_df
        'target': test_preds  # Modify column name according to your submission format
    })
    
    submission.to_csv('submission_optuna_gpu_comprehensive.csv', index=False)
    print("Submission file saved as 'submission_optuna_gpu_comprehensive.csv'")
    
    return final_model, test_preds

if __name__ == "__main__":
    print("Starting comprehensive GPU-optimized hyperparameter search...")
    print("Expected runtime: ~10 hours")
    print("Using improved parameter ranges and best practices from previous optimization")
    
    # Create study with your TPE sampler approach but optimized for longer runtime
    sampler = TPESampler(
        n_startup_trials=50,  # Increased from your 30 for 10-hour runtime
        multivariate=True,    # Your setting
        seed=42               # Changed from your 0 for consistency
    )
    
    # Add pruner for efficiency in long runtime
    pruner = MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=2,
        interval_steps=1
    )
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize"  # Maximize AUC instead of minimize RMSE
    )
    
    # Run optimization for 10 hours
    start_time = time.time()
    study.optimize(
        objective,
        timeout=36000,  # 10 hours in seconds
        n_jobs=1,       # Sequential for GPU
        gc_after_trial=True
    )
    
    end_time = time.time()
    runtime_hours = (end_time - start_time) / 3600
    
    print(f"\nOptimization completed in {runtime_hours:.2f} hours")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Results similar to your format
    print(f"\nBest AUC: {study.best_value:.6f}")
    print("Best parameters:", study.best_trial.params)
    
    # Save detailed results
    study_df = study.trials_dataframe()
    study_df.to_csv('optuna_study_results_comprehensive.csv', index=False)
    print("\nStudy results saved as 'optuna_study_results_comprehensive.csv'")
    
    # Train final model and generate predictions
    print("\nTraining final model with best parameters...")
    best_model, predictions = train_best_model_and_predict(study.best_trial.params)
    print("Final model trained and predictions generated!")
    
    # Enhanced feature importance analysis
    try:
        feature_importance = best_model.feature_importances_
        feature_names = objective.X.columns if hasattr(objective, 'X') else [f'feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Feature Importance:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Save feature importance
        importance_df.to_csv('feature_importance.csv', index=False)
        
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")
    
    # Final optimization summary
    print(f"\nOptimization Summary:")
    print(f"- Runtime: {runtime_hours:.2f} hours")
    print(f"- Total trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"- Best AUC achieved: {study.best_value:.6f}")
    print(f"- Model saved and submission generated")


auc_roc_score = pd.DataFrame({
    'algorithm': ['Random Forest','XGBoost','LGBM','CatBoost',
                  'CatBoost_XGB_average','CatBoost_LGBM_average',
                  'XGB_LGBM_average','CatBoost_XGB_LGBM_average',
                  
                 ],
    'auc_roc_score': [rf_score,xgb_score,lgbm_score,cat_score,
                      cat_xgb_score,cat_lgbm_score,lgbm_xgb_score,
                      cat_lgbm_xgb_score,
                 
           ]
})

auc_roc_sorted = auc_roc_score.sort_values(by='auc_roc_score', ascending=False)
auc_roc_sorted

