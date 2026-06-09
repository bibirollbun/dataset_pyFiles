"""
External Predictions:

"""
import optuna


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier as lgbm
import lightgbm as LGB

from catboost import CatBoostClassifier as cbc

from xgboost import XGBClassifier as XGBC
import xgboost as xgb


# LGBM No Feature Eng
lgbm_no_feat_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-lgbm/submission_probas_lgbm.csv")["diagnosed_diabetes"]
lgbm_no_feat_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-lgbm/oof_probas_lgbm.csv")["diagnosed_diabetes"]

# ---
# CatBoost No Feat Eng
cbc_no_feat_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-cat/submission_cat_probas.csv")["diagnosed_diabetes"]
cbc_no_feat_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-cat/oof_stacking_probas.csv")["diagnosed_diabetes"]
# ---
# XGBoost No Feat Eng
xgb_no_feat_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-xgb/submission_probas_xgbc.csv")["diagnosed_diabetes"]
xgb_no_feat_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-wo-feat-eng-s5-ep-12-xgb/oof_probas_xgbc.csv")["diagnosed_diabetes"]



# ---
# LGBM Feat Eng
lgbm_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-lgbm-s5-ep12/submission_probas_lgbm.csv")["diagnosed_diabetes"]
lgbm_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-lgbm-s5-ep12/oof_probas_lgbm.csv")["diagnosed_diabetes"]

# ---
# XGBoost Feat Eng
xgb_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-xgb-s5-ep-12/submission_probas_xgbc.csv")["diagnosed_diabetes"]
xgb_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-xgb-s5-ep-12/oof_probas_xgbc.csv")["diagnosed_diabetes"]

# ---
# CatBoost Feat Eng
cbc_test_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-cat-s5-ep-12/submission_cat_probas.csv")["diagnosed_diabetes"]
cbc_oof_probas = pd.read_csv("/kaggle/input/standard-pipeline-w-feat-eng-cat-s5-ep-12/oof_stacking_probas.csv")["diagnosed_diabetes"]




best_blend_test = pd.read_csv("/kaggle/input/best-sub-mikhail-naumov-70651/submission.csv")["diagnosed_diabetes"]

# CatBoost Feat Eng
external_test_probas = pd.read_csv("/kaggle/input/external-69896/submission.csv")["diagnosed_diabetes"]
external_oof_probas = pd.read_csv("/kaggle/input/external-69896/oof_predictions.csv")["pred"]



def add_row_stats(df):
    """
    Calculates the median, mean, min, max, and range for each row 
    and adds these as new columns to the original DataFrame.

    Args:
        df: The input pandas DataFrame.

    Returns:
        The original DataFrame with five new columns appended.
    """
    # Calculate statistics across axis 1 (rows)
    df['st_dev'] = np.std(df, axis=1)
    # Calculate the range using the new columns
    min_set = df.min(axis=1)
    max_set = df.max(axis=1)
    df['row_range'] = max_set - min_set
    
    return df


X = pd.concat([lgbm_no_feat_oof_probas, cbc_no_feat_oof_probas, xgb_no_feat_oof_probas, lgbm_oof_probas, cbc_oof_probas, xgb_oof_probas, external_oof_probas], axis=1)
X.columns = ["lgbm", "cat", "xgb", "lgbm_F", "cat_F", "xgb_F", "external_src"]

X_test = pd.concat([lgbm_no_feat_test_probas, cbc_no_feat_test_probas, xgb_no_feat_test_probas, lgbm_test_probas, cbc_test_probas, xgb_test_probas, external_test_probas], axis=1)
X_test.columns = ["lgbm", "cat", "xgb", "lgbm_F", "cat_F", "xgb_F", "external_src"]

X = add_row_stats(X)
X_test = add_row_stats(X_test)

y = pd.read_csv("/kaggle/input/feature-engineering-s5-e12/y.csv")["diagnosed_diabetes"]


print(X.head(5))

print(y.head(5))


"""
Official Metric Used - ROC
"""


def objective(trial):
    """

Best ROC AUC: 0.7314379748427123
Best Params: 
{'n_estimators': 1017, 'learning_rate': 0.02821602778766206, 'num_leaves': 20, 'max_depth': 45, 
'min_child_samples': 270, 'subsample': 0.8998061577643618, 'colsample_bytree': 0.7089754963487098, 
'reg_alpha': 0.29074450474410274, 'reg_lambda': 0.7722962134716419}
    """
    
    params = {
        "objective": "binary",
        "metric": "auc",          # ROC AUC
        "boosting_type": "gbdt",
        
        "n_estimators": trial.suggest_int("n_estimators", 1016, 1018),
        
        "learning_rate": trial.suggest_float("learning_rate", 0.02821, 0.02822),
        
        "num_leaves": trial.suggest_int("num_leaves", 18, 22),
        "max_depth": trial.suggest_int("max_depth", 40, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 260, 280),
        
        "subsample": trial.suggest_float("subsample", 0.1, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 0.9),
        "verbosity": -1
    }

    X_flat = np.array(X)
    y_flat = np.array(y).ravel()

    lgb_train = LGB.Dataset(X_flat, label=y_flat)

    cv_results = LGB.cv(
        params,
        lgb_train,
        nfold=5,                # Stratified K-Fold
        stratified=True,
        metrics='auc',          # ROC AUC
        seed=trial.number,
        callbacks=[
            LGB.early_stopping(stopping_rounds=100),
            LGB.log_evaluation(period=0)  # suppress printing
        ]
    )
    # maximize AUC (Optuna minimizes by default)
    return -max(cv_results['valid auc-mean'])
"""
# Create and optimize study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print("Best ROC AUC:", -study.best_value)
print("Best Params:", study.best_params)
"""


"""
import itertools

param_splits = {
    "n_estimators": [(1000, 1500), (1500, 2000)],
    "max_depth": [(10, 35), (35, 50)],
    "num_leaves": [(10, 70), (70, 100)],
    "min_child_samples": [(10, 333), (333, 666)]
}

# get all combinations of sub-ranges (3^4 = 81)
all_combinations = list(itertools.product(
    param_splits["n_estimators"],
    param_splits["max_depth"],
    param_splits["num_leaves"],
    param_splits["min_child_samples"]
))

best_results = []

for combo in all_combinations:
    n_est_range, max_depth_range, leaves_range, child_range = combo
        
    def objective(trial, ):

        params = {
            "objective": "binary",
            "metric": "auc",          # ROC AUC
            "boosting_type": "gbdt",
            
            "n_estimators": trial.suggest_int("n_estimators", *n_est_range),
            
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.03),
            
            "num_leaves": trial.suggest_int("num_leaves", *leaves_range),
            "max_depth": trial.suggest_int("max_depth", *max_depth_range),
            "min_child_samples": trial.suggest_int("min_child_samples", *child_range),
            
            "subsample": trial.suggest_float("subsample", 0.01, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.9),
            "verbosity": -1
        }
    
        X_flat = np.array(X)
        y_flat = np.array(y).ravel()
    
        lgb_train = LGB.Dataset(X_flat, label=y_flat)
    
        cv_results = LGB.cv(
            params,
            lgb_train,
            nfold=5,                # Stratified K-Fold
            stratified=True,
            metrics='auc',          # ROC AUC
            seed=trial.number,
            callbacks=[
                LGB.early_stopping(stopping_rounds=100),
                LGB.log_evaluation(period=0)  # suppress printing
            ]
        )
        # maximize AUC (Optuna minimizes by default)
        return -max(cv_results['valid auc-mean'])
    
    # Create and optimize study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print("Best ROC AUC:", -study.best_value)
    print("Best Params:", study.best_params)
"""


"""
Best Params: 
{'n_estimators': 1017, 'learning_rate': 0.02821602778766206, 'num_leaves': 20, 'max_depth': 45, 
'min_child_samples': 270, 'subsample': 0.8998061577643618, 'colsample_bytree': 0.7089754963487098, 
'reg_alpha': 0.29074450474410274, 'reg_lambda': 0.7722962134716419}
"""

meta_model = lgbm(
    boosting_type='gbdt',   # standard Gradient Boosting
    objective='binary', # ensures continuous targets
    metric='auc',
    n_estimators=1373,      # number of boosting rounds
    learning_rate = 0.0038435395838249493,     # step size shrinkage
    num_leaves=28,          # max leaves per tree
    max_depth= 34,      # -1 = no limit
    min_child_samples = 306,
    subsample = 0.4478304667848626,          # row sampling
    colsample_bytree= 0.7555222801889582,   # feature sampling
    reg_alpha = 0.5319957344115895,
    reg_lambda = 0.72544865629588265,
    random_state= 42,
    n_jobs=-1,               # use all cores
    verbose=-1
)
def meta_learning(uX, uy, uX_test, model):
    meta_model = lgbm(
                    boosting_type='gbdt',   # standard Gradient Boosting
                    objective='binary', # ensures continuous targets
                    metric='auc',
                    n_estimators=1017,      # number of boosting rounds
                    learning_rate = 0.02821602778766206,     # step size shrinkage
                    num_leaves=20,          # max leaves per tree
                    max_depth= 45,      # -1 = no limit
                    min_child_samples = 270,
                    subsample = 0.8998061577643618,          # row sampling
                    colsample_bytree= 0.7089754963487098,   # feature sampling
                    reg_alpha = 0.29074450474410274,
                    reg_lambda = 0.7722962134716419,
                    random_state= 42,
                    n_jobs=-1,               # use all cores
                    verbose=-1
                )

    model.fit(uX, uy)
    stacking_preds = model.predict(uX_test)
    stacking_probas = model.predict_proba(uX_test)

    return stacking_preds, stacking_probas


raw_preds, raw_probas = meta_learning(X, y, X_test, meta_model)


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv") # Just for ID
positive_probas = raw_probas[:, 1] 
print(positive_probas[:10])


submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": positive_probas  # predicted values
})
submission.to_csv("final_submission_stacking_raw_FullTestSet.csv", index=False)



weighted_pos_probas = 0.9 * positive_probas + 0.2 * best_blend_test 
submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": weighted_pos_probas  # predicted values
})
submission.to_csv("final_submission_stacking_raw_FullTestSet_weighted_1.csv", index=False)


best_blind_test_2 = pd.read_csv("/kaggle/input/ensemble/ensemble.csv")["diagnosed_diabetes"]
weighted_pos_probas = 0.9 * positive_probas + 0.2 * best_blind_test_2
submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": weighted_pos_probas  # predicted values
})
submission.to_csv("final_submission_stacking_raw_FullTestSet_weighted_2.csv", index=False)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from lightgbm import LGBMClassifier as lgbm
import lightgbm as LGB

def stratified_cv_evaluate(model, uX, uy, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    fold = 0
    for train_idx, val_idx in skf.split(uX, uy):
        fold +=1
        X_train, X_val = uX.iloc[train_idx], uX.iloc[val_idx]
        y_train, y_val = uy.iloc[train_idx], uy.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
        print(f"FOLD {fold} out of {n_splits}. ROC Score:", roc_auc_score(y_val, preds))
    print("All Folds Average ROC SCORE: " + str(np.mean(scores)) + "\n All Folds St.Dev of ROC Score: " + str(np.std(scores)))

meta_model = lgbm(
    boosting_type='gbdt',   # standard Gradient Boosting
    objective='binary', # ensures continuous targets
    metric='auc',
    n_estimators=1373,      # number of boosting rounds
    learning_rate = 0.0038435395838249493,     # step size shrinkage
    num_leaves=28,          # max leaves per tree
    max_depth= 34,      # -1 = no limit
    min_child_samples = 306,
    subsample = 0.4478304667848626,          # row sampling
    colsample_bytree= 0.7555222801889582,   # feature sampling
    reg_alpha = 0.5319957344115895,
    reg_lambda = 0.72544865629588265,
    random_state= 42,
    n_jobs=-1,               # use all cores
    verbose=-1
)
stratified_cv_evaluate(meta_model, X, y, n_splits = 10)




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from lightgbm import LGBMClassifier as lgbm
import lightgbm as LGB
X["test"] = 0
X_test["test"] = 1
#data = pd.concat([X.drop('diagnosed_diabetes', axis=1), X_test], axis=0)
data = pd.concat([X, X_test], axis=0)

y = data["test"]
data = data.drop(columns=["test"])
X = X.drop(columns=["test"])
X_test = X_test.drop(columns=["test"])

def stratified_cv_evaluate(model, uX, uy, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    # Initialize an array to hold OOF predictions for every row in the combined data
    oof_probas = np.zeros(len(uX))
    
    fold = 0
    for train_idx, val_idx in skf.split(uX, uy):
        fold += 1
        X_train, X_val = uX.iloc[train_idx], uX.iloc[val_idx]
        y_train, y_val = uy.iloc[train_idx], uy.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        
        # Store predictions in the indices corresponding to the validation set
        oof_probas[val_idx] = preds
        
        score = roc_auc_score(y_val, preds)
        scores.append(score)
        print(f"FOLD {fold}/{n_splits}. ROC: {score:.4f}")
        
    print(f"\nMean ROC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return oof_probas


base_model = lgbm(
    boosting_type='gbdt',   # standard Gradient Boosting
    objective='binary', # ensures continuous targets
    metric='auc',
    n_estimators=2000,      # number of boosting rounds
    learning_rate = 0.05,     # step size shrinkage
    num_leaves=31,          # max leaves per tree
    max_depth= -1,      # -1 = no limit
    min_child_samples = 20,
    subsample = .8,          # row sampling
    colsample_bytree= .8,   # feature sampling
    reg_alpha = .1,
    reg_lambda = .1,
    random_state= 42,
    n_jobs=-1,               # use all cores
    verbose=-1
)

# 3. Run Adversarial Validation
all_probas = stratified_cv_evaluate(meta_model, data, y, n_splits=10)




# 4. Split back into Train and Test probabilities
# Since we concatenated Train then Test, we slice the array
train_probas = all_probas[:len(X)]
test_probas = all_probas[len(X):]
y = pd.read_csv("/kaggle/input/feature-engineering-s5-e12/y.csv")["diagnosed_diabetes"]


print(f"Generated {len(train_probas)} train weights and {len(test_probas)} test weights.")

train_probas = train_probas / (1- train_probas)
test_probas = test_probas / (1- test_probas)



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from lightgbm import LGBMClassifier as lgbm
import lightgbm as LGB

def stratified_cv_evaluate(model, uX, uy, index, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    fold = 0
    index = np.array(index)
    for train_idx, val_idx in skf.split(uX, uy):
        current_weights = index[train_idx]
        
        fold +=1
        X_train, X_val = uX.iloc[train_idx], uX.iloc[val_idx]
        y_train, y_val = uy.iloc[train_idx], uy.iloc[val_idx]
        model.fit(X_train, y_train, sample_weight= current_weights)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
        print(f"FOLD {fold} out of {n_splits}. ROC Score:", roc_auc_score(y_val, preds))
    print("All Folds Average ROC SCORE: " + str(np.mean(scores)) + "\n All Folds St.Dev of ROC Score: " + str(np.std(scores)))

meta_model = lgbm(
    boosting_type='gbdt',   # standard Gradient Boosting
    objective='binary', # ensures continuous targets
    metric='auc',
    n_estimators=1373,      # number of boosting rounds
    learning_rate = 0.0038435395838249493,     # step size shrinkage
    num_leaves=28,          # max leaves per tree
    max_depth= 34,      # -1 = no limit
    min_child_samples = 306,
    subsample = 0.4478304667848626,          # row sampling
    colsample_bytree= 0.7555222801889582,   # feature sampling
    reg_alpha = 0.5319957344115895,
    reg_lambda = 0.72544865629588265,
    random_state= 42,
    n_jobs=-1,               # use all cores
    verbose=-1
)

stratified_cv_evaluate(meta_model, X, y, train_probas, n_splits = 10)

"""
FOLD 1 out of 10. ROC Score: 0.7314923522956585
FOLD 2 out of 10. ROC Score: 0.7322910840090052
FOLD 3 out of 10. ROC Score: 0.7307410097377176
FOLD 4 out of 10. ROC Score: 0.7301780073799347
FOLD 5 out of 10. ROC Score: 0.7318009700253246
FOLD 6 out of 10. ROC Score: 0.7310732727189575
FOLD 7 out of 10. ROC Score: 0.7296842584733721
FOLD 8 out of 10. ROC Score: 0.7334831725673305
FOLD 9 out of 10. ROC Score: 0.732733802997958
FOLD 10 out of 10. ROC Score: 0.7305169709326131
All Folds Average ROC SCORE: 0.7313994901137872
 All Folds St.Dev of ROC Score: 0.0011331269375991246

 
"""


def meta_learning(uX, uy, uX_test, model, index):
    meta_model = lgbm(
                    boosting_type='gbdt',   # standard Gradient Boosting
                    objective='binary', # ensures continuous targets
                    metric='auc',
                    n_estimators=1017,      # number of boosting rounds
                    learning_rate = 0.02821602778766206,     # step size shrinkage
                    num_leaves=20,          # max leaves per tree
                    max_depth= 45,      # -1 = no limit
                    min_child_samples = 270,
                    subsample = 0.8998061577643618,          # row sampling
                    colsample_bytree= 0.7089754963487098,   # feature sampling
                    reg_alpha = 0.29074450474410274,
                    reg_lambda = 0.7722962134716419,
                    random_state= 42,
                    n_jobs=-1,               # use all cores
                    verbose=-1
                )

    model.fit(uX, uy, sample_weight = index)
    stacking_preds = model.predict(uX_test)
    stacking_probas = model.predict_proba(uX_test)

    return stacking_preds, stacking_probas



raw_preds, raw_probas = meta_learning(X, y, X_test, meta_model, train_probas)
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv") # Just for ID
positive_probas = raw_probas[:, 1] 
print(positive_probas[:10])


submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": positive_probas  # predicted values
})
submission.to_csv("final_submission_fullTestSet_weighted_raw.csv", index=False)

weighted_pos_probas = 0.8 * positive_probas + 0.2 * best_blend_test
submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": weighted_pos_probas  # predicted values
})
submission.to_csv("final_submission_fullTestSet_weighted_raw_blend_1.csv", index=False)

best_blind_test_2 = pd.read_csv("/kaggle/input/ensemble/ensemble.csv")["diagnosed_diabetes"]
weighted_pos_probas = 0.8 * positive_probas + 0.2 * best_blind_test_2
submission = pd.DataFrame({
    
    "id": test['id'],        # from test set
    "loan_paid_back": weighted_pos_probas  # predicted values
})
submission.to_csv("final_submission_fullTestSet_weighted_raw_blend_2.csv", index=False)

