import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np 
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import optuna 



#load data 
ins_path = '/kaggle/input/playground-series-s4e10/train.csv'
ins_full_df = pd.read_csv(ins_path) 

# identify numerical data
num_cols = ((ins_full_df.dtypes == 'int64')| (ins_full_df.dtypes == 'float64'))
num_list = list(num_cols[num_cols].index)

# exclude id and loan_status 
num_drop = ['id','loan_status']
num_final = [k for k in num_list if k not in num_drop]

#identify categorical data
cat_cols = ins_full_df.dtypes == 'object'
cat_list = list(cat_cols[cat_cols].index)   # there is no need to exclude any catagorical data

# define both transformers
num_transformer = SimpleImputer(strategy='mean')

cat_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy = 'most_frequent')),
                                  ('OHE', OneHotEncoder(handle_unknown = 'ignore'))])


full_col = num_final + cat_list   #establish features and targets, from all useful data in the training data
X = ins_full_df[full_col]
y = ins_full_df.loan_status

#create preprocessor
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_final), 
                                               ( 'cat', cat_transformer, cat_list)])


import matplotlib.pyplot as plt

ax = y.value_counts().plot(kind="bar")
ax.set_title("Loan approval label counts")
ax.set_xlabel("loan_status")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()

cols = ["person_age", "loan_percent_income", "loan_amnt"]

for c in cols:
    ax = X[c].hist(bins=50)
    ax.set_title(f"Distribution: {c}")
    ax.set_xlabel(c)
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()



#defining models with different seeds
xgb_params = {'learning_rate': 0.034437561078606294, 'n_estimators': 1379, 'max_depth': 5, 'subsample': 0.939633605230379, 'colsample_bytree': 0.5379887024016051, 'min_child_weight': 4, 'gamma': 0.41725949046109445, 'reg_alpha': 0.8870451366409303, 'reg_lambda': 1.2125409014625224}
model = XGBClassifier(**xgb_params, random_state = 0)
model1 = XGBClassifier(**xgb_params, random_state = 1)
model2 = XGBClassifier(**xgb_params, random_state =2)
model3 = XGBClassifier(**xgb_params, random_state = 3)
model4 = XGBClassifier(**xgb_params, random_state = 4)

# full pipeline for each model
full_mod = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model)])
full_mod1 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model1)])
full_mod2 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model2)])
full_mod3 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model3)])
full_mod4 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model4)])





# define cv strategy
cV = StratifiedKFold(n_splits =5, shuffle = True,random_state=1)
diff_mod = [full_mod,full_mod1,full_mod2,full_mod3,full_mod4]
n = len(X)
fold_sc = []
# create array to store final Out-of-Fold predictions
oof_pred = np.zeros(n, dtype=float) 

# outer loop : runs through each of the 5 stratified folds in cV for the dataset and generates train/validation sets for each fold
for train_idx, val_idx in cV.split(X, y):       
    X_train =  X.iloc[train_idx]
    y_train =  y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # empty array that is used to sum each of the 5 sets of probabilties generated, reset to zero for each fold
    sum_preds = np.zeros(len(val_idx))  

    # inner loop : Iterates through each of the five models 
    for k in diff_mod :
        
        k_fold = clone(k)    # cloned to prevent carrying forward a fitted model into later predictions
        k_fold.fit(X_train,y_train)          
        val_preds = k_fold.predict_proba(X_val)[:, 1]
        sum_preds = sum_preds + val_preds   # adds new predictions into summation array 
    
    # averages the ensemble predictions and adds into the final store, using validation indices
    oof_pred[val_idx]  = (sum_preds/5) 
    val_score = roc_auc_score(y_val,sum_preds/5)
    fold_sc.append(val_score)

print(roc_auc_score(y, oof_pred))
print(np.std(fold_sc))


## finding standard deviation of one pipeline
scores = cross_val_score(full_mod,X,y,cv =cV,scoring = 'roc_auc')

print(scores.mean())

print(scores.std())


## testing method, allows me to try multiple values 
def testing(pipeline,X,y,cv_method) : 
    means= []
    stds = []
    for k in [0.55,0.6,0.65,0.7,0.75] :   ## example values 
        full_modtest_k = pipeline.set_params(model__subsample = k)
        score = cross_val_score(full_modtest_k,X,y,cv= cv_method,scoring= 'roc_auc')
        means.append(score.mean())
        stds.append(score.std())   
    return means,stds



mean , std = testing(full_mod,X,y,cV)
testing_df = pd.DataFrame({'k' : [0.55,0.6,0.65,0.7,0.75], 'mean' : mean, 'standard deviation' : std})
print(testing_df)


## as a result 
'''learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =1'''


def objective(trial) :     ## objective function for Bayesian optimisation
    ## parameter search space
    parameters = {'learning_rate' : trial.suggest_float('learning_rate',0.005,0.3),  
                  'n_estimators' : trial.suggest_int('n_estimators',500,2000),
                  'max_depth' : trial.suggest_int('max_depth',2,6),
                  'subsample' : trial.suggest_float('subsample',0.6,1),
                  'colsample_bytree' : trial.suggest_float('colsample_bytree',0.3,0.8),
                  'min_child_weight' : trial.suggest_int('min_child_weight',3,9),
                  'gamma' : trial.suggest_float('gamma',0,3),
                  'reg_alpha' : trial.suggest_float('reg_alpha',0,10.0),
                  'reg_lambda' : trial.suggest_float('reg_lambda',1.0,10.0)}
                  
    score_op = []
    ## cross validation loops
    model_op = XGBClassifier(**parameters, random_state =42, n_jobs = -1)
    for fold, (train_idx, val_idx) in enumerate(cV.split(X,y)) :    ## splits data into folds according to cV, and tests on each fold
        X_train, Val_X = X.iloc[train_idx], X.iloc[val_idx]
        y_train, Val_y = y.iloc[train_idx], y.iloc[val_idx]
        
        full_mod_op = Pipeline(steps=[('preprocessor',preprocessor), ## create testing pipeline
                                  ('model', model_op) ])
        full_mod_op.fit(X_train,y_train)
        score_preds = full_mod_op.predict_proba(Val_X)[:, 1]   ## predict on the validation split
        score_op.append(roc_auc_score(Val_y, score_preds))  ## add the prediction to the empty list

        running_mean = np.mean(score_op)  # running mean of scores 
        trial.report(running_mean, step=fold)

        if trial.should_prune():
            raise optuna.TrialPruned()   ## prunes a test if its score is unpromising in the first few folds 
    return float(np.mean(score_op))        ## returns mean score averaged over all folds


## create study and optimise
study = optuna.create_study(study_name = 'optuna_test_12.0', storage="sqlite:///optuna.db", direction='maximize',load_if_exists= True)
study.optimize(objective, n_trials=500,show_progress_bar= True, n_jobs = 1)

best_params = study.best_params
print(best_params)

