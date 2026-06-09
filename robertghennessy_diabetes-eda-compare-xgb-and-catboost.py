!pip install optuna-integration


import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.ensemble import IsolationForest
import scipy
import optuna

import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


KAGGLE = True
warnings.filterwarnings("ignore")


random_seed = 42
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
seed_everything(seed=random_seed)


if KAGGLE:
    train_csv = '/kaggle/input/playground-series-s5e12/train.csv'
    test_csv = '/kaggle/input/playground-series-s5e12/test.csv'
    # Download latest version
    import kagglehub
    orig_dataset_csv = os.path.join(
        kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset"),
        'diabetes_dataset.csv'
    )
else:
    train_csv = '/Users/roberthennessy/Documents/machine learning/diabetes-prediction/train.csv'
    test_csv = '/Users/roberthennessy/Documents/machine learning/diabetes-prediction/test.csv'
    orig_dataset_csv = '/Users/roberthennessy/Documents/machine learning/diabetes-prediction/diabetes_dataset.csv'
    

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
orig_df = pd.read_csv(orig_dataset_csv)


target_var = 'diagnosed_diabetes'
multi_categorical_columns = ['gender', 'ethnicity', 'education_level', 'income_level',
                        'smoking_status', 'employment_status'
                      ]
single_categorical_columns = ['family_history_diabetes', 'hypertension_history',
                        'cardiovascular_history'
                      ]
numerical_columns = ['age', 'alcohol_consumption_per_week',
                   'physical_activity_minutes_per_week', 'diet_score',
                   'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
                   'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
                   'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
                   'ldl_cholesterol', 'triglycerides']


train_df[multi_categorical_columns] = train_df[multi_categorical_columns].astype('category')
train_df[single_categorical_columns] = train_df[single_categorical_columns].astype('category')
train_df[target_var] = train_df[target_var].astype('category')

test_df[multi_categorical_columns] = test_df[multi_categorical_columns].astype('category')
test_df[single_categorical_columns] = test_df[single_categorical_columns].astype('category')



train_df.drop(columns=['id'], inplace=True)
print('Train Shape:', train_df.shape)
print('Test Shape:', test_df.shape)
print('Orig Shape:', orig_df.shape)
# drop columns that are not being used in the competition
orig_df = orig_df[train_df.columns]
print('Orig Shape:', orig_df.shape)


print("Duplicated Rows:",train_df.duplicated().sum())
print("-"*30)
print("Duplicated Rows:",test_df.duplicated().sum())
print("-"*30)
print("Duplicated Rows:",orig_df.duplicated().sum())
print("-"*30)


train_df.isnull().sum()


train_describe = train_df.isnull().sum()
train_describe.name = 'train_df'
test_describe = test_df.isnull().sum()
test_describe.name = 'test_df'
orig_describe = orig_df.isnull().sum()
orig_describe.name = 'orig_df'
print (pd.concat([train_describe, test_describe, orig_describe], axis=1))


print(train_df.nunique())


def plot_overlayed_histograms(df1, df2, df3, col, df1_label, df2_label, df3_label, bins):

    # Plot overlaid histograms
    plt.hist(df1[col], bins=bins, alpha=0.7, label=df1_label, color='blue')
    plt.hist(df2[col], bins=bins, alpha=0.7, label=df2_label, color='orange')
    plt.hist(df3[col], bins=bins, alpha=0.7, label=df3_label, color='green')
    
    plt.title('Overlaid Histograms')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_multi_hist(df1, df2, df3, col, df1_label, df2_label, df3_label, bins):
    fig, axs = plt.subplots(1, 3, figsize=(15,6)) # 1 row, 2 columns
    #df['category'].value_counts().sort_index(ascending=False).plot.bar(title='Sorted by Value (Descending)')
    axs[0].hist(df1[col], bins=bins, color='blue')
    axs[0].set_title(df1_label)
    axs[1].hist(df2[col], bins=bins, label=df2_label, color='orange')
    axs[1].set_title(df2_label)
    axs[2].hist(df3[col], bins=bins, label=df3_label, color='green')
    axs[2].set_title(df3_label)
    plt.show()

def plot_multi_hist2(df1, df2, df3, col, df1_label, df2_label, df3_label, bins):
    fig, axs = plt.subplots(1, 3, figsize=(15,6)) # 1 row, 2 columns
    df1[col].value_counts().sort_index(ascending=False).plot.hist(bins=bins, color='blue', ax=axs[0], title=df1_label)
    df2[col].value_counts().sort_index(ascending=False).plot.hist(bins=bins, color='orange', ax=axs[1], title=df2_label)
    df3[col].value_counts().sort_index(ascending=False).plot.hist(bins=bins, color='green', ax=axs[2], title=df3_label)
    plt.show()

def plot_multi_bar(df1, df2, df3, col, df1_label, df2_label, df3_label):
    fig, axs = plt.subplots(1, 3, figsize=(15,6)) # 1 row, 2 columns
    df1[col].value_counts().plot.bar(color='blue', ax=axs[0], title=df1_label)
    df2[col].value_counts().plot.bar(color='orange', ax=axs[1], title=df2_label)
    df3[col].value_counts().plot.bar(color='green', ax=axs[2], title=df3_label)
    plt.show()


plot_overlayed_histograms(train_df, test_df, orig_df, 'age', 'train_df', 'test_df','orig_df', 20)
plot_multi_hist(train_df, test_df, orig_df, 'age', 'train_df', 'test_df','orig_df', 20)
plot_multi_bar(train_df, test_df, orig_df, 'ethnicity', 'train_df', 'test_df','orig_df')


def describe_col(col): 
    train_describe = train_df[col].describe()
    train_describe.name = 'train_df'
    test_describe = test_df[col].describe()
    test_describe.name = 'test_df'
    orig_describe = orig_df[col].describe()
    orig_describe.name = 'orig_df'
    return pd.concat([train_describe, test_describe, orig_describe], axis=1)
print(describe_col('age'))


def percentage_for_col(col): 
    train_describe = train_df[col].value_counts(dropna=False, normalize=True)
    train_describe.name = 'train_df'
    test_describe =  test_df[col].value_counts(dropna=False, normalize=True)
    test_describe.name = 'test_df'
    orig_describe = orig_df[col].value_counts(dropna=False, normalize=True)
    orig_describe.name = 'orig_df'
    return pd.concat([train_describe, test_describe, orig_describe], axis=1)


print(train_df[target_var].value_counts(dropna=False, normalize=True))
print(orig_df[target_var].value_counts(dropna=False, normalize=True))


for col in multi_categorical_columns + single_categorical_columns:
    print(col)
    print(percentage_for_col(col))
    plot_multi_bar(train_df, test_df, orig_df, col, 'train_df', 'test_df','orig_df')


for col in numerical_columns:
    print(col)
    print(describe_col(col))
    plot_multi_hist(train_df, test_df, orig_df, col, 'train_df', 'test_df','orig_df', bins=None)


for col in  numerical_columns:
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    print(col)
    sns.boxplot(x=target_var, y=col, 
                data=train_df, orient='v' , ax=axes[0])
    sns.boxplot(x=target_var, y=col, 
                data=orig_df, orient='v' , ax=axes[1])
    plt.show()


# Compute correlation matrix
corr_matrix = pd.concat([train_df[numerical_columns], train_df[target_var]],axis=1).corr()


# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap (Numerical Features + Diabetes Diagnosis)', fontsize=16)
plt.tight_layout()
plt.show()


def split_data(df):
    X, y = df.drop([target_var], axis=1), df[target_var]
    #X = X.apply(pd.to_numeric)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=random_seed, 
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def analyze_predictions(model, eval_metric, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix:\n{confusion}")
    # Calculate the confusion matrix

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, 
                                  display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    
    print("\n Classification Report:\n",
          classification_report(y_test, y_pred))


def analyze_model_xgb(model, eval_metric, X_test, y_test):
    # Analyze Model
    # Retrieve the merror values from the training process
    results = model.evals_result()
    epochs = len(results['validation_0'][eval_metric])
    x_axis = range(0, epochs)
    
    # Plot the merror values
    plt.figure()
    plt.plot(x_axis, results['validation_0'][eval_metric], label='Test')
    plt.legend()
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel(eval_metric)
    plt.title('XGBoost Performance')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(4, 15))
    xgb.plot_importance(model, ax=ax)
    plt.show()

    analyze_predictions(model, eval_metric, X_test, y_test)
    


def train_model_xgb(df):

    df[multi_categorical_columns] = df[multi_categorical_columns].astype('category')
    df[single_categorical_columns] = df[single_categorical_columns].astype('category')
    df[target_var] = df[target_var].astype('category')
    
    X_train, X_test, y_train, y_test = split_data(df)
    #print(X_train.dtypes)
    eval_metric = 'auc'
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': eval_metric,
        'early_stopping_rounds': 10,
        'random_state': random_seed,
        'enable_categorical': True,
        'tree_method': 'hist', 
        'device': 'cuda',
        'n_jobs': -1
    }

    
    # Instantiate XGBClassifier with the parameters
    model = xgb.XGBClassifier(**params)
    # Train the model with early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    analyze_model_xgb(model, eval_metric, X_test, y_test)
    
  
    return model





def predict_xgb(model, df):
    df[multi_categorical_columns] = df[multi_categorical_columns].astype('category')
    df[single_categorical_columns] = df[single_categorical_columns].astype('category')

    prediction = model.predict_proba(df.drop(columns=['id']))[:,1]
    positivies = sum(prediction >= 0.5) / len(prediction)
    print(positivies)

    plt.hist(prediction, bins=50, color='green')
    plt.title('Probability of disease diagnosis ')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Build submission
    submission = pd.DataFrame({
        'id': df['id'].values,
        'diagnosed_diabetes': prediction
    })
    
    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)



if False:
    model_xgb = train_model_xgb(train_df)
    predict_xgb(model_xgb, test_df)


def analyze_model_catboost(model, eval_metric, X_test, y_test):
    analyze_predictions(model, eval_metric, X_test, y_test)
    

    importances = model.get_feature_importance()
    feature_names = X_test.columns
    sorted_indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[sorted_indices])
    plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
    plt.title("Feature Importance")
    plt.show()


def train_model_catboost(df):

    df[multi_categorical_columns] = df[multi_categorical_columns].astype('category')
    df[single_categorical_columns] = df[single_categorical_columns].astype('category')
    df[target_var] = df[target_var].astype('category')
    
    X_train, X_test, y_train, y_test = split_data(df)
    eval_metric = 'AUC'
    
    params = {
        'loss_function': 'Logloss',
        'custom_metric': eval_metric,
        'early_stopping_rounds': 10,
        'random_state': random_seed,
        'cat_features': multi_categorical_columns + single_categorical_columns,
        'verbose': 100,
        'task_type': 'GPU'
    }

    
    # Instantiate CatBoost Classifier with the parameters
    model = cb.CatBoostClassifier(**params)
    # Train the model with early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    analyze_model_catboost(model, eval_metric, X_test, y_test)
    
    return model





def predict_catboost(model, df):
    df[multi_categorical_columns] = df[multi_categorical_columns].astype('category')
    df[single_categorical_columns] = df[single_categorical_columns].astype('category')

    prediction = model.predict_proba(df.drop(columns=['id']))[:,1]
    positivies = sum(prediction >= 0.5 ) / len(prediction)
    print(positivies)

    plt.hist(prediction, bins=50, color='green')
    plt.title('Probability of disease diagnosis ')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Build submission
    submission = pd.DataFrame({
        'id': df['id'].values,
        'diagnosed_diabetes': prediction
    })
    
    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)


if False:
    model_catboost = train_model_catboost(train_df)
    predict_catboost(model_catboost, test_df)


def objective_xgb(trial: optuna.Trial) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'learning_rate': trial.suggest_float('eta', 1e-3, 3e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
        'tree_method': 'hist',
        'random_state': random_seed,
        'enable_categorical': True,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
        'callbacks': [optuna.integration.XGBoostPruningCallback(trial, 'validation_0-auc')],
        'device': 'cuda',
        'n_jobs': -1
        
    }
 
    model = xgb.XGBClassifier(**params)
 
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
 
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc


def cv_objective_xgb(trial: optuna.Trial) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'learning_rate': trial.suggest_float('eta', 1e-3, 3e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
        'tree_method': 'hist',
        'random_state': random_seed,
        'enable_categorical': True,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
        'callbacks' :[optuna.integration.XGBoostPruningCallback(trial, 'validation_0-auc')],
        'device': 'cuda',
        'n_jobs': -1
    }
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    aucs = []
    for train_idx, valid_idx in cv.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False
        )
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return float(np.mean(aucs))


def optuna_optimize_xgb(df):
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=50),
        study_name="xgb_classifier",
    )
     
    study.optimize(cv_objective_xgb, n_trials=20, timeout=None, n_jobs=1, show_progress_bar=False)
    print({"best_value": study.best_value, "best_trial": study.best_trial.number})

    best_trial = study.best_trial
    print("Best AUC:", best_trial.value)
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
     
    # Inspect top-5 trials
    top5 = sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)[:5]
    for t in top5:
        print({"trial": t.number, "value": t.value})

    return best_trial





def train_best_xgb_model(trial, train_df, test_df):
    best_xgb_params = trial.params.copy()
    best_xgb_params.update({
        'n_estimators': max(300, best_trial_xgb.params.get('n_estimators', 500)),
        'random_state': random_seed,
        'tree_method': 'hist',
        'n_jobs': 0,
        'enable_categorical': True,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
        'device': 'cuda',
        'n_jobs': -1
    })
    X, y = train_df.drop([target_var], axis=1), train_df[target_var]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    aucs = []
    test_preds = []
    for train_idx, valid_idx in cv.split(X, y):
        model = xgb.XGBClassifier(**best_xgb_params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False
        )
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
        prediction = model.predict_proba(test_df.drop(columns=['id']))[:,1]
        test_preds.append(prediction)
    print("Mean:", np.mean(aucs))
    print("Std:", np.std(aucs))
    return test_preds


if False:
    X, y = train_df.drop([target_var], axis=1), train_df[target_var]
    best_trial_xgb = optuna_optimize_xgb(train_df)
    test_preds = train_best_xgb_model(best_trial_xgb, train_df, test_df)
    test_preds = np.column_stack(test_preds)
    final_pred = test_preds.mean(axis=1)
    print(final_pred)
    # Build submission
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'diagnosed_diabetes': final_pred
    })
    
    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)





def cv_objective_catboost(trial: optuna.Trial) -> float:
    params = {
        'loss_function': 'Logloss',
        'early_stopping_rounds': 10,
        'random_state': random_seed,
        'cat_features': multi_categorical_columns + single_categorical_columns,
        'verbose': 100,
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.09, log=True),
        'depth': trial.suggest_int('depth', 5, 9),
        'iterations': trial.suggest_int('iterations', 100, 1500),
        'random_strength': trial.suggest_float('random_strength', 0, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 10.0),
        'task_type': 'GPU' # Enable GPU training
    }
    
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    aucs = []
    for train_idx, valid_idx in cv.split(X, y):
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False
        )
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return float(np.mean(aucs))


def optuna_optimize_catboost(df):
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=50),
        study_name="catboost_classifier",
    )
     
    study.optimize(cv_objective_catboost, n_trials=20, timeout=None, n_jobs=1, 
                   show_progress_bar=False)
    print({"best_value": study.best_value, "best_trial": study.best_trial.number})

    best_trial = study.best_trial
    print("Best AUC:", best_trial.value)
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
     
    # Inspect top-5 trials
    top5 = sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)[:5]
    for t in top5:
        print({"trial": t.number, "value": t.value})

    return best_trial





def train_best_catboost_model(trial, train_df, test_df):
    best_catboost_params = trial.params.copy()
    best_catboost_params.update({
        'random_state': random_seed,
        'eval_metric': 'Logloss',
        'cat_features': multi_categorical_columns + single_categorical_columns,
        'early_stopping_rounds': 50,
        'task_type': 'GPU' # Enable GPU training
    })
    X, y = train_df.drop([target_var], axis=1), train_df[target_var]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    aucs = []
    test_preds = []
    for train_idx, valid_idx in cv.split(X, y):
        model = cb.CatBoostClassifier(**best_catboost_params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False
        )
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
        prediction = model.predict_proba(test_df.drop(columns=['id']))[:,1]
        test_preds.append(prediction)
    print("Mean:", np.mean(aucs))
    print("Std:", np.std(aucs))
    return test_preds





if True:
    X, y = train_df.drop([target_var], axis=1), train_df[target_var]
    best_trial_catboost = optuna_optimize_catboost(train_df)
    test_preds = train_best_catboost_model(best_trial_catboost, train_df, test_df)
    test_preds = np.column_stack(test_preds)
    final_pred = test_preds.mean(axis=1)
    print(final_pred)
    # Build submission
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'diagnosed_diabetes': final_pred
    })
    
    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)


#test_preds = train_best_catboost_model(best_trial_catboost, train_df, test_df)

