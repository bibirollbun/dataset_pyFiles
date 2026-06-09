import os
import math
import shap
import optuna
import scipy

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import chi2_contingency

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")


!pip install --upgrade seaborn

import seaborn as sns
sns.set(color_codes=True)


def num_var_distribution_float(df,
                               title: str,
                               x1: str,
                               y1: str,
                               x1_label: str,
                               y1_label: str,
                               x2_label: str,
                               y2_label: str):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 16, fontweight ='bold')

    # Figure 1: box-plot
    dir_order = ['train', 'test']
    my_pal = {'train': 'orange', 'test': 'royalblue'}
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           palette = my_pal,
                           ax = axes[0])
    axes[0].set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians['Med'] = medians['Med'].round(2)
    medians['Tick'] = range(len(medians))
    
    medians['Cat'] = 0
    for i in range(len(medians)):
        if medians.loc[i, x1] == 'train':
            medians.loc[i, 'Cat'] = 0
        if medians.loc[i, x1] == 'test':
            medians.loc[i, 'Cat'] = 1
    
    medians = medians.sort_values(['Cat'])
    ticks = list(medians['Tick'])
    medians = list(medians['Med'])
    vertical_offset = [median * 0.025 for median in medians]
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'black', 
                      weight = 'semibold')
    
    
    # Figure 2: distplots
    kde_1 = sns.distplot(a = df.loc[df[x1] == 'train', y1],
                         kde_kws = {'color': 'orange', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'train',
                         ax = axes[1])
    kde_2 = sns.distplot(a = df.loc[df[x1] == 'test', y1],
                         kde_kws = {'color': 'royalblue', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'test',
                         ax = axes[1])
    
    axes[1].set_xlabel(x2_label, fontsize = 14, fontweight ='bold')
    axes[1].set_ylabel(y2_label, fontsize = 14, fontweight ='bold')
    
    
    plt.plot()


def corr_plot(df_1, df_2, title):

    figure, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 18, fontweight ='bold')
     
    sns.heatmap(df_1, 
                annot = True, 
                vmin = -1, 
                vmax = 1, 
                center = 0, 
                cmap = 'coolwarm',
                linewidths = 3, 
                linecolor = 'black',
                ax = ax[0])
    
    sns.heatmap(df_2, 
                annot = True, 
                vmin = -1, 
                vmax = 1, 
                center = 0, 
                cmap = 'coolwarm',
                xticklabels = True,
                yticklabels = False,
                linewidths = 3, 
                linecolor = 'black',
                ax = ax[1])
    
    ax[0].set_title("train", fontsize = 16)
    ax[1].set_title("test", fontsize = 16)
    
    plt.show()


def map_fico_tier(score):
    """Maps a credit score to its corresponding FICO tier."""
    if score >= 800:
        return 'Exceptional'
    elif score >= 740:
        return 'Very Good'
    elif score >= 670:
        return 'Good'
    elif score >= 580:
        return 'Fair'
    else: # Below 580
        return 'Poor'

def map_vantage_tier(score):
    """Maps a credit score to its corresponding VantageScore tier."""
    if score >= 781:
        return 'Excellent'
    elif score >= 661:
        return 'Good'
    elif score >= 601:
        return 'Fair'
    elif score >= 500:
        return 'Poor'
    else: # Below 500
        return 'Very Poor'


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df_orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


df_1 = df_train.drop(columns = ['loan_paid_back'])
df_1['data_type'] = 'train'

df_2 = df_test.copy()
df_2['data_type'] = 'test'

df = pd.concat([df_1, df_2], ignore_index = True)


df_train.info()


df_train.head(3)


df_train_no_id = df_train.drop(columns = ['id'])
df_train_no_id.drop_duplicates(keep = 'first', inplace = True, ignore_index = True)

print('Number of duplicates in the df_train: ', len(df_train) - len(df_train_no_id))





df_test.info()


df_test.head(3)


figure, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 4))

count_plot = sns.countplot(df_train, x = "loan_paid_back", stat = "percent")
count_plot.bar_label(count_plot.containers[0], fontsize=10)

axes.set_title('Loan Paid Back Distribution', fontsize = 12, fontweight = 'bold')
axes.set_xlabel('Loan Paid Back', fontsize = 10, fontweight ='bold')
axes.set_ylabel('Percent', fontsize = 10, fontweight ='bold')

plt.show()


df_train['loan_paid_back'].value_counts()


num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']


df_train[num_features].describe()


df_test[num_features].describe()


num_var_distribution_float(df = df,
                           title = 'Annual Income Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'annual_income',
                           x1_label = 'Data Type',
                           y1_label = 'Annual Income',
                           x2_label = 'Annual Income',
                           y2_label = 'Density')


num_var_distribution_float(df = df,
                           title = 'Debt to Income Ratio Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'debt_to_income_ratio',
                           x1_label = 'Data Type',
                           y1_label = 'Debt to Income ratio',
                           x2_label = 'Debt to Income Ratio',
                           y2_label = 'Density')


num_var_distribution_float(df = df,
                           title = 'Credit Score Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'credit_score',
                           x1_label = 'Data Type',
                           y1_label = 'Credit Score',
                           x2_label = 'Credit Score',
                           y2_label = 'Density')


num_var_distribution_float(df = df,
                           title = 'Loan Amount Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'loan_amount',
                           x1_label = 'Data Type',
                           y1_label = 'Loan Amount',
                           x2_label = 'Loan Amount',
                           y2_label = 'Density')


num_var_distribution_float(df = df,
                           title = 'Interest Rate Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'interest_rate',
                           x1_label = 'Data Type',
                           y1_label = 'Interest Rate',
                           x2_label = 'Interest Rate',
                           y2_label = 'Density')


corr_plot(df_1 = df_train[num_features].corr(method = 'spearman'), 
          df_2 = df_test[num_features].corr(method = 'spearman'), 
          title = "Spearman's rank correlation")


categorical_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


df_train[categorical_features].describe()


df_test[categorical_features].describe()


df_cramers_train = pd.DataFrame(columns = categorical_features, 
                                index = categorical_features, 
                                dtype = np.float32)
df_cramers_test = pd.DataFrame(columns = categorical_features, 
                               index = categorical_features,
                               dtype = np.float32)
for i in range(len(categorical_features)):
    var_i = categorical_features[i]
    for j in range(len(categorical_features)):
        var_j = categorical_features[j]
        
        df_temp_train = pd.crosstab(df_train[var_i], df_train[var_j])
        chi2_train, _, _, _ = chi2_contingency(df_temp_train)
        df_cramers_train.loc[var_i, var_j] = math.sqrt(chi2_train / (df_temp_train.values.sum() * min(df_temp_train.shape[0]-1, df_temp_train.shape[1]-1)))
        
        df_temp_test = pd.crosstab(df_test[var_i], df_test[var_j])
        chi2_test, _, _, _ = chi2_contingency(df_temp_test)
        df_cramers_test.loc[var_i, var_j] = math.sqrt(chi2_test / (df_temp_test.values.sum() * min(df_temp_test.shape[0]-1, df_temp_test.shape[1]-1)))


corr_plot(df_1 = df_cramers_train,
          df_2 = df_cramers_test,
          title = "Cramers' V correlation coefficients")


for data in [df_train, df_test]:
    data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']


for data in [df_train, df_test]:
    data['grade'] = data['grade_subgrade'].apply(lambda x: x[0])
    data['subgrade'] = data['grade_subgrade'].apply(lambda x: x[1])


for data in [df_train, df_test]:
    data['credit_score_FICO_tier'] = data['credit_score'].apply(map_fico_tier)
    data['credit_score_Vantage_tier'] = data['credit_score'].apply(map_vantage_tier)


ord_encoded_features = categorical_features + ['grade', 'subgrade'] + ['credit_score_FICO_tier', 'credit_score_Vantage_tier']

enc = OrdinalEncoder()
enc.fit(df_train[ord_encoded_features])

df_train[ord_encoded_features] = enc.transform(df_train[ord_encoded_features])
df_test[ord_encoded_features] = enc.transform(df_test[ord_encoded_features])


for col in ord_encoded_features:
    df_train[col] = df_train[col].astype('int32')
    df_test[col] = df_test[col].astype('int32')


for data in [df_train, df_test]:
    data['annual_income'] = np.log(data['annual_income'].values)


df_train['loan_paid_back'] = df_train['loan_paid_back'].astype('int')


predictors = num_features + categorical_features
predictors_ext = predictors + ['grade', 'subgrade'] + ['credit_score_FICO_tier', 'credit_score_Vantage_tier'] + ['loan_to_income_ratio']

target = 'loan_paid_back'


X_train, X_val, y_train, y_val = train_test_split(df_train[predictors],
                                                  df_train[target],
                                                  train_size = 0.9,
                                                  random_state = 42)


alg = XGBClassifier(n_estimators = 1000,
                    objective = 'binary:logistic',
                    eval_metric = 'auc')

alg.fit(X_train[predictors], y_train, 
        eval_set = [(X_val[predictors], y_val)],
        early_stopping_rounds = 20,
        verbose = 25)


print('Best iteration:', alg.best_iteration)
print('----------')
print('Best AUROC:', alg.best_score)


shap.initjs()


explainer = shap.TreeExplainer(alg)
shap_values = explainer.shap_values(X_val, y_val)


shap.summary_plot(shap_values, X_val, plot_type = "bar")


def objective(trial):
    params = {
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log = True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.25, 1.0, step = 0.01),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0, step = 0.01),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "max_leaves": trial.suggest_int("max_leaves", 10, 50)}
    
    kf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 42)
    a = kf.split(X = df_train[predictors_ext], y = df_train[target])
    
    oof_pred = np.zeros(len(df_train))
    for i, (train_index, test_index) in enumerate(a): 
        X_train = df_train.loc[train_index, :].copy()
        X_test = df_train.loc[test_index, :].copy()

        ## train / validation split
        X_train, X_val, y_train, y_val = train_test_split(X_train[predictors_ext],
                                                          X_train[target],
                                                          stratify = X_train[target],
                                                          train_size = 0.95,
                                                          random_state = 42)
        X_train[target] = y_train

        # target and frequency encodings of the categorical variables
        te_columns = ord_encoded_features  + ['debt_to_income_ratio', 'credit_score']
        for predictor in te_columns:
            global_mean = X_train[target].mean()
            global_length = len(X_train)
            
            df_pred = X_train.groupby(by = predictor).agg(
                FE = (target, lambda x: len(x) / global_length),
                TE = (target, lambda x: (len(x) * np.mean(x) + 10 * global_mean) / (10 + len(x)))
            ).reset_index()
            df_pred.rename(columns = {'TE': predictor + '_mean',
                                      'FE': predictor + '_freq'}, inplace = True)

            X_train = X_train.merge(df_pred, on = predictor, how = 'left')
            X_val = X_val.merge(df_pred, on = predictor, how = 'left')
            X_test = X_test.merge(df_pred, on = predictor, how = 'left')
                
            X_val[predictor + '_mean'] = X_val[predictor + '_mean'].fillna(global_mean)
            X_val[predictor + '_freq'] = X_val[predictor + '_freq'].fillna(0)

            X_test[predictor + '_mean'] = X_test[predictor + '_mean'].fillna(global_mean)
            X_test[predictor + '_freq'] = X_test[predictor + '_freq'].fillna(0)

        predictors_new = num_features + ['loan_to_income_ratio'] + ord_encoded_features +\
                         [pred + '_mean' for pred in te_columns] + [pred + '_freq' for pred in te_columns]
        eval_set = (X_val[predictors_new], y_val)
        
        ## XGBoost
        alg = XGBClassifier(**params,
                            learning_rate = 0.025,
                            n_estimators = 100000,
                            objective = 'binary:logistic',
                            eval_metric = 'auc')

        alg.fit(X_train[predictors_new], y_train, 
                eval_set = [eval_set],
                early_stopping_rounds = 200,
                verbose = 0)
        oof_pred[test_index] = alg.predict_proba(X_test[predictors_new])[:, 1]

    return roc_auc_score(df_train[target], oof_pred)


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction = 'maximize', study_name = 'xgboost')
study.optimize(func = objective, 
               n_trials = 50,
               n_jobs = 2,
               gc_after_trial = False,
               show_progress_bar = True)


print('Best set of hyper-parameters:', study.best_params)
print('---------')
print('Best AUROC:', study.best_value)


kf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 42)
a = kf.split(X = df_train[predictors_ext], y = df_train[target])


oof_pred = np.zeros(len(df_train))
test_pred = np.zeros(len(df_test))
for i, (train_index, test_index) in enumerate(a): 
    X_train = df_train.loc[train_index, :].copy()
    X_test = df_train.loc[test_index, :].copy()
    df_test_copy = df_test.copy()

    ## train / validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train[predictors_ext],
                                                      X_train[target],
                                                      stratify = X_train[target],
                                                      train_size = 0.95,
                                                      random_state = 42)
    X_train[target] = y_train

    # target and frequency encodings of the categorical variables
    te_columns = ord_encoded_features  + ['debt_to_income_ratio', 'credit_score']
    for predictor in te_columns:
        global_mean = X_train[target].mean()
        global_length = len(X_train)
        
        df_pred = X_train.groupby(by = predictor).agg(
            FE = (target, lambda x: len(x) / global_length),
            TE = (target, lambda x: (len(x) * np.mean(x) + 10 * global_mean) / (10 + len(x)))
        ).reset_index()
        df_pred.rename(columns = {'TE': predictor + '_mean',
                                  'FE': predictor + '_freq'}, inplace = True)
    
        X_train = X_train.merge(df_pred, on = predictor, how = 'left')
        X_val = X_val.merge(df_pred, on = predictor, how = 'left')
        X_test = X_test.merge(df_pred, on = predictor, how = 'left')
        df_test_copy = df_test_copy.merge(df_pred, on = predictor, how = 'left')
            
        X_val[predictor + '_mean'] = X_val[predictor + '_mean'].fillna(global_mean)
        X_val[predictor + '_freq'] = X_val[predictor + '_freq'].fillna(0)

        X_test[predictor + '_mean'] = X_test[predictor + '_mean'].fillna(global_mean)
        X_test[predictor + '_freq'] = X_test[predictor + '_freq'].fillna(0)

        df_test_copy[predictor + '_mean'] = df_test_copy[predictor + '_mean'].fillna(global_mean)
        df_test_copy[predictor + '_freq'] = df_test_copy[predictor + '_freq'].fillna(0)

    predictors_new = num_features + ['loan_to_income_ratio'] + ord_encoded_features +\
                     [pred + '_mean' for pred in te_columns] + [pred + '_freq' for pred in te_columns]
    eval_set = (X_val[predictors_new], y_val)
        
    ## XGBoost
    alg = XGBClassifier(**study.best_params,
                        learning_rate = 0.025,
                        n_estimators = 100000,
                        objective = 'binary:logistic',
                        eval_metric = 'auc')

    alg.fit(X_train[predictors_new], y_train,
            eval_set = [eval_set],
            early_stopping_rounds = 200,
            verbose = 0)
    oof_pred[test_index] = alg.predict_proba(X_test[predictors_new])[:, 1]
    test_pred += alg.predict_proba(df_test_copy[predictors_new])[:, 1]


auc = roc_auc_score(df_train[target], oof_pred)
print("8-Fold CV AUROC: ", auc)


test_pred = test_pred / 8.


submission = pd.DataFrame({'id': df_test['id'], 'loan_paid_back': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index = False)




