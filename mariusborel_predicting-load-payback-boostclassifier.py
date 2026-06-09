import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import roc_auc_score, make_scorer, roc_curve, confusion_matrix

# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, 
                              StackingClassifier, BaggingClassifier,VotingClassifier)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv
from catboost import CatBoostClassifier, Pool

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   LabelEncoder, OneHotEncoder, FunctionTransformer)

import optuna
from optuna.samplers import TPESampler
import plotly.express as px

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
train_raw.head()


test_raw = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
test_raw.head()


sub_raw = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv', index_col='id')
sub_raw.head()


orig_raw = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
orig_raw.head()


orig_raw = orig_raw[train_raw.columns.tolist()]

orig_raw.head()


target = 'loan_paid_back'


pd.DataFrame(
    {'NA in train':train_raw.isna().sum(), 
     'NA in test':test_raw.isna().sum(), 
     'NA in orig':orig_raw.isna().sum()}
).drop(target, axis=0).astype('int').style.background_gradient(cmap='YlGn', axis=1)


train_comb = pd.concat([train_raw, orig_raw], ignore_index=True)
train_comb


num_description = train_raw.describe().T
num_description['nulls'] = train_raw.isna().sum()
num_description


cat_description = train_raw.describe(exclude='number').T
cat_description['nulls'] = train_raw.isna().sum()
cat_description


# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1 = df_1[num_features].copy()
    adv_df_2 = df_2[num_features].copy()


    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1 = adv_df_1.assign(adv=1)
    adv_df_2 = adv_df_2.assign(adv=0)


    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=64)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv


    cv = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                        random_state = 64)
    xgb_model = XGBClassifier(max_depth=3,
                              learning_rate = 0.1,
                              n_estimators = 100,
                              objective = 'binary:logistic',
                              random_state = 64)

    # Get the cross validation scores
    adv_scores = []
    for i, _ in enumerate(cv.split(X_combined, y_combined)):
        X_train, X_valid, y_train, y_valid = train_test_split(X_combined, 
                                                              y_combined, 
                                                              test_size=0.3)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict_proba(X_valid)[:,1]
        score = roc_auc_score(y_valid, y_pred)
        adv_scores.append(score)

#         print(f"Fold {i+1} AUC Score: {score:.5f}")

    #Plot the roc_curve
    mean_auc = np.mean(adv_scores)
    fpr, tpr, _ = roc_curve(y_valid, y_pred)
    plt.plot(fpr, tpr, label = 'roc_curve (AUC = %0.4f)' % mean_auc)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', label = 'Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'roc_curve {name_1} vs {name_2}', weight='bold')
    plt.legend()


num_features = list(num_description.index)[:-1]
plt.figure(figsize=(18,5))
plt.subplot(1,4,1)
adversarial_validation(train_raw, test_raw, 'train', 'test')
plt.subplot(1,4,2)
adversarial_validation(test_raw, orig_raw, 'train', 'original')
plt.subplot(1,4,3)
adversarial_validation(test_raw, orig_raw, 'test', 'original')
plt.subplot(1,4,4)
adversarial_validation(train_comb, test_raw, 'train_comb', 'test')


plt.figure(figsize=(12, 4))
for n, df in enumerate([train_raw, test_raw, orig_raw]):
    color =  ['orange', 'steelblue', 'red']
    for i in enumerate(df.loc[:, num_features]):
        ax = plt.subplot(1,5,i[0]+1)
        sns.kdeplot(data=df.loc[:, num_features],
                           x=i[1],
                        fill=True,
                    color=color[n]
                      )
        plt.xlim(train_raw.loc[:, num_features][i[1]].min(),train_raw.loc[:, num_features][i[1]].max())
        plt.title(i[1], fontsize=12)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=30, fontsize=7)
        plt.suptitle('Kde plots of the three datasets', weight='bold', fontsize=14)
        plt.tight_layout()
plt.show()


train_raw[target].value_counts().plot.pie(
    autopct='%.2f%%', 
    title='Proportion of loan_status', 
    explode=[0.05, 0.05]
)
plt.ylabel('')
plt.show()


plt.figure(figsize=(14,6))
for f, feat in enumerate(num_description.index, start=1):
    plt.subplot(2,3,f)
    sns.boxenplot(train_raw, x=target, y=feat)
    # if f !=1:
    #     plt.xlabel('')
    if f < 8:
        plt.legend([])
plt.tight_layout(pad=2, h_pad=2, w_pad=2)


def box_plot_by_category(cat_feat):
    plt.figure(figsize=(14,10))
    for f, feat in enumerate(num_description.index, start=1):
        plt.subplot(4,2,f)
        sns.boxenplot(train_raw, x=target, y=feat, hue=cat_feat)
        if f !=1:
            plt.xlabel('')
        if f < 8:
            plt.legend([])
    plt.tight_layout()


box_plot_by_category('gender')


box_plot_by_category('loan_purpose')


box_plot_by_category('grade_subgrade')


plt.figure(figsize=(12,3))
for f, feat in enumerate(num_description.index[:-1], start=1):
    plt.subplot(1,5,f)
    sns.kdeplot(train_raw, x=feat, hue=target, fill=True)
    if f !=1:
        plt.ylabel('')
    if f > 1:
        plt.legend([])
plt.tight_layout()


# Get the train data
train_data = train_raw.copy(deep=False)
orig_data = orig_raw.copy(deep=False)
# Get the train_target
train_target = train_data.pop(target)
orig_target = orig_data.pop(target)


mask = np.triu(np.ones(7), k=1)

plt.figure(figsize=(14, 5))
plt.subplot(131)
num_corr = train_data.corr(numeric_only=True)
sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='Oranges', cbar=False)
plt.title('Correlation in train set', color='orange')

plt.subplot(132)
num_corr = test_raw.corr(numeric_only=True)
sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='Greens', cbar=False)
plt.yticks([])
plt.title('Correlation in test set', color='green')

plt.subplot(133)
num_corr = orig_data.corr(numeric_only=True)
sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.yticks([])
plt.title('Correlation in original set', color='blue')

plt.show()


plt.figure(figsize=(12, 5))
for f, feat in enumerate(cat_description.index, start=1):
    if f < 6:
        plt.subplot(2,3,f)
        train_raw[feat].value_counts().plot.barh(color='steelblue')
plt.tight_layout(pad=2, h_pad=4, w_pad=4)


def cross_counting(feat, a, b):
    plt.figure(figsize=(a, b))
    plt.subplot(1,3,1)
    Driving_License_Response_ctab = pd.crosstab(train_raw[feat], train_raw[target])
    sns.heatmap(Driving_License_Response_ctab, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(f'Count of categories', fontsize=10)
    plt.subplot(1,3,2)
    Driving_License_Response_ctab = pd.crosstab(train_raw[feat], train_raw[target], 
                                                normalize='index')
    sns.heatmap(Driving_License_Response_ctab, annot=True, cmap='Reds', fmt='.3f', cbar=False)
    plt.yticks([])
    plt.title(f'%tage by {target} categories', fontsize=10)
    plt.ylabel('')
    plt.subplot(1,3,3)
    Driving_License_Response_ctab = pd.crosstab(train_raw[feat], train_raw[target], 
                                                normalize='columns', margins=True, margins_name='TOTALS')
    sns.heatmap(Driving_License_Response_ctab, annot=True, cmap='Greens', fmt='.3f', cbar=False)
    plt.ylabel('')
    plt.yticks([])
    plt.title(f'%tage by {feat} categories', fontsize=10)
    plt.suptitle(f'Loan Status Distributions with respect to {feat}', fontsize=14)
    plt.show()


cross_counting(feat='gender', a=12, b=4)


cross_counting(feat='marital_status', a=12, b=4)


cross_counting(feat='education_level', a=12, b=4)


cross_counting(feat='employment_status', a=12, b=4)


cross_counting(feat='loan_purpose', a=12, b=4)


cross_counting(feat='grade_subgrade', a=12, b=10)


class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        # df['Grade'] = df['grade_subgrade'][0]
        # df['Education_Grade'] = df['education_level'] + '-' + df['Grade']
        # ----
        return df

class Feature_Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=RobustScaler()):
        self.scaler = scaler if scaler else MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.columns = None
        self.encoded_feature_names = None

    def fit(self, df, y=None):
        df = df.copy()
        # Identify categorical columns
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

        # Fit encoder and scaler
        self.encoder.fit(df[self.cat_cols])
        encoded = self.encoder.transform(df[self.cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=df.index)

        full_df = pd.concat([df[self.num_cols], encoded_df], axis=1)
        self.columns = full_df.columns
        self.scaler.fit(full_df)
        return self

    def transform(self, df):
        df = df.copy()
        encoded = self.encoder.transform(df[self.cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=df.index)

        full_df = pd.concat([df[self.num_cols], encoded_df], axis=1)
        full_df = full_df.reindex(columns=self.columns, fill_value=0)
        scaled = self.scaler.transform(full_df)
        return pd.DataFrame(scaled, columns=self.columns, index=df.index)

    def inverse_transform(self, df):
        df = pd.DataFrame(self.scaler.inverse_transform(df), columns=self.columns, index=df.index)
        # Note: inverse_transform of OneHotEncoder is not always perfect if columns were dropped or reindexed
        encoded_df = df[self.encoder.get_feature_names_out(self.cat_cols)]
        decoded = self.encoder.inverse_transform(encoded_df)
        decoded_df = pd.DataFrame(decoded, columns=self.cat_cols, index=df.index)
        return pd.concat([df[self.num_cols], decoded_df], axis=1)


prep_pipeline = make_pipeline(Feature_Eng(), Feature_Scaler())

prep_pipeline


df_1 = train_raw.copy()
df_2 = test_raw.copy()

df_p = prep_pipeline.fit_transform(df_1)
display(df_p.head(3))


# Should we use the original dataset?
use_original = False

if use_original:
    # Get the train data
    train_data = train_comb.copy(deep=False)
    print('The original data is included in the train set.')
else:
    train_data = train_raw.copy(deep=False)
    print('The original data is not included in the train set.')
# Get the train_target
train_target = train_data.pop(target)


seed = 8

Models = [
          ('lgb_clf', LGBMClassifier(verbose=-1)),
          ('cat_clf', CatBoostClassifier(verbose=False)),
          ('gbc_clf',GradientBoostingClassifier()), 
          ('xgb_clf', XGBClassifier()),
          ('rfc_clf',RandomForestClassifier()),
         ]

'''Dataset without any new columns'''
n_splits = 6
scores = [] # Empty cross validation score list
models = [] # Empty list of models

my_cv = KFold(
    n_splits=n_splits, 
    shuffle=True, 
    random_state=seed)

for model_name, model in Models:
    # Define the model pipeline
    model_pipe = make_pipeline(
        Feature_Eng(),
        Feature_Scaler(), 
        model)

    # Define X and y
    X = train_data
    y = train_target
    
    # Cross validation
    cv_score = cross_val_score(
        model_pipe, 
        X=train_data, 
        y=train_target, 
        cv=my_cv, 
        scoring='roc_auc')
    
    scores.append(cv_score) # Add the scores to the scores list
    models.append(model_name) # Add the model to the list of models
    
    scores_df = pd.DataFrame(
        scores, 
        columns=[f'cv_{n+1}' for n in range(n_splits)], 
        index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.iloc[:-1, :-1].style.background_gradient(cmap='RdYlGn', axis=0)
         # .highlight_min(axis=0, color='yellow')
         .format('{:.5f}')
         .set_properties(**{'font-size': '13pt', 'weight': 'bold'}))
       )


palette = 'Dark2'

fig = plt.figure(figsize=(8, 5))
gs = GridSpec(2, 2, height_ratios=[2, 5])

ax0 = fig.add_subplot(gs[0, :])
ax0 = sns.lineplot(scores_df.T.iloc[:-1, :], palette=palette, marker='o')
ax0.set_ylabel('Scores')
ax0.set_title('CV scores on various models', fontsize=12)
ax0.legend([])

ax1 = fig.add_subplot(gs[1:, :])
ax1 = sns.violinplot(scores_df.T.iloc[:-1, :], palette=palette, saturation=0.3)
ax1 = sns.swarmplot(scores_df.T.iloc[:-1, :], palette=palette)
ax1.set_ylabel('Scores')

plt.tight_layout(pad=1, h_pad=3, w_pad=5)


%%time

# The splitter
optuna_cv = KFold(n_splits=4, shuffle=True, random_state=33)

# Define the objective function
def objective(trial):
    param_grid = {
        "iterations": trial.suggest_int("iterations", 100, 4000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "eval_metric": "AUC",
        # "verbose": 0,
        # "task_type": "GPU",  # Use "CPU" if GPU is not available
        }
    
    # Define the model by unpacking the chosen parameters
    model = make_pipeline(Feature_Eng(), Feature_Scaler(), CatBoostClassifier(**param_grid, verbose=0))
    # Get and return the score
    # model.fit(X_train, y_train)
    scores = cross_val_score(model, train_data, train_target, cv=optuna_cv, n_jobs=-1, 
                        scoring = 'roc_auc')
    
    return scores.mean()


def run_optuna(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=72000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        # best_study_params = {'iterations': 2750, 
        #                      'learning_rate': 0.029156920584902463,
        #                      'depth': 10, 
        #                      'l2_leaf_reg': 9.397319735876648, 
        #                      'border_count': 189, 
        #                      'random_strength': 8.542797035303309, 
        #                      'scale_pos_weight': 0.5673559738459675, 
        #                      'bootstrap_type': 'MVS', 
        #                      'grow_policy': 'Lossguide',
        #                      'eval_metric': 'AUC'}
        
        best_study_params = {}
    
    print(f"Best parameters: {best_study_params}")
    return best_study_params

best_params = run_optuna(n_trials=20)


clf_pipe = make_pipeline(
    Feature_Eng(), 
    Feature_Scaler(), 
    CatBoostClassifier(
        **best_params, 
        verbose=200, 
        eval_fraction=0.1, 
        early_stopping_rounds=200, 
        eval_metric='AUC')
)

clf_pipe


ns=5

X = train_data
y = train_target

splitter = KFold(n_splits=ns, shuffle=True, random_state=84)

for f, (tr_ind, va_ind) in enumerate(splitter.split(X, y), start=1):
    print('\n'+17*'= ' + f'\033[93mFitting Fold_{f}\033[0m' + 17*' =')
    # Split data and target in folds
    X_tr, X_va = X.iloc[tr_ind],  X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind],  y.iloc[va_ind]
    # Define the predictor
    predictor = make_pipeline(
        Feature_Eng(), 
        Feature_Scaler(), 
        CatBoostClassifier(
            **best_params, 
            verbose=200, 
            eval_fraction=0.1, 
            early_stopping_rounds=200, 
            eval_metric='AUC')
    )
    # Fit the predictor
    predictor.fit(X_tr, y_tr)
    # Predict n the validation data
    y_va_proba = predictor.predict_proba(X_va)[:, 1]
    # Score the predictions
    score = roc_auc_score(y_va, y_va_proba)
    # False vs True predictions rates
    fpr, tpr, _ = roc_curve(y_va, y_va_proba)
    plt.plot(fpr, tpr, label='Fold_{} auc_score: {:.6f}'.format(f, score))
    plt.plot([0,1], [0,1], linestyle='--', color='green')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
plt.show()


# Let's fit the model to the entire train set
clf_pipe.fit(train_data, train_target)


# Separate original data and target
orig_data = orig_raw.copy(deep=False)
orig_target = orig_data.pop(target)

# predictions on orig data
orig_pred = clf_pipe.predict(orig_data)

# Plot the confusion matrices
plt.figure(figsize=(8,4))
conf_matrix = confusion_matrix(orig_target, orig_pred)
plt.subplot(121)
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.subplot(122)
conf_matrix_norm = confusion_matrix(orig_target, orig_pred, normalize='true')
sns.heatmap(conf_matrix_norm, annot=True, fmt='.3f', cbar=False, cmap='Greens')
plt.suptitle('Confusion matrix on orig_data: auc_score {:.4f}'.format(roc_auc_score(orig_target, orig_pred)))
plt.show()


test_pred = clf_pipe.predict_proba(test_raw)

sub_df = sub_raw.copy()
sub_df[target] = test_pred[:, 1]
sub_df.head(10)


pd.Series(sub_df.iloc[:, -1]).plot.hist(
    bins=50, 
    figsize=(8, 4), 
    title='Distribution of Predicted Test Probabilities',
    color='steelblue'
)
plt.xlabel('Predicted Probalities');


sub_df.to_csv('submission.csv', index=True)

print('ğŸ�¾ The submission file is ready ğŸ¥‚')

