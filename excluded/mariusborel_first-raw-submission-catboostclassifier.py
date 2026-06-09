import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
# from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, RFE, chi2
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

import optuna
from optuna.samplers import TPESampler

from yellowbrick.regressor import ResidualsPlot, PredictionError

import shap

import warnings
warnings.filterwarnings('ignore')

# Set Seaborn theme with dark grid
sns.set_theme(style="white", palette="tab20_r", font_scale=0.8)

# Update matplotlib parameters
plt.rcParams.update({
    'axes.facecolor': '#222222', 
    'figure.facecolor': '#222222', 
    'text.color': '#FFF9C4',   
    'axes.labelcolor': '#FFF9C4',    
    'xtick.color': '#FFF9C4',      
    'ytick.color': '#FFF9C4',        
    'grid.color': '#444444',         
    'axes.edgecolor': 'white'        
})

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
orig_raw = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

target = 'loan_paid_back'

train.head()


orig_raw.head()


orig = orig_raw[train.columns.tolist()]

orig.head()


train.info()


num_feats = test.select_dtypes(include='number').columns.tolist()
cat_feats = test.select_dtypes(exclude='number').columns.tolist()

print(f'numeric features: {num_feats}, \n\nCategory features: {cat_feats}')


#### plt.figure(figsize=(15, 4))
for n, feat in enumerate(num_feats, start=1):
    plt.subplot(1, 5, n)
    sns.kdeplot(train, x=feat, hue=target, fill=True)
    plt.suptitle(f'Distribution of numeric features grouped by {target}', fontsize=14, color='white')
    
plt.tight_layout(pad=2, h_pad=2, w_pad=2)


for cat_feat in cat_feats[:-1]:
    plt.figure(figsize=(15, 3.6))
    for n, feat in enumerate(num_feats, start=1):
        plt.subplot(1, 5, n)
        sns.violinplot(train, x=feat, y=cat_feat)
        if n > 1:
            plt.ylabel('')
            plt.yticks([])
        # plt.title(feat, color='orange', fontsize=8)
    plt.suptitle(f'Distribution of numeric features grouped by {cat_feat}', fontsize=14, color='white')
    plt.tight_layout()


plt.figure(figsize=(15, 11))
for n, feat in enumerate(num_feats, start=1):
    plt.subplot(2, 3, n)
    sns.violinplot(train, x=feat, y=cat_feats[-1])
    plt.title(feat, color='orange', fontsize=12)
plt.title(f'{feat} by {cat_feat}')
plt.tight_layout()


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(5, 2, width_ratios=[1, 1])

for c, col in enumerate(cat_feats, start=0):
    if col != 'grade_subgrade':
        ax0 = fig.add_subplot(gs[c, 0])
        train[col].value_counts().plot.barh(color='darkgrey')
        for count in ax0.containers:
            ax0.bar_label(count, label_type='center')
        plt.title(col, fontsize=10)
        plt.ylabel('')

ax1 = fig.add_subplot(gs[:, 1])
train[col].value_counts().plot.barh(color='darkgrey')
plt.title(col, fontsize=10)
for count in ax1.containers:
    ax1.bar_label(count, label_type='center')
plt.ylabel('')
plt.suptitle('Counts of elements within cat_features', fontsize=14, color='white')
plt.tight_layout(pad=1, h_pad=1, w_pad=5)
plt.show()


target_counts = train[target].value_counts()

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

ax0 = fig.add_subplot(gs[:, :-1])
# ax0 = target_counts.plot.bar(color=['#e86100', '#da1d81'])
ax0 = target_counts.plot.bar()
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = target_counts.plot.pie(autopct='%.1f%%',
                            shadow = True,
                            radius=1.2,
                            explode=[0.05, 0.1],
                            startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.38, ax=ax1)
ax1.set_ylabel('')
plt.suptitle('Counts of target classes in train data')
plt.tight_layout()


# Create a grade feature from grade_subgrade
for df in [train, test]:
    df['grade'] = df['grade_subgrade'].apply(lambda x: x[0])
    df['interest_amount'] = df['loan_amount']*df['interest_rate']/100
    df['int_amnt/ann_income'] = df['annual_income']/df['interest_amount']

# Order of the ordinal features
educ_level_order = ['PhD', "Master's", "Bachelor's", 'High School', 'Other' ]
grade_subgrade_order = sorted(train['grade_subgrade'].unique())#.tolist())
grade_order = sorted(train['grade'].unique().tolist())

custom_categories = [
    educ_level_order,
    grade_subgrade_order,
    grade_order
]


num_feats = test.select_dtypes(include='number').columns.tolist()
cat_feats = test.select_dtypes(exclude='number').columns.tolist()

print(f'numeric features: {num_feats}, \n\nCategory features: {cat_feats}')


use_original = False

if use_original:
    train_ = pd.concat([train, orig], ignore_index=True)
else:
    train_ = train

X = train_.copy()
y = X.pop(target)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=64)

[d.shape for d in [X_train, X_valid]]


X.head()


# Ordinal cat features
ord_cat_feat = ['education_level', 'grade_subgrade', 'grade']
# Nominal cat features
ohe_cat_feat = ['gender', 'marital_status', 'employment_status', 'loan_purpose']


scaler = MinMaxScaler()
# scaler = RobustScaler()
ohe_enc = OneHotEncoder()
# encoder = LabelEncoder()
ord_enc = OrdinalEncoder(categories=custom_categories)

features_trans = make_column_transformer(
 #   (scaler, num_feats),
    (ohe_enc, ohe_cat_feat),
    (ord_enc, ord_cat_feat),
    remainder='passthrough', 
    sparse_threshold=0
)


def objective(trial):
    params = {
    "iterations": trial.suggest_int("iterations", 10000, 20000, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    "depth": trial.suggest_int("depth", 2, 8),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    "border_count": trial.suggest_int("border_count", 32, 255),
    "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
    "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
    "eval_metric": "AUC",
    "verbose": 0,
    # "task_type": "GPU",  # Use "CPU" if GPU is not available
    }

    model = make_pipeline(
        features_trans,
        CatBoostClassifier(**params)
    )

    if cv_scorer:
        # Cross-validation (recommended)
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return scores.mean()
    else:
        # Alternatively
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
        return score


def Run_Pass_lgbm_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=72000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        best_study_params = {'iterations': 2750, 
                             'learning_rate': 0.029156920584902463,
                             'depth': 10, 
                             'l2_leaf_reg': 9.397319735876648, 
                             'border_count': 189, 
                             'random_strength': 8.542797035303309, 
                             'scale_pos_weight': 0.5673559738459675, 
                             'bootstrap_type': 'MVS', 
                             'grow_policy': 'Lossguide',
                             'eval_metric': 'AUC'
                            }
    
    print(f"\nBest parameters: {best_study_params}")
    return best_study_params


# Decide how optuna is scored
cv_scorer=False

# Run the optimization
cat_best_params = Run_Pass_lgbm_study(n_trials=50)


cat_params = cat_best_params

model = make_pipeline(features_trans,
                      CatBoostClassifier(
                          **cat_params, verbose=500, 
                          eval_fraction=0.1, 
                          early_stopping_rounds=200,
                          eval_metric='AUC'
                      ))

model.fit(X_train, y_train)


ns=5

splitter = KFold(n_splits=ns, shuffle=True, random_state=84)

for f, (tr_ind, va_ind) in enumerate(splitter.split(X, y), start=1):
    print('\n'+17*'= ' + f'\033[93mFitting Fold_{f}\033[0m' + 17*' =')
    # Split data and target in folds
    X_tr, X_va = X.iloc[tr_ind],  X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind],  y.iloc[va_ind]
    # Define the predictor
    predictor = make_pipeline(features_trans, 
                              CatBoostClassifier(
                                  **cat_params, 
                                  verbose=500, 
                                  eval_fraction=0.1, 
                                  early_stopping_rounds=200,
                                  eval_metric='AUC'
                              ))
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
   


pred_proba = model.predict_proba(test)

pred_proba


subm[target] = pred_proba[:, 1]

subm.head()


subm.to_csv('submission.csv', index=False)

print('\033[92mThe file is ready for submission\033[0m')

