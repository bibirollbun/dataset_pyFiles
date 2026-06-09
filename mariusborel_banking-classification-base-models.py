import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer, roc_curve
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Accent_r", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222', 
    'figure.facecolor': '#222222', 
    'text.color': '#ff8c00',   
    'axes.labelcolor': '#82e0aa',    
    'xtick.color': '#82e0aa',      
    'ytick.color': '#82e0aa',        
    'grid.color': '#444444',         
    'axes.edgecolor': 'white'        
})


include_ext = False

seed = 324


# Competition data
train_0 = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Original data
orig_0 = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')

# Define the target column
target = 'y'
train_0.head()


orig_0.head()


# Get ame column names for competition and external data
orig_0.columns = train_0.columns

# Binarize the target in the external data
orig_0['y'] = (orig_0['y']=='yes')*1


month_dico = ({'jan':1, 'feb':2, 'mar':3,
              'apr':4, 'may':5, 'jun':6,
              'jul':7, 'aug':8, 'sep':9,
              'oct':10, 'nov':11, 'dec':12})
for df in [train_0, orig_0, test_0]:
    df['month'] = df['month'].map(month_dico)


orig_0.y.value_counts()


train_0['month'].value_counts().to_frame().sort_values('month').plot.bar()
plt.show()


train_profile_report = ProfileReport(train_0, title='profile report of test data')

train_profile_report


target_counts = train_0[target].value_counts()

fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = target_counts.plot.bar(color=['#e86100', '#da1d81'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = target_counts.plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.28,
                            explode=[0.05, 0.1],
                            startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.38, ax=ax1)
ax1.set_ylabel('')
plt.suptitle('Counts of target classes in train data')
plt.tight_layout()


target_counts = orig_0[target].value_counts()

fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = target_counts.plot.bar(color=['#e86100', '#da1d81'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = target_counts.plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.28,
                           # explode=[0.05, 0.1],
                            startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.38, ax=ax1)
ax1.set_ylabel('')
plt.suptitle('Counts of target classes in external data')
plt.tight_layout()


# Decide if external data should be invluded
use_external = True

if use_external:
    train_0 = pd.concat([train_0, orig_0], ignore_index=True)
else:
    pass


cat_feats = test_0.select_dtypes(exclude='number').columns.tolist()
num_feats = test_0.select_dtypes(include='number').columns.tolist()

le = LabelEncoder()
# Define function for data preparation
def df_preparator(df):
    df = df.copy()
    for cat_feat in cat_feats:
        df[cat_feat] = le.fit_transform(df[cat_feat])
    return df


plt.figure(figsize=(8, 4))
sns.countplot(train_0, x=cat_feats[0], hue=target, width=0.7)
plt.xticks(rotation=90)
plt.title(f'Count of unique {cat_feats[0]} grouped by target classes', fontsize=12)
plt.show()

for cat_feat in cat_feats[1:]:
    sns.catplot(train_0, x=cat_feat, 
                kind='count', orient='v', 
                col=target, aspect=1.2, height=3)
    plt.suptitle(f'Count of unique {cat_feat} grouped by target classes')
    plt.tight_layout()
    plt.show()


for num_feat in num_feats:
    sns.catplot(train_0, x=num_feat,
                kind='violin', orient='v', 
                col=target, aspect=1.2, height=3)
    plt.suptitle(f'Boxplot of {num_feat} grouped by target classes')
    plt.tight_layout()
    plt.show()


# le = LabelEncoder()
train_2 = df_preparator(train_0)

train_data = train_2.copy()
train_target = pd.Series(le.fit_transform(train_data.pop(target)))

test_data = df_preparator(test_0)


lgbm_params = {'n_estimators':990, "max_depth":12, "learning_rate":0.17, 
               'subsample':0.77, 'colsample_bytree':0.743, 'reg_alpha':3.13, 
               'reg_lambda':8.71, 'random_state':42, 'boosting_type':"gbdt", 
               'objective':"binary", 'metric':"auc", 'verbose':-1}

cat_params = 


# Compare lgbm, xgboost and catboost base models

Models = [
          # ('lgb_clf', LGBMClassifier(verbose=-1)),
          # ('lgb_clf_300', LGBMClassifier(verbose=-1, n_estimators=300)),
          ('lgb_clf_1500', LGBMClassifier(verbose=-1, n_estimators=1500)),
          ('cat_clf', CatBoostClassifier(verbose=False)),
          ('lgbm_optim', LGBMClassifier(n_estimators=990, max_depth=12,learning_rate=0.17, 
                                        subsample=0.77, colsample_bytree=0.743, reg_alpha=3.13,
                                        reg_lambda=8.71, random_state=42, boosting_type="gbdt",
                                        objective="binary", metric="auc", verbose=-1)),
          # ('cat_clf_1000', CatBoostClassifier(verbose=False, iterations=1000)),
          ('xgb_clf', XGBClassifier()),
         ]

'''Dataset without any new columns'''
n_splits=10
scores = [] # Empty cross validation score list
models = [] # Empty list of models
my_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
for model_name, model in Models:
    X = train_data
    y = train_target
    # Cross validation
    cv_score = cross_val_score(model, X=X, y=y, cv=my_cv, scoring='roc_auc')

    scores.append(cv_score) # Add the scores to the scores list
    models.append(model_name) # Add the model to the list of models
    scores_df = pd.DataFrame(scores, 
                             columns=[f'cv_{n+1}' for n in range(n_splits)], 
                             index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display((scores_df.style
       .background_gradient(cmap='RdYlGn', axis=0)
       # .highlight_min(axis=0, color='yellow')
       .format('{:.5f}')
       .set_properties(**{'font-size': '14pt', 'weight': 'bold'}))
       )


palette = 'Dark2'

fig = plt.figure(figsize=(9, 6))
gs = GridSpec(3, 2, height_ratios=[2, 3, 3], width_ratios=[1, 1])

ax0 = fig.add_subplot(gs[0, :])
ax0 = sns.lineplot(scores_df.T.iloc[:-1, :], palette=palette, marker='o')
ax0.set_ylabel('Scores')
ax0.set_title('CV scores on various models', fontsize=10)
ax0.legend([])

ax1 = fig.add_subplot(gs[1:, :-1])
ax1 = sns.boxplot(scores_df.T.iloc[:-1, :], palette=palette, saturation=0.3)
ax1 = sns.swarmplot(scores_df.T.iloc[:-1, :], palette=palette)
ax1.set_ylabel('Scores')
ax1.set_title('CV scores on various models', fontsize=10)

ax2 = fig.add_subplot(gs[1:, -1:])
ax2 = sns.barplot(scores_df.iloc[:, :-1].T, palette=palette)
for avge in ax2.containers:
    ax2.bar_label(avge, fmt='%.4f')
plt.ylim(0.96, 0.973)
ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_title('Average CV scores on various models', fontsize=10)
plt.tight_layout()


my_spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

def objective(trial):
    # Sample weights for the three estimators
    
    weight_lgb = trial.suggest_int('weight_lgb', 0, 10, 1)
    weight_cat = trial.suggest_int('weight_cat', 0, 10, 1)
    weight_xgb = trial.suggest_int('weight_xgb', 0, 10, 1)

    # Create the Voting Regressor
    weights=[
             weight_lgb, 
             weight_cat, 
             weight_xgb, 
    ] # The weights

    scores = []
    vot_clf = VotingClassifier(estimators=Models, weights=weights, n_jobs=-1) # The voting regressor

    for f, (tr_ind, va_ind) in enumerate(my_spliter.split(train_data, train_target), start=1):
        X_tr, X_va = train_data.iloc[tr_ind], train_data.iloc[va_ind]
        y_tr, y_va = train_target.iloc[tr_ind], train_target.iloc[va_ind]
        # The model
        model = vot_clf
        # Fit the model
        model.fit(X_tr, y_tr)
        
        pred_ = mode.predict_proba(X_va)
        score = roc_auc_score(y_va, pred_)
        scores.append(score)
        
    return np.mean(scores)

# Define the function to run optuna optimization
def Run_Pass_cat_study(n_trials=1):
    if n_trials>1:
        # Optimize using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        # Get the best weights
        best_weights = study.best_params

    else:
        print('No need to run optuna, we will use the parameters obtained earlier')
        best_weights = {'weight_xgb': 2, 
                             'weight_lgb': 1, 
                             'weight_cat': 7 
                             }

    print('\n\nbest params: {}'.format(best_weights))
    return best_weights


weights_list = list(Run_Pass_cat_study(1).values())

vot_clf = VotingClassifier(
    estimators=Models,
    # weights = weights_list,
    voting='soft'
)

vot_clf


from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

bagg_clf_base = BaggingClassifier(
    estimator=CatBoostClassifier(verbose=200),
    max_samples=0.8, n_estimators=5, 
    random_state=0, oob_score=True
)


from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

bagg_clf_lgbm = BaggingClassifier(
    estimator=LGBMClassifier(**lgbm_params), max_samples=0.8,
    n_estimators=10, random_state=0, oob_score=True
)


# Choice of the final model
model = 'bag_base'

if model == 'vot':
    final_model = vot_clf
elif model == 'bag_base':
    final_model = bagg_clf_base
elif model == 'bag_lgbm':
    final_model = bagg_clf_lgbm
else:
    final_model = Models[2][1]


splits = 5

spliter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

scores_with_prep = []

for f, (trn_ind, val_ind) in enumerate(spliter.split(train_data, train_target), start=1):
    X_trn, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_trn, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    clf = final_model.fit(X_trn, y_trn)
    y_val_hat = clf.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, y_val_hat)
    scores_with_prep.append(score)

    #Plot the roc_curve
    fpr, tpr, _ = roc_curve(y_val, y_val_hat)
    plt.plot(fpr, tpr, label = 'roc_auc_fold_{} : {:.6f}'.format(f, score))
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('average auc_score with_prep: {:.6} ± {:.4}'.format(np.mean(scores_with_prep), np.std(scores_with_prep)))
    plt.legend()


# my_spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

# for f, (tr_ind, va_ind) in enumerate(my_spliter.split(train_data, train_target), start=1):
#     X_tr, X_va = train_data.iloc[tr_ind], train_data.iloc[va_ind]
#     y_tr, y_va = train_target.iloc[tr_ind], train_target.iloc[va_ind]

#     model = final_model

#     model.fit(X_tr, y_tr)
#     pred_va = model.predict_proba(X_va)[:, 1]

#     score = roc_auc_score(y_va, pred_va)
#     c = 90 + f # choice of color
#     print((2*f-2)*'  ' + '\033[{}m|=> Fold_{} accuracy: {:.6f}...\n\033[0m'.format(c, f, score))


final_model.fit(train_data, train_target)


pred_proba = final_model.predict_proba(test_data)

pred_proba


np.argmax(pred_proba, axis=1)


fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

# target_count = sub_file[target].value_counts()
pred_proba_df = pd.Series(np.argmax(pred_proba, axis=1))

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = pred_proba_df.value_counts().plot.bar(color=['#e86100', '#da1d81'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = pred_proba_df.value_counts().plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.28,
                            explode=[0.05, 0.1],
                            startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.38, ax=ax1)
ax1.set_ylabel('')
plt.suptitle('Counts of target classes in test prediction')
plt.tight_layout()


pd.Series(pred_proba[:, 1]).plot.hist(bins=25, figsize=(10, 4), title='Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probalities');


preds = final_model.predict_proba(test_data)[:, 1]

subm[target] = preds

subm.head()


subm.to_csv('submission.csv', index=False)

print('\033[92mThe file is ready for submission\033[0m')

