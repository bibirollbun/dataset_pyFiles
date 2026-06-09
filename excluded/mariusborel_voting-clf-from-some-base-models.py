import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
    'text.color': 'gold',   
    'axes.labelcolor': 'gold',    
    'xtick.color': '#82e0aa',      
    'ytick.color': '#82e0aa',        
    'grid.color': '#444444',         
    'axes.edgecolor': 'white'        
})


include_ext = True

seed = 324


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

ext_0 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')

train_ext = pd.concat([train_0, ext_0], ignore_index=True)

target = 'Personality'

train_0.head()


print('Shapes\ntrain: {}\ntest: {}\nexternal: {}'.format(train_0.shape, test_0.shape, ext_0.shape))


la_enc = LabelEncoder()

train_x = train_0.copy()
test_x = test_0.copy()

train_x['Stage_fear'] = la_enc.fit_transform(train_x['Stage_fear'])
test_x['Stage_fear'] = la_enc.transform(test_x['Stage_fear'])

train_x['Drained_after_socializing'] = la_enc.fit_transform(train_x['Drained_after_socializing'])
test_x['Drained_after_socializing'] = la_enc.transform(test_x['Drained_after_socializing'])

train_x['sum_'] = train_x[['Stage_fear', 'Drained_after_socializing']].sum(axis=1)
test_x['sum_'] = test_x[['Stage_fear', 'Drained_after_socializing']].sum(axis=1)


if include_ext:
    train_1 = train_ext.copy()
else:
    train_1 = train_0.copy()


# Define function for data preparation
def df_preparator(df):
    # create a copy of the dataset
    df = df.copy()
    
    # fillna in cat_features
    df['Stage_fear'] = df['Stage_fear'].fillna(df['Drained_after_socializing'])
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Stage_fear'])

    # Binarize the cat_features
    df['Stage_fear'] = df['Stage_fear']=='Yes'
    df['Drained_after_socializing'] = df['Drained_after_socializing']=='Yes'

    return df


le = LabelEncoder()
train_2 = df_preparator(train_x)

train_data = train_2.copy()
train_target = pd.Series(le.fit_transform(train_data.pop(target)))

test_data = df_preparator(test_x)


Models = [
          ('lgb_clf', LGBMClassifier(verbose=-1)),
          # ('lgb_clf_', LGBMClassifier(n_estimators=300, verbose=-1)),
          ('cat_clf', CatBoostClassifier(verbose=False)),
          # ('cat_clf_', CatBoostClassifier(n_estimators=50, verbose=False)),
          ('xgb_clf', XGBClassifier()),
          # ('xgb_clf_', XGBClassifier(n_estimators=200)),
          # ('gb_clf', GradientBoostingClassifier())
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
    cv_score = cross_val_score(model, 
                               X=X, 
                               y=y, 
                               cv=my_cv,
                               scoring='accuracy'
                              )
    
    scores.append(cv_score) # Add the scores to the scores list
    models.append(model_name) # Add the model to the list of models
    scores_df = pd.DataFrame(scores, 
                             columns=[f'cv_{n+1}' for n in range(n_splits)], 
                             index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.style.background_gradient(cmap='RdYlGn', axis=0)
         # .highlight_min(axis=0, color='yellow')
         .format('{:.5f}')
         .set_properties(**{'font-size': '14pt', 'weight': 'bold'}))
       )


palette = 'Dark2'

fig = plt.figure(figsize=(8, 5))
gs = GridSpec(3, 2, height_ratios=[2, 3, 3], width_ratios=[1, 1])

ax0 = fig.add_subplot(gs[0, :])
ax0 = sns.lineplot(scores_df.T.iloc[:-1, :], palette=palette, marker='o')
ax0.set_ylabel('Scores')
ax0.set_title('CV scores on various models', fontsize=12)
ax0.legend([])

ax1 = fig.add_subplot(gs[1:, :-1])
ax1 = sns.boxplot(scores_df.T.iloc[:-1, :], palette=palette, saturation=0.3)
ax1 = sns.swarmplot(scores_df.T.iloc[:-1, :], palette=palette)
ax1.set_ylabel('Scores')
ax1.set_title('CV scores on various models', fontsize=12)

ax2 = fig.add_subplot(gs[1:, -1:])
ax2 = sns.barplot(scores_df.iloc[:, :-1].T, palette=palette)
for avge in ax1.containers:
    ax1.bar_label(avge, fmt='%.4f')
plt.ylim(0.96, 0.973)
ax2.set_ylabel('')
ax2.set_title('Averave CV scores on various models', fontsize=12)
plt.tight_layout()


my_spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

def objective(trial):
    # Sample weights for the three estimators
    
    weight_lgb = trial.suggest_int('weight_lgb', 0, 10, 1)
    # weight_lgb_ = trial.suggest_int('weight_lgb_', 0, 10, 1)
    weight_cat = trial.suggest_int('weight_cat', 0, 10, 1)
    # weight_cat_ = trial.suggest_int('weight_cat_', 0, 10, 1)
    weight_xgb = trial.suggest_int('weight_xgb', 0, 10, 1)
    # weight_xgb_ = trial.suggest_int('weight_xgb_', 0, 10, 1)

    # Create the Voting Regressor
    weights=[
             weight_lgb, 
             # weight_lgb_,  
             weight_cat, 
             # weight_cat_, 
             weight_xgb, 
             # weight_xgb_
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
        
        score = model.score(X_va, y_va)

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
                             'weight_cat': 5 
                             }

    print('\n\nbest params: {}'.format(best_weights))
    return best_weights

# weights_dico = Run_Pass_cat_study(1)

weights_list = list(Run_Pass_cat_study(50).values())


vot_clf = VotingClassifier(
    estimators=Models,
    weights = weights_list,
    voting='soft'
)

vot_clf


my_spliter = StratifiedKFold(n_splits=8, shuffle=True, random_state=44)

for f, (tr_ind, va_ind) in enumerate(my_spliter.split(train_data, train_target), start=1):
    X_tr, X_va = train_data.iloc[tr_ind], train_data.iloc[va_ind]
    y_tr, y_va = train_target.iloc[tr_ind], train_target.iloc[va_ind]

    model = vot_clf

    model.fit(X_tr, y_tr)

    score = model.score(X_va, y_va)
    c = 90 + f # choice of color
    print((2*f-2)*'  ' + '\033[{}m|__ Fold_{} =-> accuracy: {:.6f}...\n\033[0m'.format(c, f, score))


# from sklearn.svm import SVC
# from sklearn.ensemble import BaggingClassifier
# from sklearn.datasets import make_classification

# final_model = BaggingClassifier(estimator=LGBMClassifier(),
#                         n_estimators=10, random_state=0).fit(X, y)


final_model = vot_clf.fit(train_data, train_target)


pred_proba = final_model.predict_proba(test_data)

pred_proba


pd.Series(pred_proba[:, 1]).plot.hist(bins=100, figsize=(10, 4), title='Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probalities');


preds = final_model.predict(test_data)

subm[target] = le.inverse_transform(preds)

subm.head()


fig = plt.figure(figsize=(6.6, 5))
gs = GridSpec(2, 2, height_ratios=[2, 2], width_ratios=[2, 2])

ax0 = fig.add_subplot(gs[:, :])
ax1 = subm[target].value_counts().plot.bar(color=['#d35400', '#a2006d'], alpha=0.9)
for count in ax0.containers:
    ax0.bar_label(count, label_type='center')
ax1 = fig.add_subplot(gs[:-1, -1:])
ax1 = subm[target].value_counts().plot.pie(autopct='%.2f%%', radius=1.1, explode=[0, 0.1])
ax1.set_ylabel('')
plt.tight_layout()


subm.to_csv('submission.csv', index=False)

print('\033[92mThe file is ready for submission\033[0m')

