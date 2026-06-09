import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
color_1 = '#a3e4d7'
# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222',     # Dark gray plot background
    'figure.facecolor': '#222222',   # Dark gray around the figure
    'text.color': color_1,           # White text everywhere
    'axes.labelcolor': color_1,      # White axis labels
    'xtick.color': color_1,          # White x-axis tick labels
    'ytick.color': color_1,          # White y-axis tick labels
    'grid.color': '#444444',         # Slightly lighter grid
    'axes.edgecolor': 'white'        # White border of the plot
})


include_ext = True
use_imputed = False

seed = 25


if use_imputed:
    train_0 = pd.read_csv('/kaggle/input/nan-imputed-datasets-for-introvert-extrovert-comp/imputed_train_data.csv')
    test_0 = pd.read_csv('/kaggle/input/nan-imputed-datasets-for-introvert-extrovert-comp/imputed_test_data.csv')
else:
    train_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
    test_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
    
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

ext_0 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')

if include_ext:
    train_0 = pd.concat([train_0, ext_0], ignore_index=True)
else:
    pass

target = 'Personality'

train_0.head()


ext_0.isna().sum()


# merge_cols = ['Time_spent_Alone', 'Stage_fear', 
#               'Social_event_attendance','Going_outside', 
#               'Drained_after_socializing', 
#               'Friends_circle_size', 'Post_frequency']

# train_df = train_0.merge(ext_0, how='left', on=merge_cols)
# test_df = test_0.merge(ext_0, how='left', on=merge_cols)

# train_df.isna().sum()


plt.figure(figsize=(7,5))
train_no_nan = train_0.dropna()
sns.scatterplot(train_no_nan, 
                x='Going_outside',
                y='Friends_circle_size', 
                hue=target, 
                s=train_no_nan['Social_event_attendance'],
                cmap=['gold', 'grey']
               );


plt.figure(figsize=(7,5))
train_no_nan = train_0.dropna()
sns.scatterplot(train_no_nan, 
                x='Time_spent_Alone',
                y='Friends_circle_size', 
                hue=target, 
                s=train_no_nan['Post_frequency'],
                cmap=['gold', 'grey']
               );


# To prevent encoding on fractions obtained from imputation

if not use_imputed: 
    la_enc = LabelEncoder()
    
    train_x = train_0.copy()
    test_x = test_0.copy()
    
    train_x['Stage_fear'] = la_enc.fit_transform(train_x['Stage_fear'])
    test_x['Stage_fear'] = la_enc.transform(test_x['Stage_fear'])
    
    train_x['Drained_after_socializing'] = la_enc.fit_transform(train_x['Drained_after_socializing'])
    test_x['Drained_after_socializing'] = la_enc.transform(test_x['Drained_after_socializing'])
    
    train_x['sum_'] = train_x[['Stage_fear', 'Drained_after_socializing']].sum(axis=1)
    test_x['sum_'] = test_x[['Stage_fear', 'Drained_after_socializing']].sum(axis=1)
else:
    train_x = train_0.copy()
    test_x = test_0.copy()


if include_ext:
    train_1 = train_0.copy()
else:
    train_1 = train_0.copy()


num_feats = test_0.select_dtypes(include='number').columns.tolist()


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
    for feat_1 in num_feats:
        for feat_2 in num_feats:
            if feat_1 != feat_2:
                df[f'{feat_1}*{feat_2}'] = df[feat_1]*df[feat_2]
                df[f'{feat_1}/{feat_2}'] = np.clip(np.divide(df[feat_1], df[feat_2]), 0, 10)
    return df


le = LabelEncoder()
train_2 = df_preparator(train_x)

train_data = train_2.copy()
train_target = pd.Series(le.fit_transform(train_data.pop(target)))

test_data = df_preparator(test_x)


test_data.head()


class_weight_ratio = train_target.mean()
class_weight_ratio


Models = [
          ('lgb_clf', LGBMClassifier(verbose=-1, class_weights=[class_weight_ratio, 1-class_weight_ratio])),
          ('lgb_clf_', LGBMClassifier(n_estimators=300, verbose=-1)),
          ('cat_clf', CatBoostClassifier(verbose=False)),
          ('cat_clf_', CatBoostClassifier(verbose=False, learning_rate=0.2, 
                                          depth=5, iterations=200, reg_lambda=3)),
          ('xgb_clf', XGBClassifier()),
          ('xgb_clf_', XGBClassifier(n_estimators=200)),
         ]


n_splits=10
scores = [] # Empty cross validation score list
models = [] # Empty list of models
my_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
for model_name, model in Models:
    X = train_data
    y = train_target
    # Cross validation
    cv_score = cross_val_score(model, X=X, y=y, cv=my_cv, scoring='accuracy')
    
    scores.append(cv_score) # Add the scores to the scores list
    models.append(model_name) # Add the model to the list of models
    scores_df = pd.DataFrame(scores, 
                             columns=[f'cv_{n+1}' for n in range(n_splits)], 
                             index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1) # Get the average score for each model
scores_df['std_score'] = scores_df.std(axis=1) # Get the standard deviation of scores for each model
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.iloc[:, :-2].style.background_gradient(cmap='RdYlGn', axis=0)
         .highlight_min(axis=0, color='lightgrey').format('{:.5f}')
         .set_properties(**{'font-size': '11pt', 'weight': 'bold'}))
       )

display(
        (scores_df.iloc[:, -2:].style.background_gradient(cmap='RdYlGn', axis=0)
         .highlight_min(axis=0, color='lightgrey').format('{:.5f}')
         .set_properties(**{'font-size': '11pt', 'weight': 'bold'}))
       )


palette = 'Dark2'

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 3, height_ratios=[2.6, 3, 3], width_ratios=[4, 0.5, 4])

ax0 = fig.add_subplot(gs[0, :])
ax0 = sns.lineplot(scores_df.T.iloc[:-2, :], palette=palette, markers='d', dashes=False)
plt.legend([])
ax0.set_title('Comapre scores within cv', fontsize=12)
ax0.set_ylabel('Scores')

ax1 = fig.add_subplot(gs[1:, :1])
ax1 = sns.swarmplot(scores_df.T.iloc[:-2, :], palette=palette)
ax1 = sns.boxplot(scores_df.T.iloc[:-2, :], palette=palette, saturation=0.3)
ax1.set_title('Scores distribution within models', fontsize=12)
ax1.grid(False)
ax1.set_ylabel('Scores')

ax2 = fig.add_subplot(gs[1:, 2:])
ax2 = sns.barplot(scores_df.iloc[:, :-2].T, palette=palette, saturation=0.6)
ax2.set_title('Scores average within models', fontsize=12)
for avge in ax2.containers:
    ax2.bar_label(avge, fmt='%.4f')
plt.ylim(0.95, 0.98)
ax2.grid(False)
ax1.set_ylabel('')
plt.tight_layout()


# Choice of model
model = 'lgbm'

if model == 'cat':
    # model = CatBoostClassifier(verbose=False, learning_rate=0.2, depth=5, iterations=200, reg_lambda=3)
    model = CatBoostClassifier(verbose=False)
elif model == 'lgbm':
    model = LGBMClassifier(verbose=-1, class_weights=[class_weight_ratio, 1-class_weight_ratio])
elif model == 'xgb':
    model = XGBClassifier(class_weights=[class_weight_ratio, 1-class_weight_ratio])

my_spliter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)

for f, (tr_ind, va_ind) in enumerate(my_spliter.split(train_data, train_target), start=1):
    X_tr, X_va = train_data.iloc[tr_ind], train_data.iloc[va_ind]
    y_tr, y_va = train_target.iloc[tr_ind], train_target.iloc[va_ind]

    model.fit(X_tr, y_tr)

    score = model.score(X_va, y_va)
    c = 90 + f # choice of color
    print((2*f-1)*'  ' + '\033[{}m|__ Fold_{} =-> accuracy: {:.8f}...\n\033[0m'.format(c, f, score))


final_model = model


preds = final_model.predict(test_data)

subm[target] = le.inverse_transform(preds)

subm.head()


fig = plt.figure(figsize=(6, 4.5))
gs = GridSpec(2, 2, height_ratios=[2, 2], width_ratios=[2, 2])

ax0 = fig.add_subplot(gs[:, :])
ax0 = subm[target].value_counts().plot.bar(color=['#d35400', '#a2006d'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center')
ax1 = fig.add_subplot(gs[:-1, -1:])
ax1 = subm[target].value_counts().plot.pie(autopct='%.2f%%', radius=1.1, explode=[0.05, 0.05])
ax1.set_ylabel('')
plt.tight_layout()


subm.to_csv('submission.csv', index=False)

print('\033[92mThe file is ready for submission\033[0m')

