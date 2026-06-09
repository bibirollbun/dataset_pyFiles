import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
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
import shap

import warnings
warnings.filterwarnings('ignore')

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Dark2", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222',     # Dark gray plot background
    'figure.facecolor': '#222222',   # Dark gray around the figure
    'text.color': '#82e0aa',           # White text everywhere
    'axes.labelcolor': 'gold',      # White axis labels
    'xtick.color': '#82e0aa',          # White x-axis tick labels
    'ytick.color': '#82e0aa',          # White y-axis tick labels
    'grid.color': '#444444',         # Slightly lighter grid
    'axes.edgecolor': 'white'        # White border of the plot
})


include_ext = False
seed = 0


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

ext_0 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')

train_ext = pd.concat([train_0, ext_0], ignore_index=True)

target = 'Personality'

train_0.head()


train_0.corrwith(train_0[target]=='Introvert', numeric_only=True).sort_values().plot.barh()


pos_corr_target = train_0[(train_0.corrwith(train_0[target]=='Introvert', numeric_only=True)>0).index]
pos_corr_target


for df in [train_0, test_0, ext_0, train_ext]:
    df['numb_of_NaN'] = df.isna().sum(axis=1)
    df['Sum_extro_favors'] = df[['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']].sum(axis=1)
    df['Diff_extro_favors'] = df[['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']].sum(axis=1) - df['Time_spent_Alone']
    df['ratio_extro_favors'] = np.clip(df['Time_spent_Alone']/df['Sum_extro_favors'], -1, 100)
    df['going_out/friends'] = np.clip(np.divide(df['Going_outside'], df['Friends_circle_size']), -1, 10)
    df['social_events/post_freq'] = np.clip(np.divide(df['Social_event_attendance'], df['Post_frequency']), -1, 5)
    # df['constant'] = 1


train_0.head()


fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 0.5], width_ratios=[3, 2.2])

ax0 = fig.add_subplot(gs[:, :])
ax0 = sns.countplot(train_0, x='numb_of_NaN', hue=target)
for count in ax0.containers:
    ax0.bar_label(count)
ax1 = fig.add_subplot(gs[:-1, -1:])
ax1 = sns.countplot(train_0[train_0['numb_of_NaN']>2], x='numb_of_NaN', hue=target)
for count in ax1.containers:
    ax1.bar_label(count, label_type='center')
ax1.set_ylabel('')

plt.suptitle('Number of NaNs in Personalities')
plt.tight_layout()


if include_ext:
    train_1 = train_ext.copy()
else:
    train_1 = train_0.copy()


# Define function for data preparation
def df_preparator(df):
    # create a copy of the dataset
    df = df.copy()
    
    # # fillna in cat_features
    # df['Stage_fear'] = df['Stage_fear'].fillna(df['Drained_after_socializing'])
    # df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Stage_fear'])

    # Binarize the cat_features
    df['Stage_fear'] = df['Stage_fear']=='Yes'
    df['Drained_after_socializing'] = df['Drained_after_socializing']=='Yes'

    return df


le = LabelEncoder()
train_2 = df_preparator(train_1)

X = train_2.copy()
y = pd.Series(le.fit_transform(X.pop(target)))

test_data = df_preparator(test_0)


Models = [
          ('lgb_clf', LGBMClassifier(n_estimators=100, verbose=-1)),
          ('cat_clf', CatBoostClassifier(verbose=False)),
          ('xgb_clf', XGBClassifier(colsample_bytree=0.2, max_depth=3))
         ]

'''Dataset without any new columns'''
scores = [] # Empty cross validation score list
models = [] # Empty list of models
n_splits=10
my_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

for model_name, model in Models:
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
                             columns=[f'cv_{n}' for n in range(1, n_splits+1)], 
                             index=models) # Get the acores into a data frame

scores_df['avg_score'] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by='avg_score', ascending=False)


display(
        (scores_df.style.background_gradient(cmap='Greens', axis=0)
         # .highlight_min(axis=0, color='yellow')
         .format('{:.5f}')
         .set_properties(**{'font-size': '14pt', 'weight': 'bold'}))
       )


palette = 'Dark2'

fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[5, 3])

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = sns.lineplot(scores_df.T.iloc[:-1, :], palette=palette, marker='o')
ax0.set_ylabel('Scores')
ax0.set_title('CV scores on various models', fontsize=12)

ax1 = fig.add_subplot(gs[:, 1:])
ax1 = sns.barplot(scores_df.iloc[:, :-1].T, palette=palette)
for avge in ax1.containers:
    ax1.bar_label(avge, fmt='%.4f')
plt.ylim(0.96, 0.973)
ax1.set_ylabel('')
ax1.set_title('Averave CV scores on various models', fontsize=12)
plt.tight_layout()


my_spliter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)

for f, (tr_ind, va_ind) in enumerate(my_spliter.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

    model = CatBoostClassifier(n_estimators=40, verbose=False)

    model.fit(X_tr, y_tr)

    score = model.score(X_va, y_va)
    c=90+f # choice of color
    print('\033[{}mFold_{} •••▶ accuracy: {}\n\033[0m'.format(c, f, score))


# X, y = train_data.select_dtypes(include='number'), train_target
X_num = X.select_dtypes(include='number')
y_dec = le.inverse_transform(y.copy())
X_train, X_test, y_train, y_test = train_test_split(X_num, y_dec)

# Train model
model = CatBoostClassifier(n_estimators=10, verbose=False).fit(X_train, y_train)

# Explain predictions
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize
shap.summary_plot(shap_values, 
                  X_test, cmap='BrBG',
                  axis_color='#82e0aa', 
                  plot_size=(10, 8),
                  class_names=['Introvert', 'Extrovert'],
                  show_values_in_legend=True, alpha=0.3
                 )


# Train model
model = LGBMClassifier(n_estimators=50, verbose=-1).fit(X_train, y_train)

# Explain predictions
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize
shap.summary_plot(shap_values, 
                  X_test, cmap='BrBG',
                  axis_color='#82e0aa', 
                  plot_size=(10, 8),
                  class_names=['Introvert', 'Extrovert'],
                  show_values_in_legend=True, alpha=0.3
                 )


final_model = CatBoostClassifier(verbose=False).fit(X, y)


preds = final_model.predict(test_data)

subm[target] = le.inverse_transform(preds)

subm.head()


fig = plt.figure(figsize=(6, 4.5))
gs = GridSpec(2, 2, height_ratios=[2, 1.6], width_ratios=[2, 2])

target_count = subm[target].value_counts()

ax0 = fig.add_subplot(gs[:, :])
ax1 = target_count.plot.bar(color=['#00ac74', '#8b4513'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center')
ax1 = fig.add_subplot(gs[0, 1:])
ax1 = target_count.plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.1,
                            explode=[0.1, 0.05], #cmap='BrBG_r',
                            startangle=90)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.4, ax=ax1)
ax1.set_ylabel('')
plt.tight_layout()


subm.to_csv('submission.csv', index=False)

print('\033[92mThe file is ready for submission\033[0m')

