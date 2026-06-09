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
    'axes.facecolor': '#222222',     # Dark gray plot background
    'figure.facecolor': '#222222',   # Dark gray around the figure
    'text.color': 'white',           # White text everywhere
    'axes.labelcolor': 'gold',      # White axis labels
    'xtick.color': '#82e0aa',          # White x-axis tick labels
    'ytick.color': '#82e0aa',          # White y-axis tick labels
    'grid.color': '#444444',         # Slightly lighter grid
    'axes.edgecolor': 'white'        # White border of the plot
})


include_external_data = False


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

ext_01 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')
ext_02 = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')

train_ext = pd.concat([train_0, ext_01, ext_02], ignore_index=True)

target = 'Personality'

train_0.head()


if include_external_data:
    train = train_ext
else:
    train = train_0


# Keep only Cat_features: Stage_fear and Drained_after_socializing
train_ = train.select_dtypes('object').copy()
test_ = test_0.select_dtypes('object').copy()


train_.head()


# train_['missing_Stage_fear'] = train_['Stage_fear'].isna()
# train_['missing_Drained_after_socializing'] = train_['Drained_after_socializing'].isna()


# test_['missing_Stage_fear'] = test_['Stage_fear'].isna()
# test_['missing_Drained_after_socializing'] = test_['Drained_after_socializing'].isna()

# train_


le_1 = LabelEncoder()

train_['Stage_fear'] = le_1.fit_transform(train_['Stage_fear'])
test_['Stage_fear'] = le_1.fit_transform(test_['Stage_fear'])

le_2 = LabelEncoder()

train_['Drained_after_socializing'] = le_2.fit_transform(train_['Drained_after_socializing'])
test_['Drained_after_socializing'] = le_2.fit_transform(test_['Drained_after_socializing']) 


X = train_.copy()
y_ = X.pop(target)

le = LabelEncoder()

y = le.fit_transform(y_)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12)

[d.shape for d in [X_train, X_valid, y_train, y_valid]]


clf = CatBoostClassifier(n_estimators=100, 
                         verbose=50, 
                         eval_fraction=0.2, 
                         eval_metric='Accuracy',
                         learning_rate = 0.1,
                         early_stopping_rounds=5
                        )

clf.fit(X_train, y_train, cat_features=X.columns.tolist())


score = clf.score(X_valid, y_valid)

print("\033[92m The model gets an accuracy of {:.6f} when trained with only the two cat_features.\033[0m".format(score))


preds = clf.predict(X_valid)

class_report = classification_report(y_valid, preds)
conf_matrix = confusion_matrix(y_valid, preds)

print(f"\033[95m{class_report}\033[0m")
target_labels = ['Introvert', 'Extrovert']
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, square=True, 
            xticklabels=target_labels, yticklabels=target_labels)
plt.show()


conf_matrix_norm = confusion_matrix(y_valid, preds, normalize='pred')

sns.heatmap(conf_matrix_norm, annot=True, cbar=False, square=True, 
            xticklabels=target_labels, yticklabels=target_labels)
plt.show()


my_spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

for f, (tr_ind, va_ind) in enumerate(my_spliter.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y[tr_ind], y[va_ind]
    c=90+f # choice of color
    print('\033[{}mFitting Fold_{}\n\033[0m'.format(c, f))
    model = clf = CatBoostClassifier(n_estimators=2000, 
                         verbose=50, 
                         eval_fraction=0.2,
                         learning_rate = 0.5,
                         eval_metric='Accuracy',
                         early_stopping_rounds=200
                                    )

    model.fit(X_tr, y_tr)

    score = model.score(X_va, y_va)
    print('\033[{}m==> Accuracy: {}\n\n\033[0m'.format(c, score))


clf.fit(X, y)


test_preds = clf.predict(test_)

subm[target] = le.inverse_transform(test_preds)

subm.head(10)


fig = plt.figure(figsize=(6, 5))
gs = GridSpec(2, 2, height_ratios=[2, 2], width_ratios=[2, 2])

ax0 = fig.add_subplot(gs[:, :])
ax1 = subm[target].value_counts().plot.bar(color=['#d35400', '#a2006d'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center')
ax1 = fig.add_subplot(gs[:-1, -1:])
ax1 = subm[target].value_counts().plot.pie(autopct='%.2f%%', radius=1.1)
ax1.set_ylabel('')
plt.tight_layout()


subm.to_csv('submission.csv', index=False)

print('The file is ready for submission!')

