import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.gridspec import GridSpec
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, 
                              StackingClassifier, BaggingClassifier,VotingClassifier)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice
import plotly.express as px
from scipy.stats import rankdata

my_colors = ['lightblue', 'gold']
my_colors_r = ['gold', 'lightblue']
seed = 32
n_splits = 6

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


data = [{'a':i, 'b':2*i, 'c':i**2, 'd':2**i} for i in range(10)]
data


data_df = pd.DataFrame(data)
data_df


# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Dark2", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222',  
    'figure.facecolor': '#222222', 
    'text.color': '#00FFFF',  
    'axes.labelcolor': '#00FFFF',  
    'xtick.color': '#00FFFF',      
    'ytick.color': '#00FFFF',     
    'grid.color': '#444444',         
    'axes.edgecolor': 'white'      
})

colors = ['#00ac74', '#8b4513', 'darkviolet', 'violet', 'lightgreen']

first_colors = ['lightblue', 'gold']
first_colors_r = ['gold', 'lightblue']

seed = 32


target = 'Personality'


train_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test_0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')

orig_0 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')

submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train_0.head()


orig_0.head()


orig_2 = orig_0.copy()

orig_2.rename(columns={target:'match'}).drop_duplicates(test_0.columns.tolist())

train_m = train_0.merge(orig_2, how='left')
train_m


test_m = test_0.merge(orig_2, how='left').drop(columns=[target])
test_m


train_0.shape, test_0.shape, train_m.shape, test_m.shape, orig_2.shape


n = 0
for df_name, df in [('train Dataset', train_0), ('external Dataset', orig_0), ('test Dataset', test_0)]:
    n += 1
    print('\033[{}m{}\n{}\n\033[0m'.format(92+n, df_name.upper(), df.isnull().sum()))


# train_0 = train_0.dropna(subset=['Stage_fear','Drained_after_socializing'])


for df_name, df in [('train', train_0), ('test', test_0), ('original', orig_0)]:
    nunb_of_duplicates = df.duplicated().sum()
    if nunb_of_duplicates != 0:
        print(f'{df_name} dataset has {nunb_of_duplicates} duplicates.')
    else:
        print(f'{df_name} dataset has no duplicates')


orig_0 = orig_0.drop_duplicates(subset=test_0.columns.tolist())


plt.figure(figsize=(10, 4))

plt.subplot(121)
train_0[target].value_counts().plot.pie(autopct='%.2f%%',
                                              colors=colors,
                                              explode=[0, 0.1], radius=1.2)
plt.ylabel('')
plt.subplot(122)
ax = sns.countplot(train_0, x=target)
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.suptitle('Count Introvert vs Extrovert in competion train population')
plt.ylabel('')
plt.show()


plt.figure(figsize=(10, 4))

plt.subplot(121)
orig_0[target].value_counts().plot.pie(autopct='%.2f%%',
                                              colors=colors,
                                              explode=[0, 0.1], radius=1.2)
plt.ylabel('')
plt.subplot(122)
ax = sns.countplot(orig_0, x=target)
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.suptitle('Count Introvert vs Extrovert in external population')
plt.ylabel('')
plt.show()


sns.pairplot(orig_0, hue=target, height=1.8);


def check_stage_drained_similarity(df):
    df['Stage_vs_Drained'] = (df['Stage_fear']==df['Drained_after_socializing'])*1
    # print("\033[{}m{:.2f}%\033[0m".format(df['Stage_vs_Drained'].mean()*100) + "of the rows have similar stage_fear and drained response")
    # print("\033[32m{:.2f}%\033[0m of the rows have similar stage_fear and drained response".format(df['Stage_vs_Drained'].mean()*100))
    percentage = df['Stage_vs_Drained'].mean() * 100

    # Decide color based on value
    if percentage >= 80:
        color_code = '32'  # Green
    # elif percentage >= 50:
    #     color_code = '33'  # Yellow
    else:
        color_code = '31'  # Red

    # Print with color
    print("\033[{}m{:.2f}%\033[0m of the rows have similar stage_fear and drained response".format(color_code, percentage))



check_stage_drained_similarity(test_0)


check_stage_drained_similarity(train_0)


check_stage_drained_similarity(orig_0)


num_feats = train_0.select_dtypes(include='number').columns.tolist()
cat_feats = test_0.select_dtypes(exclude='number').columns.tolist()


# Decide if features should be engineered
feat_eng = True
n = 3


def df_processing(df):
    if feat_eng:
        df = df.copy()
        df['Stage_vs_Drained'] = (df['Drained_after_socializing'] == df['Stage_fear'])*1
        df[cat_feats] = df[cat_feats].fillna('missing')
        df[num_feats] = df[num_feats].fillna(df[num_feats].mean())
        df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1, 'missing': -5})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1, 'missing':-5})
        df['Stage_+_Drained'] = df['Drained_after_socializing'] + df['Stage_fear']
        # df['Stage_or_Drained'] = df['Drained_after_socializing']== | df['Stage_fear']==1
        # df['Stage_and_Drained'] = df['Drained_after_socializing']==1 & df['Stage_fear']==1
        for num_feat in num_feats:
            df[f'sin({num_feat})'] = np.sin(df[num_feat]*np.pi/2)
            df[f'cos({num_feat})'] = np.cos(df[num_feat]*np.pi/2)
    X = df.copy()
    try:
        y = X.pop(target)
        return X, y
    except:
        pass
        return X
        # pass


# We chose to use only the external population with minority class (introverts)
train_orig_over = pd.concat([train_0, orig_0[orig_0[target]!='Introvert']], axis=0, ignore_index=True)
train_orig_over.shape


plt.figure(figsize=(10, 4))

plt.subplot(121)
train_orig_over[target].value_counts().plot.pie(autopct='%.2f%%', 
                                                 colors=colors, 
                                                 explode=[0, 0.1], 
                                                 radius=1.2)
plt.ylabel('')
plt.subplot(122)
ax = sns.countplot(train_orig_over, x=target)
for label in ax.containers:
    ax.bar_label(label, fontsize=10)
plt.suptitle('Count Introvert vs Extrovert in external set')
plt.ylabel('')
plt.show()


train_orig = pd.concat([train_0, orig_0], axis=0, ignore_index=True)
train_orig.shape


def get_the_train_set(SET=None):
    if SET=='merge':
        train = train_m.copy()
        used_set = 'train merged to original'
    elif SET=='over_class':
        train = train_orig_over.copy()
        used_set = 'train + original with overclass in original'
    elif SET=='orig':
        train = train_orig.copy()
        used_set = 'train + original sets'
    else:
        train = train_0.copy()
        used_set = 'only train set'
    print("\033[43;32m we are using {}\033[0m".format(used_set))
    return train

train = get_the_train_set('over_class')

# Prepare the train sets
X_tr, y_tr = df_processing(train)

# Prepare the original sets
X_or, y_or = df_processing(orig_0)

# Prepare the combined train_orig sets
X_tr_or, y_tr_or = df_processing(train_orig)

# Prepare the test set
X_ts = df_processing(test_0)
X_ts.sample(5)


features_trans = make_column_transformer(
    (MinMaxScaler(), num_feats),
    # (OneHotEncoder(), X_tr.select_dtypes(exclude='number').columns.tolist()),
    remainder='drop', 
    sparse_threshold=0)


X_prep = X_tr.copy()

x_prep = features_trans.fit_transform(X_prep)

X_prep.sample(5)


X_prep['Stage_vs_Drained'].mean()


X, y = X_tr, y_tr

X_train, X_val, y_train, y_val = train_test_split(X, y_tr, test_size=0.25, shuffle=True, random_state=seed)

[d.shape for d in [X_train, X_val, y_train, y_val]]


estimators = [
    ('hgb', HistGradientBoostingClassifier()),
    ('cat', CatBoostClassifier(verbose=0)),
    ('lgb', LGBMClassifier(verbose=-1)),
    # ('xgb', XGBClassifier()),
]

stack_clf = make_pipeline(features_trans,
    StackingClassifier(
            estimators=estimators, stack_method='predict',
            # final_estimator=XGBClassifier(),
            final_estimator=LogisticRegression(C=0.1, n_jobs=-1, 
                                               multi_class='ovr', 
                                               # class_weight={1:1-0.3, 0:0.3}
                                              ),
            # n_jobs=-1,
            passthrough=True
    ))

stack_clf


my_spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

for f, (train_idx, test_idx) in enumerate(my_spliter.split(X, y), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = stack_clf
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    c = 90 + f # choice of color
    print((2*f-2)*'  ' + '\033[{}m|__ Fold_{} =-> accuracy: {:.5f}...\n\033[0m'.format(c, f, score))


final_model = stack_clf

final_model.fit(X_tr, y_tr)


pred_val = final_model.predict(X_val)

plt.figure(figsize=(8,4))
conf_matrix = confusion_matrix(y_val, pred_val)
plt.subplot(121)
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, 
            cmap='Blues', xticklabels=['Introvert', 'Extrovert'], 
            yticklabels=['Introvert', 'Extrovert'])
plt.subplot(122)
conf_matrix_norm = confusion_matrix(y_val, pred_val, normalize='true')*100
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cbar=False, 
            cmap='Greens', xticklabels=['Introvert', 'Extrovert'], 
            yticklabels=['Introvert', 'Extrovert'])

plt.suptitle('Accuracy of the final model: {:.4f} %'.format(accuracy_score(y_val, pred_val)*100))
plt.show()


pred_test_sc = final_model.predict(X_ts)

sub_pred_sc = submission.copy()

sub_pred_sc[target] = pred_test_sc

sub_pred_sc[target] = sub_pred_sc[target]

sub_pred_sc.head(10)


fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

# Define the explode values for pie chart
n_classes = sub_pred_sc[target].value_counts()
explode = [0.05 for n in n_classes]

target_count = sub_pred_sc[target].value_counts()

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = target_count.plot.bar(color=colors)
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = target_count.plot.pie(autopct='%.2f%%',shadow = True,
                            radius=1.28, colors=colors,
                            explode=explode, startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['white'], radius=0.4, ax=ax1)
ax1.set_ylabel('')
plt.tight_layout()


sub_pred_sc.to_csv('submission.csv', index=False)
print('The file is ready for submission.')


train_0['Stage_fear'].unique()


def cat_dist_plotter(cat_feat_1, cat_feat_2, cat_feat_3=None):
    sns.catplot(train_0.dropna(),
                y='Time_spent_Alone',
                x=cat_feat_1,
                hue=cat_feat_2,
                col=cat_feat_3,
                split=True, 
                kind='violin',
                aspect=1.2
               )
    plt.show()


cat_dist_plotter(target, 'Stage_fear', 'Drained_after_socializing')


cat_dist_plotter('Drained_after_socializing', 'Stage_fear', target)


cat_dist_plotter('Stage_fear', target, 'Drained_after_socializing')


#plt.figure(figsize=(12,4))
sns.catplot(train_0, kind='boxen', aspect=1.6, height=3.8, col=target, orient='h')
# plt.xticks(rotation=90)
# plt.title('Distribution of num_feats', fontsize=13)
plt.tight_layout()
plt.show()


ax = sns.catplot(train_0, col='Stage_fear', kind='count', 
                 x=target, height=4, aspect=1.2)


ax = sns.catplot(train_0, x=target, kind='count', 
                 col='Drained_after_socializing', 
                 height=4, aspect=1.2)


cmap = 'Reds'

ctab = pd.crosstab(train_0[target], train_0['Stage_fear'], normalize='columns')
plt.figure(figsize=(4,4))
ax = sns.heatmap(ctab, annot=True,fmt='.2%', cbar=False,
                linecolor='grey', linewidth=0.5, cmap=cmap)
plt.title('% of Personality vs Stage_fear')
plt.tight_layout()


ctab = pd.crosstab(train_0[target], train_0['Drained_after_socializing'], normalize='columns')
plt.figure(figsize=(4,4))
ax = sns.heatmap(ctab, annot=True,fmt='.2%', cbar=False,
                linecolor='grey', linewidth=0.5, cmap=cmap)
plt.title('% of personality vs Drained_after_socializing')
plt.tight_layout()


ctab = pd.crosstab(train_0['Stage_fear'], train_0['Drained_after_socializing'], normalize='columns')
plt.figure(figsize=(4,4))
ax = sns.heatmap(ctab, annot=True,fmt='.2%', cbar=False,
                linecolor='grey', linewidth=0.5, cmap=cmap)
plt.title('% of Stage_fear vs Drained_after_socializing')
plt.tight_layout()


ctab = pd.crosstab(train_0['Stage_fear'], train_0['Drained_after_socializing'])
plt.figure(figsize=(4,4))
ax = sns.heatmap(ctab, annot=True,fmt='d', cbar=False,
                linecolor='grey', linewidth=0.5, cmap=cmap)
plt.title('count of Stage_fear vs Drained_after_socializing')
plt.tight_layout()


x = np.arange(1, 10)
np.add.accumulate(x)


x = np.arange(1, 8)
np.multiply.reduce(x)


x = np.arange(1, 8)
np.cumprod(x)


x = np.arange(1, 13)
multi = np.multiply.outer(x, x)
multi 


sns.heatmap(multi, annot=True, cmap='Reds', fmt='d', cbar=False, square=True)

