import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from ydata_profiling import ProfileReport 

from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)

import warnings
warnings.filterwarnings('ignore')

seed = 42

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


h2o.init()


# Import a sample binary outcome train/test set into H2O
train_raw = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test_raw = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')
ext_raw = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')

ext_raw['y'] = (ext_raw['y']=='yes')*1


cat_feats = test_raw.select_dtypes(exclude='number').columns.tolist()
num_feats = test_raw.select_dtypes(include='number').columns.tolist()

def detailed_violin(col=None):
    for num_feat in num_feats:
        sns.catplot(train_raw, x=num_feat,
                    kind='violin', orient='v', row='y',
                    col=col, aspect=1.2, height=3)
        plt.suptitle(f'Boxplot of {num_feat} grouped by target classes')
        plt.tight_layout()
        plt.show()


detailed_violin(col='job')


cat_feats = test_raw.select_dtypes(exclude='number').columns.tolist()
num_feats = test_raw.select_dtypes(include='number').columns.tolist()

def detailed_violin(col=None):
    for cat_feat in cat_feats:
        sns.catplot(train_raw, x=cat_feat,
                    kind='count', orient='h', row='y',
                    col=col, aspect=1.2, height=3)
        plt.suptitle(f'Boxplot of {cat_feat} grouped by target classes')
        plt.tight_layout()
        plt.show()


detailed_violin(col='marital')


# Get ame column names for competition and external data
ext_raw.columns = train_raw.columns

# Binarize the target in the external data
ext_raw['y'] = (ext_raw['y']=='yes')*1


month_dico = ({'jan':1, 'feb':2, 'mar':3,
              'apr':4, 'may':5, 'jun':6,
              'jul':7, 'aug':8, 'sep':9,
              'oct':10, 'nov':11, 'dec':12})

quarter_dico = {
    'jan': 1, 'feb': 1, 'mar': 1, 
    'apr': 2, 'may': 2, 'jun': 2,
    'jul': 3, 'aug': 3, 'sep': 3, 
    'oct': 4, 'nov': 4, 'dec': 4
}

feats_to_encode = []
for df in [train_raw, ext_raw, test_raw]:
    # Time realated features
    df['quarter'] = df['month'].map(quarter_dico)
    df['month'] = df['month'].map(month_dico)
    df['year'] = '2025'
    df['day'] = np.clip(df['day'], 1, 28) # Make the max day be 28, does not reflect reality but required to go around some misn=matches
    df['date'] = pd.to_datetime(df[['year','month','day']])
    df.drop(columns=['year'], inplace=True)
    # Transformation of time related features
    # --->
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    # --->
    df['day_sin'] = np.sin(2*np.pi*df['day']/28)
    df['day_cos'] = np.cos(2*np.pi*df['day']/28)
    # --->
    df['week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    # --->
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    # Status on bank products
    df['housing_loan'] = df['housing']==df['loan']
    df['housing_default'] = df['housing']==df['default']
    df['loan_default'] = df['loan']==df['default']
    
    # df['balance/duration'] = df['balance']/(df['duration'] + 1)
    # df['pdays/duration'] = df['pdays']/(df['duration'] + 1)
    # df['previous/duration'] = df['previous']/(df['duration'] + 1)
    # df['duration/campaign'] = df['duration']/(1 + df['campaign'])
    # df['contacted_before'] = df['pdays'] != -1
    # df['campaign/previous'] = df['campaign']/(1 + df['previous'])
    # df['exp_campaign'] = np.clip(1.1**df['campaign'], 0, 100)
    df = df.drop(columns=['date'])
    for col in cat_feats:
        if df[col].unique().tolist()[0] in ['yes', 'no']:
            df[col] = (df[col]=='yes')*1
        else:
            feats_to_encode.append(col)


for col in set(feats_to_encode):
    le = LabelEncoder()
    le.fit(train_raw[col])
    train_raw[col] = le.transform(train_raw[col])
    test_raw[col] = le.transform(test_raw[col])
    ext_raw[col] = le.transform(ext_raw[col])


# Decide if ext data should be used
use_ext = True

if use_ext:
    train = pd.concat([train_raw, ext_raw], ignore_index=True)
else:
    train = train_raw.copy


train_frame = h2o.H2OFrame(train)
test_frame = h2o.H2OFrame(test_raw)
train_frame


train_frame.tail()


# Identify the response and set of predictors
y = "y" # the response
x = list(train_frame.columns)  # The predictors
x.remove(y)
# For binary classification, response should be a factor
train_frame[y] = train_frame[y].asfactor()
# Split the train dataframe
tr, ts = train_frame.split_frame(ratios=[0.9])
# onfirm the shapes of the splits
print(f'Shapes:{[d.shape for d in [tr, ts]]}\n')

# Define and run the H2OAutoML
aml = H2OAutoML(
    max_runtime_secs = 800, 
    # max_models=10, 
    nfolds=6, 
    sort_metric='auc'
)

aml.train(x = x, y = y, training_frame = tr)
# Print Leaderboard (ranked by xval metrics)
lb = aml.leaderboard
display(lb)
# (Optional) Evaluate performance on a test set
perf = aml.leader.model_performance(ts)
perf.auc()


from h2o.explanation import explain

# explain(aml.leader, tr, figsize=(8, 6), top_n_features=3)


perf


lb = aml.leaderboard
lb


perf = aml.leader.model_performance(ts)
perf.auc()


preds_proba_h2o = aml.predict(test_frame)


preds_proba = preds_proba_h2o.as_data_frame()
preds_proba


fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[4, 3, 3])

# target_count = sub_file[target].value_counts()
preds_counts = preds_proba['predict'].value_counts()

ax0 = fig.add_subplot(gs[:, :1])
ax0 = sns.histplot(preds_proba, x='p1', bins=25, kde=True)
ax0.set_title('Distribution of predicted')


ax1 = fig.add_subplot(gs[:, 1:])
ax1 = preds_counts.plot.bar()
for count in ax1.containers:
    ax1.bar_label(count, label_type='center', fmt='%d')
ax1.set_title('Counts of target classes in test prediction')
ax1.grid(False)
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[:1, 2:])
ax2 = preds_counts.plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.2,
                            explode=[0.05, 0.1],
                            cmap='YlGn',
                            startangle=270)
ax2 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.38)
ax2.set_ylabel('')
plt.tight_layout()


sub_file = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sub_file['y'] = np.array(preds_proba['p1'])


sub_file


sub_file.to_csv('submission.csv', index=False)
print('The submission file is ready!')

