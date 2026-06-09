%%time

import pandas as pd 
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase
SEED = 42



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

original['y'] = original['y'].map({'no': 0, 'yes': 1})


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid", font_scale=1.1)

# 1) è®¡ç®—éœ€è¦�çš„ç»Ÿè®¡é‡�
num_cols = train.select_dtypes(include=['float64', 'int64']).columns   # å�ªä¿�ç•™æ•°å€¼åˆ—
desc = train[num_cols].describe().T                                    # åŸºç¡€æ��è¿°ç»Ÿè®¡
desc['skew'] = train[num_cols].skew()                                  # å��åº¦
desc['kurtosis'] = train[num_cols].kurt()                              # å³°åº¦

# 2) æŠŠè¦�ç”»çš„åˆ—æŒ‘å‡ºæ�¥ï¼ˆä½ ä¹Ÿå�¯ä»¥æŒ‰éœ€å¢�åˆ ï¼‰
plot_cols = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']
plot_df = desc[plot_cols].reset_index().melt(id_vars='index',
                                             var_name='stat',
                                             value_name='value')

# 3) ç”»å›¾
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# å·¦å›¾ï¼šå�Ÿå§‹å°ºåº¦
sns.barplot(data=plot_df[~plot_df['stat'].isin(['skew', 'kurtosis'])],
            x='index', y='value', hue='stat', ax=ax[0])
ax[0].set_title('Descriptive Statistics (mean, std, quartiles â€¦)')
ax[0].tick_params(axis='x', rotation=90)

# å�³å›¾ï¼šskew & kurtosisï¼ˆé€šå¸¸æ•°å€¼è¾ƒå°�ï¼Œå�•å¼€ä¸€å›¾ï¼‰
skew_kurt = plot_df[plot_df['stat'].isin(['skew', 'kurtosis'])]
sns.barplot(data=skew_kurt, x='index', y='value', hue='stat', ax=ax[1])
ax[1].set_title('Skewness & Kurtosis')
ax[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*use_inf_as_na.*")

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
sns.histplot(train['age'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Age')


plt.subplot(1, 2, 2)
sns.histplot(train['balance'], kde=True, color='orange', bins=20)
plt.title('Distribution of Balance')

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=train, palette='Set2')
plt.title('Distribution of Target Variable (y)')
plt.show()


categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

plt.figure(figsize=(14, 12))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=train, palette='Set2')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))


plt.subplot(1, 3, 1)
sns.boxplot(x='y', y='age', data=train, palette='Set2')
plt.title('Age vs Target Variable (y)')


plt.subplot(1, 3, 2)
sns.boxplot(x='y', y='balance', data=train, palette='Set2')
plt.title('Balance vs Target Variable (y)')


plt.subplot(1, 3, 3)
sns.boxplot(x='y', y='duration', data=train, palette='Set2')
plt.title('Duration vs Target Variable (y)')

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))


sns.countplot(x='job', hue='y', data=train, palette='Set2')
plt.title('Job vs Target Variable (y)')
plt.xticks(rotation=45)
plt.show()


sns.countplot(x='marital', hue='y', data=train, palette='Set2')
plt.title('Marital Status vs Target Variable (y)')
plt.xticks(rotation=45)
plt.show()


%%time
COLS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome',]



def NEW_FE(df):
    
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 800)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 800)

    return df

train = NEW_FE(train)
test = NEW_FE(test)


plt.figure(figsize=(10, 6))
sns.violinplot(x='y', y='balance_log', data=train)
plt.title('Relationship between Log of Balance and Service Acceptance')
plt.xlabel('Accepts Deposit Service (0=No, 1=Yes)')
plt.ylabel('Log of Balance')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()


job_edu_counts = train.groupby('job_edu')['y'].value_counts(normalize=True).unstack().fillna(0)

plt.figure(figsize=(12, 7))
job_edu_counts.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Proportion of Customers Accepting the Service by Job and Education Level')
plt.xlabel('Job and Education Level')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Accepts Service', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


print(job_edu_counts.columns)


# è®¡ç®—å�„ç»„ä¸­ y çš„æ¯”ä¾‹
job_edu_counts = (
    train.groupby('job_edu')['y']
         .value_counts(normalize=True)
         .unstack()
         .fillna(0)
)

# æŒ‰â€œYesâ€�åˆ—ä»�é«˜åˆ°ä½�æ�’åº�
job_edu_counts = job_edu_counts.sort_values(by=1, ascending=False)

# ç»˜å›¾
plt.figure(figsize=(12, 7))
job_edu_counts.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Proportion of Customers Accepting the Service by Job and Education Level')
plt.xlabel('Job and Education Level')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Accepts Service', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# Create a crosstab to count the data
contact_y_crosstab = pd.crosstab(train['contacted_before'], train['y'], normalize='index')

contact_y_crosstab.plot(kind='bar', figsize=(8, 6))
plt.title('Relationship Between Being Contacted Before and Service Acceptance')
plt.xlabel('Contacted Before (0=No, 1=Yes)')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Accepts Service', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


import seaborn as sns, matplotlib.pyplot as plt

plt.figure(figsize=(12,3))

# --- duration_sin ä¸� y çš„å…³ç³» ---
plt.subplot(1,2,1)
sns.kdeplot(data=train, x='duration_sin', hue='y', fill=True, palette={0:'#1f77b4', 1:'#ff7f0e'})
plt.title('Distribution of duration_sin by y')
# --- duration_cos ä¸� y çš„å…³ç³» ---
plt.subplot(1,2,2)
sns.kdeplot(data=train, x='duration_cos', hue='y', fill=True, palette={0:'#1f77b4', 1:'#ff7f0e'})
plt.title('Distribution of duration_cos by y')

plt.tight_layout()
plt.show()


cat_cols = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

mean = train['y'].mean() 

for c in COLS:
    new_col = f"{c}_mean_target_orig"
    train[new_col] = train[c].map(original.groupby(c)['y'].mean())
    train[new_col] = train[new_col].fillna(mean)
    test[new_col] = test[c].map(original.groupby(c)['y'].mean())
    test[new_col] = test[new_col].fillna(mean)



for c in COLS:
    mapping_count = original[c].value_counts()
    train[f"{c}_count"] = train[c].map(mapping_count).fillna(0)
    test[f"{c}_count"] = test[c].map(mapping_count).fillna(0)


def update(df):

    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df

train = update(train)
test = update(test)

train.head()


# %%time

# from sklearn.metrics import roc_auc_score

# def ROC_AUC(y_true, y_pred_proba):
#     return roc_auc_score(y_true, y_pred_proba)


# cat_cols = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

# encode_c = {'cat_c': cat_cols}

# base = AbdBase(train_data=train, test_data=test, target_column='y',gpu=False, prob=True, test_prob=True,
#                  problem_type="classification", metric="custom", seed=SEED,ohe_fe=False,ordinal_encoder=encode_c,
#                  n_splits=5,early_stop=True,num_classes=2,cat_features=False,custom_metric=ROC_AUC,
#                  fold_type='SKF')


# %%time

# ParamsLgb = {'n_estimators': 40000, 'learning_rate': 0.0358306214515723, 'num_leaves': 228, 'max_depth': 6,
#              'min_child_samples': 83, 'subsample': 0.8700304020753131, 'colsample_bytree': 0.6169349166144594,
#              'reg_alpha': 3.700714656885025, 'reg_lambda': 4.709578317972932,"objective": "binary",
#              "metric": "binary_logloss"}

# results_Lgb_1 = base.Train_ML(ParamsLgb,'LGBM',e_stop=150)


import pandas as pd

sub1 = pd.read_csv("/kaggle/input/s05e08data/submission_97763.csv")
sub2 = pd.read_csv("/kaggle/input/s05e08data/submission_97772.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
y = 0.3*sub1['y'] + 0.7*sub2['y']

sub['y'] = y

sub.to_csv("submission.csv", index=False)


# %%time

# def save_outputs(base_file_name, oof, pred):
#     oof_df = pd.DataFrame(oof)
#     pred_df = pd.DataFrame(pred)

#     oof_df.to_csv(f"{base_file_name}_OOF.csv", index=False)
#     pred_df.to_csv(f"{base_file_name}_PREDS.csv", index=False)

# save_outputs('LGBM_0.9743',results_Lgb_1[0], results_Lgb_1[1])
# mp = results_Lgb_1[1]

# sample['y'] = mp
# sample.to_csv('submission.csv', index=False)
# sample.head()

