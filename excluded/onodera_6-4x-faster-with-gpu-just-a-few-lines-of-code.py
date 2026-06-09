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
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
train.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].plot(kind='bar', figsize=(12, 6))
plt.title('Descriptive Statistics for Continuous Features')
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

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 400)

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


%%time

from sklearn.metrics import roc_auc_score

def ROC_AUC(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)


cat_cols = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

encode_c = {'cat_c': cat_cols}

base = AbdBase(train_data=train, test_data=test, target_column='y',gpu=True, prob=True, test_prob=True,
                 problem_type="classification", metric="custom", seed=SEED,ohe_fe=False,ordinal_encoder=encode_c,
                 n_splits=5,early_stop=True,num_classes=2,cat_features=False,custom_metric=ROC_AUC,
                 fold_type='SKF')


%%time

ParamsXgb = {
    'n_estimators': 40000,
    'learning_rate': 0.0358306214515723,
    'max_depth': 6,
    'min_child_weight': 83,
    'subsample': 0.8700304020753131,
    'colsample_bytree': 0.6169349166144594,
    'reg_alpha': 3.700714656885025,
    'reg_lambda': 4.709578317972932,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss', 
    'gpu_id': 0,
}

results_Xgb_1 = base.Train_ML(ParamsXgb,'XGB',e_stop=150)


%%time

def save_outputs(base_file_name, oof, pred):
    oof_df = pd.DataFrame(oof)
    pred_df = pd.DataFrame(pred)

    oof_df.to_csv(f"{base_file_name}_OOF.csv", index=False)
    pred_df.to_csv(f"{base_file_name}_PREDS.csv", index=False)

save_outputs('XGB_0.9740',results_Xgb_1[0], results_Xgb_1[1])
mp = results_Xgb_1[1]

sample['y'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()




