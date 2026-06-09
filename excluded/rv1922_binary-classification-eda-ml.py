import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as pl  
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp
import plotly.figure_factory as ff  
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pio.renderers.default = 'iframe_connected'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.head()


train.info()


train.describe().T


print("Null Values in Train data:")
train.isnull().sum()


print("Number of Rows in Train data:",train.shape[0])
print("-"*30)
print("Number of Columns in Train data:",train.shape[1])
print("-"*30)
print("Number of Rows in Test data:",test.shape[0])
print("-"*30)
print("Number of Columns in Test data:",test.shape[1])
print("-"*30)


print("Check for Duplicated Rows in Train data:")
print(train.duplicated().sum())
print("-"*30)
print("Check for Duplicated Rows in Train data:")
print(test.duplicated().sum())


print("Numeric Col Names of Train data:",train.select_dtypes(include=['number']).columns)
print("-"*30)
print("Categorical Col Names of Train data:",train.select_dtypes(include=['object']).columns)


num_col = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays',
       'previous']
cat_col = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'poutcome']


binary_cols = ['default','loan','housing']
bar_chart_cols = ['job', 'marital', 'education', 'contact',
       'month', 'poutcome']
target_col = 'y'


for col in cat_col:
    print(f"â”€â”€ Value counts for '{col}' â”€â”€")
    print(train[col].value_counts())
    print()


counts = train['job'].value_counts().reset_index()
counts.columns = ['job', 'count']  

fig = px.bar(
    counts,
    x='job',
    y='count',
    title='Distribution of Job',
    color_discrete_sequence=['#FF4E50'],  
    template='plotly_dark'
)
fig.update_layout(xaxis_title='Job', yaxis_title='Count', title_x=0.5)
fig.show()


marital_counts = train['marital'].value_counts().reset_index()
marital_counts.columns = ['marital', 'count']

fig = px.bar(
    marital_counts,
    x='marital',
    y='count',
    title='Distribution of Marital Status',
    color_discrete_sequence=['#FF4E50'],
    template='plotly_dark',
    width=600,   
    height=400
)

fig.update_layout(
    xaxis_title='Marital Status',
    yaxis_title='Count',
    title_x=0.5
)

fig.show()


education_counts = train['education'].value_counts().reset_index()
education_counts.columns = ['education', 'count']

fig = px.bar(
    education_counts,
    x='education',
    y='count',
    title='Distribution of Education Levels',
    color_discrete_sequence=['#FF4E50'],
    template='plotly_dark',
    width=600,   
    height=400
)
fig.update_layout(
    xaxis_title='Education Level',
    yaxis_title='Count',
    title_x=0.5
)

fig.show()


contact_counts = train['contact'].value_counts().reset_index()
contact_counts.columns = ['contact', 'count']

fig = px.bar(
    contact_counts,
    x='contact',
    y='count',
    title='Distribution of Contact Communication Type',
    color_discrete_sequence=['#FF4E50'],
    template='plotly_dark',
    width=600,   
    height=400
)
fig.update_layout(xaxis_title='Contact Type', yaxis_title='Count', title_x=0.5)
fig.show()


month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_dtype = CategoricalDtype(categories=month_order, ordered=True)
train['month'] = train['month'].astype(month_dtype)

month_counts = train['month'].value_counts().sort_index().reset_index()
month_counts.columns = ['month', 'count']

fig = px.bar(
    month_counts,
    x='month',
    y='count',
    title='Distribution of Contact Month',
    color_discrete_sequence=['#FF4E50'],
    template='plotly_dark'
)
fig.update_layout(xaxis_title='Month', yaxis_title='Count', title_x=0.5)
fig.show()


poutcome_counts = train['poutcome'].value_counts().reset_index()
poutcome_counts.columns = ['poutcome', 'count']

fig = px.bar(
    poutcome_counts,
    x='poutcome',
    y='count',
    title='Distribution of Previous Campaign Outcome',
    color_discrete_sequence=['#FF4E50'],
    template='plotly_dark',
    width=600,   
    height=400
)
fig.update_layout(xaxis_title='Previous Outcome', yaxis_title='Count', title_x=0.5)
fig.show()


default_counts = train['default'].value_counts().reset_index()
default_counts.columns = ['default', 'count']

fig = px.pie(
    default_counts,
    names='default',
    values='count',
    title='Distribution of Default',
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig.update_traces(textinfo='percent+label')

fig.update_layout(
    title_x=0.5,
    height=600,
    width=600
)

fig.show()


loan_counts = train['loan'].value_counts().reset_index()
loan_counts.columns = ['loan', 'count']

fig = px.pie(
    loan_counts,
    names='loan',
    values='count',
    title='Distribution of Personal Loan',
    color_discrete_sequence=px.colors.sequential.RdBu
)

fig.update_traces(textinfo='percent+label')
fig.update_layout(
    title_x=0.5,
    height=600,
    width=600
)

fig.show()


default_counts = train['default'].value_counts().reset_index()
default_counts.columns = ['default', 'count']

fig = px.pie(
    default_counts,
    names='default',
    values='count',
    title='Distribution of Credit Default',
    color_discrete_sequence=px.colors.sequential.RdBu
)

fig.update_traces(textinfo='percent+label')
fig.update_layout(
    title_x=0.5,
    height=600,
    width=600
)

fig.show()


for col in num_col:
    print(f"â”€â”€ Summary Statistics for Numerical Features '{col}' â”€â”€")
    print(train[col].describe())
    print()


for col in num_col:
    print(f"â”€â”€ Value  for '{col}' â”€â”€")
    print(train[col].value_counts())
    print()


y_counts = train['y'].value_counts().reset_index()
y_counts.columns = ['y', 'count']

fig = px.pie(
    y_counts,
    names='y',
    values='count',
    title='Distribution of Target Label',
    color_discrete_sequence=px.colors.sequential.RdBu
)

fig.update_traces(textinfo='percent+label')
fig.update_layout(
    title_x=0.5,
    height=600,
    width=600
)

fig.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['age'], bins=30, kde=True, color='#FF4E50')
plt.title('Histogram of age')
plt.xlabel('age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['balance'], bins=100, kde=True, color='#FF4E50')
plt.title('Histogram of balance')
plt.xlabel('balance')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['day'], bins=30, kde=True, color='#FF4E50')
plt.title('Histogram of day')
plt.xlabel('day')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['duration'], bins=100, kde=True, color='#FF4E50')
plt.xlim(0, 300)
plt.title('Histogram of duration (Zoomed In)')
plt.xlabel('duration')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['campaign'], bins=50, kde=False, color='#FF4E50')
plt.title('Histogram of campaign')
plt.xlabel('campaign')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['pdays'], bins=100, kde=False, color='#FF4E50')
plt.title('Histogram of pdays')
plt.xlabel('pdays')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(train['previous'], bins=50, kde=False, color='#FF4E50')
plt.title('Histogram of previous')
plt.xlabel('previous')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


fig = px.scatter(
    train,
    x="age",
    y="balance",
    color="marital",
    facet_col="housing",
    facet_row="loan",
    color_discrete_sequence=px.colors.sequential.Sunsetdark
)
fig.show()


fig = px.scatter_matrix(
    train,
    dimensions=["age", "balance", "duration", "campaign"],
    color="y",
    title="Pairwise Scatter Matrix",
    labels={
        "age": "Client Age",
        "balance": "Account Balance",
        "duration": "Call Duration",
        "campaign": "Campaign Contacts"
    },
    color_discrete_sequence=px.colors.sequential.Sunsetdark
)
fig.show()


print("Chi-Square Test for Categorical Features:")
for col in cat_col:
    contingency_table = pd.crosstab(train[col], train[target_col])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    print(f"\n'{col}' vs '{target_col}':")
    print(f"  Chi-Square: {chi2:.4f}")
    print(f"  p-value: {p:.4f}")


corr = train[num_col].corr()
print(corr)


plt.figure(figsize=(8, 7))
sns.heatmap(corr, annot=True, fmt=".2f",        
    cmap="coolwarm",  center=0, linewidths=0.5    
)

plt.title("Correlation Matrix Heatmap", fontsize=16, pad=20)
plt.show()


# ==================== FEATURE ENGINEERING ====================

# ---------- TRAIN ----------
train['age_group'] = pd.cut(train['age'], 
                           bins=[0, 25, 35, 50, 65, 100], 
                           labels=['young', 'adult', 'middle_aged', 'senior', 'elderly'])

train['balance_positive'] = (train['balance'] > 0).astype(int)
train['balance_high'] = (train['balance'] > train['balance'].quantile(0.75)).astype(int)
train['balance_negative'] = (train['balance'] < 0).astype(int)
train['balance_log'] = np.log1p(train['balance'] + abs(train['balance'].min()) + 1)

train['duration_minutes'] = train['duration'] / 60
train['duration_group'] = pd.cut(train['duration'], 
                               bins=[0, 120, 300, 600, float('inf')], 
                               labels=['very_short', 'short', 'medium', 'long'])

train['campaign_intensive'] = (train['campaign'] > 3).astype(int)
train['first_contact'] = (train['campaign'] == 1).astype(int)

train['contacted_before'] = (train['previous'] > 0).astype(int)
train['pdays_contacted'] = (train['pdays'] != -1).astype(int)

job_groups = {
    'management': 'white_collar',
    'admin.': 'white_collar',
    'technician': 'skilled',
    'services': 'service',
    'blue-collar': 'manual',
    'retired': 'inactive',
    'unemployed': 'inactive',
    'student': 'inactive',
    'self-employed': 'entrepreneur',
    'entrepreneur': 'entrepreneur',
    'housemaid': 'service',
    'unknown': 'unknown'
}
train['job_group'] = train['job'].map(job_groups)

education_order = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0}
train['education_level'] = train['education'].map(education_order)

month_success_rate = {
    'jan': 0.073, 'feb': 0.058, 'mar': 0.179, 'apr': 0.293,
    'may': 0.073, 'jun': 0.065, 'jul': 0.073, 'aug': 0.072,
    'sep': 0.167, 'oct': 0.282, 'nov': 0.104, 'dec': 0.475
}
train['month_success_rate'] = train['month'].map(month_success_rate)

economic_months = ['mar', 'apr', 'sep', 'oct', 'dec']
train['economic_month'] = train['month'].isin(economic_months).astype(int)

train['age_job_risk'] = ((train['age'] < 30) & 
                       (train['job'].isin(['student', 'unemployed']))).astype(int)

train['balance_loan_risk'] = ((train['balance'] < 0) & 
                            (train['loan'] == 'yes')).astype(int)

train['duration_campaign_ratio'] = train['duration'] / (train['campaign'] + 1)
train['prev_success'] = ((train['poutcome'] == 'success')).astype(int)


test['age_group'] = pd.cut(test['age'], 
                          bins=[0, 25, 35, 50, 65, 100], 
                          labels=['young', 'adult', 'middle_aged', 'senior', 'elderly'])

test['balance_positive'] = (test['balance'] > 0).astype(int)
test['balance_high'] = (test['balance'] > test['balance'].quantile(0.75)).astype(int)
test['balance_negative'] = (test['balance'] < 0).astype(int)
test['balance_log'] = np.log1p(test['balance'] + abs(test['balance'].min()) + 1)

test['duration_minutes'] = test['duration'] / 60
test['duration_group'] = pd.cut(test['duration'], 
                              bins=[0, 120, 300, 600, float('inf')], 
                              labels=['very_short', 'short', 'medium', 'long'])

test['campaign_intensive'] = (test['campaign'] > 3).astype(int)
test['first_contact'] = (test['campaign'] == 1).astype(int)

test['contacted_before'] = (test['previous'] > 0).astype(int)
test['pdays_contacted'] = (test['pdays'] != -1).astype(int)

test['job_group'] = test['job'].map(job_groups)
test['education_level'] = test['education'].map(education_order)
test['month_success_rate'] = test['month'].map(month_success_rate)
test['economic_month'] = test['month'].isin(economic_months).astype(int)

test['age_job_risk'] = ((test['age'] < 30) & 
                      (test['job'].isin(['student', 'unemployed']))).astype(int)

test['balance_loan_risk'] = ((test['balance'] < 0) & 
                           (test['loan'] == 'yes')).astype(int)

test['duration_campaign_ratio'] = test['duration'] / (test['campaign'] + 1)
test['prev_success'] = ((test['poutcome'] == 'success')).astype(int)


train.head()


X = train.drop(columns=['id', 'y'])
y = train['y']
test_ids = test['id'].copy()
test = test.drop(columns=['id'])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# CatBoost Pools
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 4000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'verbose': 0,
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, use_best_model=True, verbose=0)

    preds = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, preds)
    return roc_auc


#study = optuna.create_study(direction='maximize', study_name='catboost_optuna_gpu')
#study.optimize(objective, n_trials=50, timeout=1800)


#print("Best AUC: ", study.best_value)
#print("Best Params: ", study.best_params)


best_params = {
    'iterations': 3641,
    'depth': 5,
    'learning_rate': 0.043772074179388616,
    'l2_leaf_reg': 0.46826504842994493,
    'random_strength': 5.468135444185625,
    'bagging_temperature': 0.6148695408854639,
    'border_count': 207,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'verbose': 100,
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42,
}

cat_clf = CatBoostClassifier(**best_params)

cat_clf.fit(
    Pool(X, y, cat_features=cat_features),
    verbose=100
)


test_pred = cat_clf.predict_proba(test.astype('str'))[:, 1]


submission['y'] = test_pred
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved to submission.csv")


submission.head()


plt.figure(figsize=(8, 5))
plt.hist(submission['y'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Predicted Probabilities (submission["y"])')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

