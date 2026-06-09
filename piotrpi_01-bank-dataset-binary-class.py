import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore",  category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,)
warnings.filterwarnings("ignore", category=UserWarning)



train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


display(train_df.head(4))
display(test_df.head(4))


display(train_df.info())
display(test_df.info())


display("Summary for TRAIN data:",train_df.describe())
display("Summary for TEST data:",test_df.describe())


display("TRAIN data", train_df.nunique())
print('\n')
display('TEST data',test_df.nunique())


X = train_df.drop(columns=['id', 'y'])
y = train_df['y']

# Separation of numeric and categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Pipeline: przetwarzanie cech
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# Dimensions reduction with t-SNE
# t-SNE is very slow for large datasets. We will check only 20k random samples
SAMPLE_SIZE = 20000

idx = np.random.choice(len(X_processed), SAMPLE_SIZE, replace=False)
X_sample = X_processed[idx]
y_sample = y.iloc[idx]

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sample)


# Creating plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y_sample,
#    palette='coolwarm',
    s=5,
    alpha=0.6,
    linewidth=0
)
plt.title("t-SNE 2D wizualizacja danych (kolor = y)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title='y')
plt.grid(True)
plt.tight_layout()
plt.show()


# we can create separate column with flag for -1 value
train_df['no_previous_contact'] = (train_df['pdays'] == -1).astype(int)
test_df['no_previous_contact'] = (test_df['pdays'] == -1).astype(int)

# We can create additional column with pdays only without -1 values
train_df['pdays_cleaned'] = train_df['pdays'].where(train_df['pdays'] != -1, np.nan) 
test_df['pdays_cleaned'] = test_df['pdays'].where(test_df['pdays'] != -1, np.nan) 

# We can create additional column with numeric months
train_df['month_as_num'] = train_df['month'].map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11, 'dec':12})
test_df['month_as_num'] = test_df['month'].map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11, 'dec':12})


print("Duplicates in TRAIN data:", train_df.duplicated().sum())
print("Duplicates in TEST data:", test_df.duplicated().sum())


print("Missing values in TRAIN data:\n",train_df.isna().mean().apply(lambda x: f"{x:.2%}"))
print("\nMissing values  in TEST data:\n",test_df.isna().mean().apply(lambda x: f"{x:.2%}"))


for col in test_df.columns:
    if col != 'id' and test_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(train_df[col], label='test', fill=True)
        sns.kdeplot(test_df[col], label='train', fill=True)
        plt.title(f"Drift check for Column: {col}")
        plt.legend()
        plt.show()


feature ='balance'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.99)

# Zoom 
sns.kdeplot(data=train_df, x=feature, ax=axes[0], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[0], label='test', fill=True, alpha=0.4)
axes[0].set_xlim(-1, upper_limit)
axes[0].set_title('Data with zoom')

# Full data
sns.kdeplot(data=train_df, x=feature, ax=axes[1], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[1], label='test', fill=True, alpha=0.4)
axes[1].set_title('Full data')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='duration'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.99)

# Zoom 
sns.kdeplot(data=train_df, x=feature, ax=axes[0], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[0], label='test', fill=True, alpha=0.4)
axes[0].set_xlim(-1, upper_limit)
axes[0].set_title('Data with zoom')

# Full data
sns.kdeplot(data=train_df, x=feature, ax=axes[1], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[1], label='test', fill=True, alpha=0.4)
axes[1].set_title('Full data')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='campaign'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.99)

# Zoom 
sns.kdeplot(data=train_df, x=feature, ax=axes[0], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[0], label='test', fill=True, alpha=0.4)
axes[0].set_xlim(-1, upper_limit)
axes[0].set_title('Data with zoom')

# Full data
sns.kdeplot(data=train_df, x=feature, ax=axes[1], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[1], label='test', fill=True, alpha=0.4)
axes[1].set_title('Full data')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='pdays'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.9)

# Zoom 
sns.kdeplot(data=train_df, x=feature, ax=axes[0], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[0], label='test', fill=True, alpha=0.4)
axes[0].set_xlim(-1, upper_limit)
axes[0].set_title('Data with zoom')

# Full data
sns.kdeplot(data=train_df, x=feature, ax=axes[1], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[1], label='test', fill=True, alpha=0.4)
axes[1].set_title('Full data')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='previous'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.999)

# Zoom 
sns.kdeplot(data=train_df, x=feature, ax=axes[0], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[0], label='test', fill=True, alpha=0.4)
axes[0].set_xlim(-1, upper_limit)
axes[0].set_title('Data with zoom')

# Full data
sns.kdeplot(data=train_df, x=feature, ax=axes[1], label='train', fill=True, alpha=0.4)
sns.kdeplot(data=test_df, x=feature, ax=axes[1], label='test', fill=True, alpha=0.4)
axes[1].set_title('Full data')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


def plot_category_drift(feature):
    pd.concat([
        train_df[feature].value_counts(normalize=True).rename("train"),
        test_df[feature].value_counts(normalize=True).rename("test")
    ], axis=1).plot(kind="bar", title=f"Category drift: {feature}")

columns = [ 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month', 'poutcome'  ]

for col in columns:
    plot_category_drift(col)


sns.heatmap(train_df[['age', 'balance','day', 'duration','campaign', 'pdays','previous', 'month_as_num', 'pdays_cleaned']].corr(),
            annot = True, cmap='coolwarm')


sns.heatmap(test_df[['age', 'balance','day', 'duration','campaign', 'pdays','previous','month_as_num', 'pdays_cleaned']].corr(), 
            annot = True, cmap='coolwarm')


sns.countplot(data=train_df, x='y')
round(train_df['y'].value_counts(normalize=True)*100,2)


columns = [ 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',  'poutcome', 
            'month_as_num', 'no_previous_contact']
for col in columns:
    train_df.groupby([col,'y']).size().unstack().plot(kind='bar', stacked=True, title=col)
    plt.show()
    print('Percentage summary:')
    display((pd.crosstab(train_df[col], train_df["y"], normalize='index') * 100).round(1))
    print('Quantitative summary:')
    display((pd.crosstab(train_df[col], train_df["y"])))


columns = [ 'age','balance', 'day', 'duration', 'campaign', 'pdays', 'previous',  'pdays_cleaned', 'month_as_num']

for feature in columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, label='y = 0', fill=True, alpha=0.4)
    sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, label='y = 1', fill=True, alpha=0.4)
    plt.title(f'KDE for {feature}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


feature ='balance'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.99)

# Zoom
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[0], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[0], label='y = 1', fill=True, alpha=0.4)
axes[0].set_xlim(0, upper_limit)
axes[0].set_title(f'Zoom for {feature}')

# Full
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[1], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[1], label='y = 1', fill=True, alpha=0.4)
axes[1].set_title(f'Full data for {feature}')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='duration'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.995)

# Zoom
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[0], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[0], label='y = 1', fill=True, alpha=0.4)
axes[0].set_xlim(0, upper_limit)
axes[0].set_title(f'Zoom for {feature}')

# Full
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[1], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[1], label='y = 1', fill=True, alpha=0.4)
axes[1].set_title(f'Full data for {feature}')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='campaign'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.97)

# Zoom
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[0], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[0], label='y = 1', fill=True, alpha=0.4)
axes[0].set_xlim(0, upper_limit)
axes[0].set_title(f'Zoom  for {feature}')

# Full
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[1], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[1], label='y = 1', fill=True, alpha=0.4)
axes[1].set_title(f'Full data for {feature}')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


feature ='previous'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.995)

# Zoom
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[0], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[0], label='y = 1', fill=True, alpha=0.4)
axes[0].set_xlim(0, upper_limit)
axes[0].set_title(f'Zoom for {feature}')

# Full
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[1], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[1], label='y = 1', fill=True, alpha=0.4)
axes[1].set_title(f'Full data for {feature}')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


columns = [ 'age','balance', 'day', 'duration', 'campaign', 'pdays', 'previous',  'pdays_cleaned', 'month_as_num']

for col in columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='y', y=col, data=train_df)
    plt.title(f'Boxplot of {col} by y label column')
    plt.show()


def stability(x):
    if x=='management' or x=='technician' or x=='admin.' or x=='services':
        return 0
    elif x=='blue-collar' or x=='self-employed' or x=='entrepreneur':
        return 1
    else:
        return 2

train_df['job_stability'] = train_df['job'].apply(stability)
print('Percentage summary:')
display((pd.crosstab(train_df['job_stability'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['job_stability','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='job_stability')
train_df.groupby(['job','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='job')
plt.tight_layout()
plt.show()


def stability(x):
    #High
    if x=='management' or x=='entrepreneur' or x=='self-employed':
        return 0
    # Middle
    elif x=='technician' or x=='admin.' or x=='services':
        return 1
    # Low
    elif x=='blue-collar' or x=='housemaid':
        return 2
    # No earnings
    else:
        return 3

train_df['job_earnings'] = train_df['job'].apply(stability)
print('Percentage summary:')
display((pd.crosstab(train_df['job_earnings'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['job_earnings','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='job_earnings')
train_df.groupby(['job','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='job')
plt.tight_layout()
plt.show()


def stability(x):
    #Not working
    if x=='retired' or x=='student' or x=='unemployed' or x=='unknown': 
        return 0
    # working
    else:
        return 1

train_df['job_is_working'] = train_df['job'].apply(stability)
print('Percentage summary:')
display((pd.crosstab(train_df['job_is_working'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['job_is_working','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='job_is_working')
train_df.groupby(['job','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='job')
plt.tight_layout()
plt.show()


train_df['prev_success']=(train_df['poutcome'] == 'success').astype(int)
print('prev_success - Percentage summary:')
display((pd.crosstab(train_df['prev_success'], train_df["y"], normalize='index') * 100).round(1))
print('poutcome - Percentage summary:')
display((pd.crosstab(train_df['poutcome'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['prev_success','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='prev_success')
train_df.groupby(['poutcome','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='poutcome')
plt.tight_layout()
plt.show()



train_df['duration_mult_age'] = train_df['duration'] * train_df['age']

feature ='duration_mult_age'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
upper_limit = train_df[feature].quantile(0.97)

# Zoom
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[0], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[0], label='y = 1', fill=True, alpha=0.4)
axes[0].set_xlim(0, upper_limit)
axes[0].set_title(f'Zoom  for {feature}')

# Full
sns.kdeplot(data=train_df[train_df['y'] == 0], x=feature, ax=axes[1], label='y = 0', fill=True, alpha=0.4)
sns.kdeplot(data=train_df[train_df['y'] == 1], x=feature, ax=axes[1], label='y = 1', fill=True, alpha=0.4)
axes[1].set_title(f'Full data for {feature}')

for ax in axes:
    ax.legend()
    ax.grid(True)
    
plt.tight_layout()
plt.show()


train_df['job_and_education'] = train_df['education'] + " " + train_df['job']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['job','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='job')
train_df.groupby(['education','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='education')
plt.tight_layout()
plt.show()

train_df.groupby(['job_and_education','y']).size().unstack().plot(kind='bar', stacked=True, figsize=(14,5), title='job_and_education')
plt.tight_layout()
plt.show()
print('Percentage summary:')
display((pd.crosstab(train_df['job_and_education'], train_df["y"], normalize='index') * 100).round(1))



train_df['contact_plus_poutcome'] = train_df['contact'] + ' ' + train_df['poutcome']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['contact','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='contact')
train_df.groupby(['poutcome','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[1], title='poutcome')
plt.tight_layout()
plt.show()

train_df.groupby(['contact_plus_poutcome','y']).size().unstack().plot(kind='bar', stacked=True, figsize=(14,5), title='contact_plus_poutcome')
plt.tight_layout()
plt.show()
print('Percentage summary:')
display((pd.crosstab(train_df['contact_plus_poutcome'], train_df["y"], normalize='index') * 100).round(1))


def stability(x):
    if x < 30: 
        return 0
    elif x >= 30 and x <= 60:
        return 1
    else:
        return 2

train_df['age_group'] = train_df['age'].apply(stability)

print('Percentage summary:')
display((pd.crosstab(train_df['age_group'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['age_group','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='age_group')
sns.kdeplot(data=train_df[train_df['y'] == 0], x='age', label='y = 0', fill=True, alpha=0.4, ax=axes[1])
sns.kdeplot(data=train_df[train_df['y'] == 1], x='age', label='y = 1', fill=True, alpha=0.4, ax=axes[1])
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()



def month(x):
    if x < 5:
        return 0
    elif x >= 5 and x <=8:
        return 1
    else:
        return 2 

train_df['month_group'] = train_df['month_as_num'].apply(month)

print('Percentage summary:')
display((pd.crosstab(train_df['month_group'], train_df["y"], normalize='index') * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.groupby(['month_group','y']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='month_group')
sns.kdeplot(data=train_df[train_df['y'] == 0], x='month_as_num', label='y = 0', fill=True, alpha=0.4, ax=axes[1])
sns.kdeplot(data=train_df[train_df['y'] == 1], x='month_as_num', label='y = 1', fill=True, alpha=0.4, ax=axes[1])
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()



def duration(x):
    # short
    if x < 120:
        return 0
    # medium
    elif x >= 120 and x <=250:
        return 1
    # long
    else:
        return 2

train_df['duration_long'] = train_df['duration'].apply(duration)

train_df.groupby(['duration_long','y']).size().unstack().plot(kind='bar', stacked=True, title='duration_long')
plt.show()
print('Percentage summary:')
display((pd.crosstab(train_df['duration_long'], train_df["y"], normalize='index') * 100).round(1))


mapping = ({'yes':1,"no":0})
train_df['loan_code'] = train_df['loan'].map(mapping)
train_df['housing_code'] = train_df['housing'].map(mapping)

train_df['loan_plus_housing'] = (train_df['loan_code']+train_df['housing_code']).astype(int)

train_df.groupby(['loan_plus_housing','y']).size().unstack().plot(kind='bar', stacked=True, title='loan_plus_housing')
plt.show()
print('Percentage summary:')
display((pd.crosstab(train_df['loan_plus_housing'], train_df["y"], normalize='index') * 100).round(1))


train_df['deep_debt'] = ((train_df['balance'] < 0 ) & (train_df['loan_plus_housing'] == 2)).astype(int)

train_df.groupby(['deep_debt','y']).size().unstack().plot(kind='bar', stacked=True, title='deep_debt')
plt.show()
print('Percentage summary:')
display((pd.crosstab(train_df['deep_debt'], train_df["y"], normalize='index') * 100).round(1))


# Data shows that each month has 31 days so to keep it simple we will treat each month as it is 31 days
DAYS_IN_YEAR = 31 * 12  

# Number of the day in whole year
train_df['day_of_year'] = (train_df['month_as_num'] - 1) * 31 + train_df['day']
# Coding as cyclic data
train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / DAYS_IN_YEAR)
train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / DAYS_IN_YEAR)

# SAme operation for test data
test_df['day_of_year'] = (test_df['month_as_num'] - 1) * 31 + test_df['day']
test_df['day_of_year_sin'] = np.sin(2 * np.pi * test_df['day_of_year'] / DAYS_IN_YEAR)
test_df['day_of_year_cos'] = np.cos(2 * np.pi * test_df['day_of_year'] / DAYS_IN_YEAR)



counts = train_df.groupby(['day_of_year', 'y']).size().reset_index(name='count')

# Plots
plt.figure(figsize=(16, 5))  
sns.lineplot(data=counts, x='day_of_year', y='count', hue='y', marker=".")
plt.title("Number of y=0 and y=1 in time")
plt.xlabel("Number of day in year")
plt.ylabel("Count of category y")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 5))
sns.kdeplot(
    data=train_df,
    x='day_of_year',
    hue='y',
    fill=True,     
    common_norm=False
)
plt.title("KDE for y=0 and y=1")
plt.xlabel("Day in year")
plt.ylabel("Density")
plt.tight_layout()
plt.show()


train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

df_all = pd.concat([train_df[['day_of_year', 'dataset']], test_df[['day_of_year', 'dataset']]])

plt.figure(figsize=(16, 5))
sns.histplot(data=df_all, x='day_of_year', hue='dataset', element='step', stat='density', common_norm=False)
plt.title("Drift check for train/test data")
plt.show()

