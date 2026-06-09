import numpy as np
import pandas as pd
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


print(f"This dataset has {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"There are {train.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(train.duplicated().sum())} duplicates in the dataset.")

# quick look at the data
train.head(3)

# There are quite a few NA's in this dataset, amounting to ~ 1.8% of the dataset.


print(f"This dataset has {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"There are {test.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(test.duplicated().sum())} duplicates in the dataset.")


# quick look at the data
test.head(3)

# There are no NA's in the test, but according to the competition material, we can expect them in test as well.


target = 'gesture'
unique_id = 'sequence_id'
categorical_columns = [col for col in train.columns if col not in [target,unique_id] if train[col].dtype in ['object', 'category']]
numerical_columns = [col for col in train.columns if col not in [target,unique_id]  if train[col].dtype not in ['object','category']]

print(f'There are {len(categorical_columns)} categorical columns: {categorical_columns}')
print(f'There are {len(numerical_columns)} numerical columns: {numerical_columns}')


skewness_threshold = .5 # can tune / experiment with this value
skewed_cols = [col for col in numerical_columns if train[col].skew() > skewness_threshold]

print(f'There are {len(skewed_cols)} skewed columns: {str(skewed_cols)}')

# According to our threshold of .5, the majority of numerical columns are skewed.


imbalance_threshold = 3.33
imbalanced_cols = [col for col in categorical_columns if (train[col].count() / train[col].value_counts().values.min()) > imbalance_threshold]

print(f'There are {len(imbalanced_cols)} imbalanced columns: {str(imbalanced_cols)}')

# According to our imbalance_threshold of 3.33, of our 6 categorical features, 4 are imbalanced.  We note row_id is probably not going to be used as a feature for modelling.


high_cardinality_threshold = 8
high_cardinality_columns = [x for x in categorical_columns if train[x].nunique() > high_cardinality_threshold]

print(f'There are {len(high_cardinality_columns)} high cardinality columns: {str(high_cardinality_columns)}')

# According to our high cardinality threshold, row_id & subject are high cardinality columns.  Again, row_id is probably not going to be used as a feature for modelling.


train_unique_cols = [x for x in train.drop([target],axis=1).columns if x not in test.columns]
print('All columns in train exist in test.') if not train_unique_cols else print(f'The following train columns are not in test: {train_unique_cols}')

test_unique_cols = [x for x in test.columns if x not in train.columns]
print('All columns in test exist in train.') if not test_unique_cols else print(f'The following test columns are not in train: {test_unique_cols}')

train_unique_cols = [x for x in train_demographics.columns if x not in test_demographics.columns]
print('All columns in train_demographics exist in test_demographics.') if not train_unique_cols else print(f'The following train columns are not in test: {train_unique_cols}')

test_unique_cols = [x for x in test_demographics.columns if x not in train_demographics.columns]
print('All columns in test_demographics exist in train_demographics.') if not test_unique_cols else print(f'The following test columns are not in train: {test_unique_cols}')


# According to the competition material, sequence_type, orientation, & behavior are train only.  We discover 'phase' is also only train, which is not unexpected given what we are trying to predict in this competition.


not_skewed_num = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'thm_1', 'thm_3', 'thm_4']
train_samp = train[not_skewed_num].sample(frac=0.1, replace=False, random_state=1)

for col in not_skewed_num:
    _, axes = plt.subplots(1,2,figsize=(8,4),sharex=False,sharey=False)
   
    sns.histplot(data=train_samp.dropna(), x=col,bins=10,ax=axes[0],log_scale=False, color = 'lightblue')
    axes[0].set_title(f'{col}')

    sns.boxplot(data=train_samp.dropna(),x=col,ax=axes[1],showfliers=False, color = 'honeydew')
    axes[1].set_title(f'{col} (no outliers)')

    plt.tight_layout()
    plt.show()

del train_samp
gc.collect()


cat_plot_cols = [col for col in categorical_columns if col not in high_cardinality_columns]
train_samp = train[cat_plot_cols].sample(frac=0.1, replace=False, random_state=1)

for col in cat_plot_cols:
    _, ax = plt.subplots(1,1,figsize=(6,3))

    sns.countplot(data=train_samp.dropna(), y = col, color = 'lightblue')
    ax.set_title(f'{col}')

    plt.tight_layout()
    plt.show()

del train_samp
gc.collect()


print(f"This dataset has {train_demographics.shape[0]} rows and {train_demographics.shape[1]} columns.")
print(f"There are {train_demographics.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(train_demographics.duplicated().sum())} duplicates in the dataset.")

# quick look at the data
train_demographics.head(3)

# There are 0 NA's in this dataset.


print(f"This dataset has {test_demographics.shape[0]} rows and {test_demographics.shape[1]} columns.")
print(f"There are {test_demographics.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(test_demographics.duplicated().sum())} duplicates in the dataset.")

# quick look at the data
test_demographics.head(3)

# There are 0 NA's in this dataset, but according to the competition material, we can expect some.


demo_categorical_columns = [col for col in train_demographics.columns if col not in [target,unique_id] if train_demographics[col].dtype in ['object', 'category']]
demo_numerical_columns = [col for col in train_demographics.columns if col not in [target,unique_id]  if train_demographics[col].dtype not in ['object','category']]

print(f'There are {len(demo_categorical_columns)} categorical columns: {demo_categorical_columns}')
print(f'There are {len(demo_numerical_columns)} numerical columns: {demo_numerical_columns}')


skewness_threshold = .5 # can tune / experiment with this value
demo_skewed_cols = [col for col in demo_numerical_columns if train_demographics[col].skew() > skewness_threshold]

print(f'There are {len(demo_skewed_cols)} skewed columns: {str(demo_skewed_cols)}')

# According to our threshold of .5, 2 of 7 numerical columns in train_demographics are skewed.


imbalance_threshold = 3.33
demo_imbalanced_cols = [col for col in demo_categorical_columns if (train_demographics[col].count() / train_demographics[col].value_counts().values.min()) > imbalance_threshold]

print(f'There are {len(demo_imbalanced_cols)} imbalanced columns: {str(demo_imbalanced_cols)}')

# According to our imbalance_threshold of 3.33, subject is imbalanced. 'subject' is used as an identifier, so it will not be used in modelling efforts.


high_cardinality_threshold = 8
demo_high_cardinality_columns = [x for x in demo_categorical_columns if train_demographics[x].nunique() > high_cardinality_threshold]

print(f'There are {len(demo_high_cardinality_columns)} high cardinality columns: {str(demo_high_cardinality_columns)}')

# According to our high cardinality threshold, subject has a high cardinality.  'subject' is used as an identifier, so it will not be used in modelling efforts.


train_samp = train_demographics[demo_numerical_columns].sample(frac=0.1, replace=False, random_state=1)

for col in demo_numerical_columns:
    _, axes = plt.subplots(1,2,figsize=(8,4),sharex=False,sharey=False)

    sns.histplot(data=train_samp.dropna(), x=col,bins=10,ax=axes[0],log_scale=False, color = 'lightblue')
    axes[0].set_title(f'{col}')

    sns.boxplot(data=train_samp.dropna(),x=col,ax=axes[1],showfliers=True, color = 'honeydew')
    axes[1].set_title(f'{col} (no outliers)')

    plt.tight_layout()
    plt.show()

del train_samp
gc.collect()

# Clearly, some of the numerical features in train_demographics would be better treated as categorical, such as adult_child, sex, & handedness.


train_samp = pd.DataFrame(train[target]).sample(frac=0.1, replace=False, random_state=1).sort_index()

_, ax = plt.subplots(1,1,figsize=(8,4))
sns.countplot(data=train_samp.dropna(), y = target, color = 'lightblue')
ax.set_title(f'{target}')

plt.tight_layout()
plt.show()

del train_samp
gc.collect()

# The most prevalent gestures are 'Neck - scratch' & 'Text on phone'.  The least prevalent are 'Write name on leg' & 'Pinch knee/leg skin'.

