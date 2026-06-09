import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import levene
from sklearn.metrics import mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mutual_info_score

import warnings
warnings.simplefilter('ignore')

sns.set_style("whitegrid")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print('=' * 15 + ' First 5 Rows of Train Data ' + '=' * 15)
display(df_train.head())

print('\n')

print('=' * 15 + ' First 5 Rows of Test Data ' + '=' * 15)
display(df_test.head())


print('=' * 15 + ' Info of Train Data ' + '=' * 15)
display(df_train.info())

print('\n')

print('=' * 15 + ' Info of Trest Data ' + '=' * 15)
display(df_test.info())


features = [col for col in list(df_test.columns) if col != 'id']
num_features = list(df_test[features].select_dtypes(include=['float64']).columns)
cat_features = list(df_test[features].select_dtypes(include=['object']).columns)
print('All Features:')
print(features)
print('\n')

print('Numerical Features:')
print(num_features)
print('\n')

print('Categorical Features:')
print(cat_features)


print('=' * 15 + ' Summary Statistics of Train Data ' + '=' * 15)
display(df_train.describe(include='all'))

print('\n')

print('=' * 15 + ' Summary Statistics of Test Data ' + '=' * 15)
display(df_test.describe(include='all'))


# Testing for Differences in Variance between Train and Test
levene_results = {}

for col in num_features:
    s1 = df_train[col].dropna()
    s2 = df_test[col].dropna()

    stat, p = levene(s1, s2)
    levene_results[col] = {
        'Variance in Train': s1.var(),
        'Variance in Test': s2.var(),
        'p-value': p
    }

levene_df = pd.DataFrame.from_dict(levene_results, orient='index')
levene_df['Significant'] = levene_df.apply(lambda x: True if x['p-value'] <= 0.05 else False, axis=1)
print('=' * 15 + ' Statistical Significance in Difference in Variances ' + '=' * 15)
display(levene_df.sort_values('p-value'))


# Testing for Differences in mean between Train and Test
ttest_results = {}

for col in features:
    s1 = df_train[col].isna().astype(int)
    s2 = df_test[col].isna().astype(int)
    
    # Perform two-sample t-test
    stat, p = ttest_ind(s1, s2, equal_var=True) # Using equal_variance since earlier shown that variance between train and test are the same

    ttest_results[col] = {
        'Train Mean': s1.mean(),
        'Test Mean': s2.mean(),
        'p-value': p
    }

# Raw ttest results
ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index')
print('=' * 15 + ' Statistical Significance in Difference Between Means ' + '=' * 15)
display(ttest_df.sort_values('p-value'))

print('\n')

# with bonferroni correction
corrected_ttest_res = multipletests(ttest_df['p-value'], alpha=0.05, method='bonferroni')
ttest_df['p-value corrected'] = corrected_ttest_res[1]
ttest_df['Significant'] = corrected_ttest_res[0]
print('=' * 15 + ' Statistical Significance in Difference Between Means (Corrected) ' + '=' * 15)
display(ttest_df.sort_values('p-value corrected'))


# checking missing values numerically
print('=' * 15 + ' Missing Values in Train Data ' + '=' * 15)
display(df_train[features].isna().sum())

print('=' * 15 + ' Proportion of Values Missing in Train Data ' + '=' * 15)
display(df_train[features].isna().sum() / len(df_train))

print('\n')

print('=' * 15 + ' Missing Values in Test Data ' + '=' * 15)
display(df_test[features].isna().sum())

print('=' * 15 + ' Proportion of Values Missing in Test Data ' + '=' * 15)
display(df_test[features].isna().sum() / len(df_test))


# Testing for Statistical Significance in Different Proportions of Missing Values
ttest_results = {}

for col in features:
    missing_train = df_train[col].isna().astype(int)
    missing_test = df_test[col].isna().astype(int)
    
    stat, p = ttest_ind(missing_train, missing_test, equal_var=True) # Using equal_variance since earlier shown that variance between train and test are the same

    ttest_results[col] = {
        'Train Missing Rate': missing_train.mean(),
        'Test Missing Rate': missing_test.mean(),
        'p-value': p
    }

# raw results
ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index')
print('=' * 15 + ' Statistical Significance in Missing Rate Differences ' + '=' * 15)
display(ttest_df.sort_values('p-value'))

print('\n')

# With bonferroni correction for multiple statistical tests
corrected_ttest_res = multipletests(ttest_df['p-value'], alpha=0.05, method='bonferroni')
ttest_df['p-value corrected'] = corrected_ttest_res[1]
ttest_df['Significant'] = corrected_ttest_res[0]
print('=' * 15 + ' Statistical Significance in Missing Rate Differences (Corrected) ' + '=' * 15)
display(ttest_df.sort_values('p-value corrected'))


fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes = axes.flatten()

sns.heatmap(df_train[features].isna(), ax=axes[0])
axes[0].set_title('Missing Values in Train')

sns.heatmap(df_test[features].isna(), ax=axes[1])
axes[1].set_title('Missing Values in Test')

plt.tight_layout()
plt.show()


sns.histplot(df_train['Personality'])
plt.title('Distribution of Personality in Train')
plt.show()


# Histograms
fig, axes = plt.subplots(5, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(len(num_features) * 2):
    feature = num_features[i // 2]
    if i % 2 == 0:
        sns.histplot(df_train[feature], bins=df_train[feature].nunique(), ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} in Train')
    else:
        sns.histplot(df_test[feature], bins=df_train[feature].nunique(), ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} in Test')

plt.tight_layout()
plt.show()


# Scatterplots
sns.pairplot(data=df_train[num_features + ['Personality']], corner=True, kind='scatter', hue='Personality')
plt.show()


# Contours
sns.pairplot(data=df_train[num_features + ['Personality']], corner=True, kind='kde', hue='Personality')
plt.show()


# Correlation Matrix
corr_matrix = df_train[num_features].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.xticks(rotation=45)
plt.title('Correlation of Features in df_train')
plt.show()


# variance inflation factors
X = df_train[num_features].copy()
X = X.dropna()
X_vif = add_constant(X)

vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
})

vif_data = vif_data[vif_data['Feature'] != 'const']

print('=' * 15 + ' Variance Inflation Factors ' + '=' * 15)
display(vif_data.sort_values(by='VIF', ascending=False).reset_index(drop=True))


fig, axes = plt.subplots(5, 1, figsize=(8, 16))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    sns.boxplot(x=df_train['Personality'], y=df_train[feature], ax=axes[i])
    axes[i].set_title(f'Relationship Between {feature} and Personality')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i in range(len(cat_features) * 2):
    feature = cat_features[i // 2]
    if i % 2 == 0:
        sns.histplot(df_train[feature], bins=df_train[feature].nunique(), ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} in Train')
    else:
        sns.histplot(df_test[feature], bins=df_train[feature].nunique(), ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} in Test')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes = axes.flatten()

# Relationship between Stage_fear and Personality
counts = df_train.groupby(['Stage_fear', 'Personality']).size().rename('Count')
total = counts.groupby(level=0).transform('sum')
proportions = (counts / total).reset_index(name='Proportion')

sns.barplot(data=proportions, x='Personality', y='Proportion', hue='Stage_fear', ax=axes[0])
axes[0].set_title('Relationship Between Personality and Stage_fear')
axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., prop={'size': 8})

# Relationship between Drained_after_socializing and Personality
counts = df_train.groupby(['Drained_after_socializing', 'Personality']).size().rename('Count')
total = counts.groupby(level=0).transform('sum')
proportions = (counts / total).reset_index(name='Proportion')

sns.barplot(data=proportions, x='Personality', y='Proportion', hue='Drained_after_socializing', ax=axes[1])
axes[1].set_title('Relationship Between Personality and Drained_after_socializing')
axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., prop={'size': 8})

plt.tight_layout()
plt.show()


# Relationship Between All Numeric Features and Stage_fear
fig, axes = plt.subplots(5, 2, figsize=(16, 16))
axes = axes.flatten()

for i in range(len(num_features) * 2):
    feature = num_features[i // 2]

    if i % 2 == 0:
        sns.boxplot(data=df_train, x='Stage_fear', y=feature, ax=axes[i])
        axes[i].set_title(f'Relation Between Stage_fear and {feature} in Train')
    else:
        sns.boxplot(data=df_test, x='Stage_fear', y=feature, ax=axes[i])
        axes[i].set_title(f'Relation Between Stage_fear and {feature} in Test')
    
plt.tight_layout()
plt.show()


# Relationship Between All Numeric Features and Drained_after_socializing
fig, axes = plt.subplots(5, 2, figsize=(16, 16))
axes = axes.flatten()

for i in range(len(num_features) * 2):
    feature = num_features[i // 2]

    if i % 2 == 0:
        sns.boxplot(data=df_train, x='Drained_after_socializing', y=feature, ax=axes[i])
        axes[i].set_title(f'Relation Between Drained_after_socializing and {feature} in Train')
    else:
        sns.boxplot(data=df_test, x='Drained_after_socializing', y=feature, ax=axes[i])
        axes[i].set_title(f'Relation Between Drained_after_socializingr and {feature} in Test')
    
plt.tight_layout()
plt.show()


# Mutual information
all_columns = df_train.drop(columns=['id']).columns
mi_matrix = pd.DataFrame(index=all_columns, columns=all_columns, dtype=float)
df_temp = df_train.dropna()

for i in all_columns:
    for j in all_columns:
        mi_matrix.loc[i, j] = mutual_info_score(df_temp[i], df_temp[j])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(mi_matrix.astype(float), annot=True, fmt=".2f", cmap="viridis")
plt.title("Mutual Information Matrix")
plt.show()


df_missing_train = df_train.copy()
df_missing_train = df_missing_train[df_missing_train.isnull().any(axis=1)]
df_missing_train.info()


df_missing_train.head()


df_missing_train.describe()


# Distributions of Features
fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(df_train[feature], ax=axes[i], bins=df_train[feature].nunique())
    axes[i].set_title(f'Distribution of {feature} in Missing Dataframe')

plt.tight_layout()
axes[7].remove()
plt.show()


# Making Data Compatible with XGBoost
outcome_mapping = {
    'Introvert': 0,
    'Extrovert': 1
}

reverse_mapping = {v: k for k, v in outcome_mapping.items()}

def clean_data(df, test=False):
    """
    Function for Cleaning Data
    Takes in dataframe and applies data cleaning:
    - Encodes 'objects' as 'category'
    - Drops the id column
    - if test is equal to False, then also encode the outcome numerically
    """
    df_temp = df.copy()
    df_temp[cat_features] = df_temp[cat_features].astype('category')
    df_temp.drop(columns=['id'], inplace=True)
    if not test:
        df_temp['Personality'] = df_temp['Personality'].map(outcome_mapping)

    return df_temp


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

SEED = 30


df_train_clean = clean_data(df_train)

X = df_train_clean[features]
y = df_train_clean['Personality']


# Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
scores = []

for i, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        enable_categorical=True
    )
    model.fit(X_train_fold, y_train_fold)
    fold_val_pred = model.predict(X_val_fold)

    score = accuracy_score(y_val_fold, fold_val_pred)
    print(f'========== Fold {i} accuracy Score: {score} ==========')
    scores.append(score)
print(f'Average Score: {np.mean(scores)}')


df_test_clean = clean_data(df_test, test=True)


model = XGBClassifier(
    enable_categorical=True
)

model.fit(X, y)
y_test_pred = model.predict(df_test_clean)


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = y_test_pred
submission['Personality'] = submission['Personality'].map(reverse_mapping)
submission.to_csv('submission.csv', index=False)
submission.head()

