import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# top rows
train_df.head()


# iformation
train_df.info()


# shape
print(f'Rows: {train_df.shape[0]}\nCols: {train_df.shape[1]}')


# statistical information
train_df.describe()


# null value
train_df.isnull().sum()


# duplicate
train_df.duplicated().sum()


# univariate analysis | frequency distribution | numeric data

# numeric columns only
ncols = train_df.select_dtypes(include='number').columns.to_list()


plt.figure(figsize=(10, 8))
for i, col in enumerate(ncols):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=train_df, x=col, kde=True)
    plt.title(f'{col} distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# skewness
train_df[ncols].skew()


# univariate analysis | outliers | numeric data

plt.figure(figsize=(10, 8))
for i, col in enumerate(ncols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=train_df, x=col)
    plt.title(f'{col}')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# univariate analysis | frequency distribution | categorical feature

# object columns only
ocols = train_df.select_dtypes(include='object').columns.to_list()


plt.figure(figsize=(10, 8))
for i, col in enumerate(ocols):
    plt.subplot(2, 2, i+1)
    sns.countplot(data=train_df, x=col, hue=col, palette='rocket')
    plt.title(f'{col} distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# bivariate analysis | num vs cat | num vs num

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
sns.barplot(data=train_df, x='Personality', y='Time_spent_Alone', errorbar=None)

plt.subplot(2, 2, 2)
sns.barplot(data=train_df, x='Drained_after_socializing', y='Social_event_attendance', hue='Personality', errorbar=None)

plt.subplot(2, 2, 3)
sns.scatterplot(data=train_df, x='Going_outside', y='Friends_circle_size', hue='Personality')

plt.subplot(2, 2, 4)
sns.barplot(data=train_df, x='Personality', y='Post_frequency', errorbar=None)

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# multivariate analysis

sns.pairplot(train_df, hue='Personality')
plt.show()


# correlation

correlation = train_df.corr(numeric_only=True)

sns.heatmap(correlation, annot=True, fmt='0.2f', cmap='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.show()


# target variable analysis

sns.countplot(data=train_df, x='Personality')
plt.show()


train_df['Personality'].value_counts()


# drop ids

train_df = train_df.drop('id', axis=1)

tid = test_df['id']
test_df = test_df.drop('id', axis=1)


# analyzing null data

null_rows = train_df.isnull().any(axis=1).sum()
print('null rows in Train:', null_rows)

null_rows = test_df.isnull().any(axis=1).sum()
print('null rows in Test:', null_rows)


# encoding

le = LabelEncoder()
train_df['Stage_fear'] = le.fit_transform(train_df['Stage_fear'])
train_df['Drained_after_socializing'] = le.fit_transform(train_df['Drained_after_socializing'])
train_df['Personality'] = train_df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

test_df['Stage_fear'] = le.transform(test_df['Stage_fear'])
test_df['Drained_after_socializing'] = le.transform(test_df['Drained_after_socializing'])


# imputing null values

iter_imputer = IterativeImputer()

train_df = pd.DataFrame(iter_imputer.fit_transform(train_df), columns=train_df.columns)
test_df = pd.DataFrame(iter_imputer.fit_transform(test_df), columns=test_df.columns)


# reduce skewness - normalization | sqrt

train_df['Time_spent_Alone'] = np.sqrt(train_df['Time_spent_Alone'])
train_df.skew(numeric_only=True)


# split the data

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# personality types proportion
percentage = train_df['Personality'].value_counts(normalize=True) * 100
print(percentage.map('{:.2f}%'.format))


# model initialization

models = {
    'logistic': LogisticRegression(class_weight='balanced'),
    'Forest': RandomForestClassifier(class_weight='balanced'),
    'XGBoost': XGBClassifier(scale_pos_weight=13699/4825),
}


# cross validation

cv = RepeatedStratifiedKFold(n_splits=20, n_repeats=3, random_state=42)
scoring = ['accuracy', 'roc_auc', 'f1']

best_model = None
best_score = 0
best_metric = 'accuracy' 

for name, model in models.items():
    score = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
    avg_scores = {m: score[f'test_{m}'].mean() for m in scoring}
    print(f"{name} : " + " | ".join([f"{m}: {avg_scores[m]:.5f}" for m in scoring]))
    
    if avg_scores[best_metric] > best_score:
        best_score = avg_scores[best_metric]
        best_model = name

print(f"\nBest model based on {best_metric}: {best_model} ({best_metric} = {best_score:.5f})")


# best model | training

final_model = RandomForestClassifier()
final_model.fit(X, y)


# prediction
y_pred = final_model.predict(test_df)
y_pred


# submission

submission = pd.DataFrame({
    'id': tid.values,
    'Personality': pd.Series(y_pred).map({0: 'Extrovert', 1: 'Introvert'})
})

submission.head()


# save submission to csv

submission.to_csv('submission.csv', index=False)

