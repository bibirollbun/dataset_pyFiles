import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, chi2


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df.head()


df.info()


df.describe()


df.isna().sum()


df.duplicated().sum()


categorical_attribs = df.select_dtypes(include='object').columns

print('Unique value for each categorical attribs')
for attrib in categorical_attribs:
    print(attrib, ':' , df[attrib].unique())


numerical_attribs = df.select_dtypes(include=np.number).columns

n = len(numerical_attribs)
ncols = 2
nrows = (n + 1) // 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
axes = axes.flatten()

for i, attrib in enumerate(numerical_attribs):
    sns.boxplot(
        x='diagnosed_diabetes',
        y=attrib,
        data=df,
        ax=axes[i]
    )
    axes[i].set_title(f'Diagnosed diabetes by {attrib}')
    axes[i].set_xlabel('Diagnosed diabetes')
    axes[i].set_ylabel(attrib)
    axes[i].set_xticks([0, 1], ['No Diabetes', 'Diabetes'])

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='gender', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by gender')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='ethnicity', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='education_level', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by education level')
plt.xlabel('Education level')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='income_level', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by income level')
plt.xlabel('Income level')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='smoking_status', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by smoking status')
plt.xlabel('Smoking status')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='employment_status', hue='diagnosed_diabetes', palette='viridis')
plt.title('Diagnosed diabetes by employment status')
plt.xlabel('Employment status')
plt.ylabel('Count')

plt.legend(title='Diagnosed diabetes', labels=['No diabetes', 'Diabetes'])
plt.show()


df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

X_train = df_train.drop(columns=['diagnosed_diabetes'])
y_train = df_train['diagnosed_diabetes']

X_test = df_test.drop(columns=['diagnosed_diabetes'])
y_test = df_test['diagnosed_diabetes']

print('Train dataset size:', df_train.shape)
print('Test dataset size:', df_test.shape)


numerical_attributes = df.drop(columns=['id', 'diagnosed_diabetes']).select_dtypes(include=np.number).columns.to_list()
ordinal_attributes = ['education_level', 'income_level']
one_hot_attributes = ['gender', 'ethnicity', 'smoking_status', 'employment_status']

print('Numerical attributes:', numerical_attributes)
print('Ordinal attributes:', ordinal_attributes)
print('One hot attributes:', one_hot_attributes)


numerical_attribute_pipeline = Pipeline(
    steps=[
        ('standard_scaler', StandardScaler()),
    ]
)

ordinal_attribute_pipeline = Pipeline(
    steps=[
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan))
    ]
)

one_hot_attribute_pipeline = Pipeline(
    steps=[
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    [
        ('numerical_attribute_pipeline', numerical_attribute_pipeline, numerical_attributes),
        ('ordinal_attribute_pipeline', ordinal_attribute_pipeline, ordinal_attributes),
        ('one_hot_attribute_pipeline', one_hot_attribute_pipeline, one_hot_attributes),
    ]
)


preprocessor.fit(X_train, y_train)

X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


logistic_regression = LogisticRegression(solver="lbfgs", C=10, max_iter=1000)
logistic_regression.fit(X_train_preprocessed, y_train)


y_train_prediction = logistic_regression.predict(X_train_preprocessed)
y_test_prediction = logistic_regression.predict(X_test_preprocessed)

y_train_prediction_proba = logistic_regression.predict_proba(X_train_preprocessed)[:, 1]
y_test_prediction_proba = logistic_regression.predict_proba(X_test_preprocessed)[:, 1]


train_set_accuracy_score = accuracy_score(y_train, y_train_prediction)
test_set_accuracy_score = accuracy_score(y_test, y_test_prediction)

print(f'Accuracy score for training set: {train_set_accuracy_score:.4f}', )
print(f'Accuracy score for test set: {test_set_accuracy_score:.4f}')


train_set_roc_auc_score = roc_auc_score(y_train, y_train_prediction_proba)
test_set_roc_auc_score = roc_auc_score(y_test, y_test_prediction_proba)

print(f'Roc auc score for training set: {train_set_roc_auc_score:.4f}', )
print(f'Roc auc score for test set: {test_set_roc_auc_score:.4f}')


df_final_result = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
X_final_result_preprocessed = preprocessor.transform(df_final_result)

predictions = logistic_regression.predict_proba(X_final_result_preprocessed)[:, 1]


pd.DataFrame({'id': df_final_result['id'], 'diagnosed_diabetes': np.round(predictions, 1)}).to_csv('submission.csv', index=False)
df_submission = pd.read_csv('submission.csv')
df_submission.head()

