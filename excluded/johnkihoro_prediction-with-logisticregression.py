import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBRFClassifier
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
print(df.shape)
df.head()


df.duplicated().sum()


df.info()


df.describe()


df.describe(include=object)


y = df['loan_paid_back']
X = df.drop('loan_paid_back', axis=1)


numerical_columns = [col for col in X.columns if X[col].dtype!=object]
categorical_columns = [col for col in X.columns if X[col].dtype==object]
print(f'list of numerical_columns: {numerical_columns}')
print(f'list of categorical_columns: {categorical_columns}')


plt.figure(figsize=(15, 11))
for i, col in enumerate(numerical_columns):
    ax = plt.subplot(3,2,i+1, alpha=.5, )
    sns.violinplot(X[col], color='lightblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()


plt.figure(figsize=(15, 11))
for i, col in enumerate(numerical_columns):
    ax = plt.subplot(3,2,i+1, alpha=.5, )
    sns.histplot(X[col], kde=True, color='forestgreen')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.axvline(X[col].mean(), linestyle='-.', color='red', label='Mean')
    plt.axvline(X[col].median(), linestyle='--', color='black', label='Median')
    plt.legend()
    plt.tight_layout()


for col in numerical_columns:
    print(col.ljust(40, '.'), X[col].skew())


X['log_annual_income'] = np.log(X['annual_income'])
X['log_debt_to_income_ratio'] = np.log(X['debt_to_income_ratio'])


plt.figure(figsize=(15, 11))
for i, col in enumerate(X.iloc[:, 11:].columns):
    ax = plt.subplot(3,2,i+1, alpha=.5, )
    sns.histplot(X[col], kde=True, color='lightgreen')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.axvline(X[col].mean(), linestyle='--', color='red', label='Mean')
    plt.axvline(X[col].median(), linestyle='--', color='black', label='Median')
    plt.legend()
    plt.tight_layout()


plt.figure(figsize=(14, 11))
for i, col in enumerate(categorical_columns):
    if col!='grade_subgrade':
        ax = plt.subplot(3,3,i+1)
        sns.countplot(x=X[col], hue=X[col])
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, fontsize=9)
        plt.title(f'Count plot for {col}')
        plt.tight_layout()


X['grade_subgrade'].value_counts(normalize=True)


plt.figure(figsize=(14, 11))
sns.pairplot(df, hue='loan_paid_back');


plt.figure(figsize=(15, 11))
for i, col in enumerate(X.iloc[:, 11:].columns):
    ax = plt.subplot(3,3, i+1)
    sns.histplot(X[col], kde=True, color='skyblue', ax=ax)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.axvline(X[col].mean(), linestyle='-.', color='red', label='Mean')
    plt.axvline(X[col].median(), linestyle='--', color='black', label='Median')
    plt.legend()
    plt.tight_layout()


[numerical_columns.append(col) for col in X.iloc[:, 11:].columns]
numerical_columns


numerical_columns.remove('annual_income')
numerical_columns.remove('debt_to_income_ratio')


plt.figure(figsize=(15, 11))
for i, col in enumerate(numerical_columns):
    ax = plt.subplot(4,3, i+1)
    sns.boxplot(x=X[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    #plt.ylabel('Frequency')
    plt.axvline(X[col].mean(), linestyle='-.', color='red', label='Mean')
    plt.axvline(X[col].median(), linestyle='--', color='black', label='Median')
    plt.legend()
    plt.tight_layout()


plt.figure(figsize=(17, 13))
for i, col in enumerate(numerical_columns):
    ax = plt.subplot(4,3, i+1)
    q1 = X[col].quantile(.25)
    q3 = X[col].quantile(.75)
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    X[col] = X[col].clip(lower_bound, upper_bound)
    sns.boxplot(x=X[col])


for col in categorical_columns:
    print(col.ljust(30, '.'), X[col].unique())


ordinal_columns = ['education_level', 'grade_subgrade']
nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
print(nominal_columns)


numeric_scaler = preprocessing.StandardScaler()
ordinal_transformer = preprocessing.OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
nominal_transformer = preprocessing.OneHotEncoder(drop='first', sparse_output=False, dtype=np.float32)

column_transformers = ColumnTransformer([
    ('scaler', numeric_scaler, numerical_columns),
    ('ordinal_columns', ordinal_transformer, ordinal_columns),
    ('nominal_scaler', nominal_transformer, nominal_columns)
], remainder='drop').set_output(transform='pandas')


X


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.1, random_state=87, stratify=y)
X_valid_processed = column_transformers.fit_transform(X_valid)
X_valid_processed


lr = LogisticRegression(C=.001, max_iter=10000, random_state=56, verbose=0)
pipe = Pipeline([
    ('columns_transformers', column_transformers),
    ('model', lr)
])
pipe.fit(X_train, y_train)


params = {
    'model__C': [.001, .01, 1], 
    'model__class_weight':[{1:v, 0:1} for v in np.arange(1, 7)],
    'model__solver':['saga', 'lbfgs', 'newton-cg'],
    'model__penalty':['l1', 'l2', None]
}
grid = GridSearchCV(estimator=pipe, param_grid=params, cv=5, scoring='roc_auc')
#grid.fit(X_train, y_train)


#grid.best_params_


y_preds = pipe.predict(X_valid)
print('Validation roc_auc is: ', roc_auc_score(y_valid, y_preds))
print(classification_report(y_valid, y_preds))


y_preds = pipe.predict(X_valid)
y_preds


cm = confusion_matrix(y_valid, y_preds)
cm


ConfusionMatrixDisplay(cm).plot()


test_data = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
test_data.head()


test_data.info()


test_data['log_annual_income'] = np.log(test_data['annual_income'])
test_data['log_debt_to_income_ratio'] = np.log(test_data['debt_to_income_ratio'])


plt.figure(figsize=(15, 11))
for i, col in enumerate(numerical_columns):
    ax = plt.subplot(4,3,i+1, alpha=.5, )
    sns.histplot(test_data[col], kde=True, color='forestgreen')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.axvline(test_data[col].mean(), linestyle='-.', color='red', label='Mean')
    plt.axvline(test_data[col].median(), linestyle='--', color='black', label='Median')
    plt.legend()
    plt.tight_layout()


test_data


test_preds = pipe.predict_proba(test_data)[:, 1]
test_preds


sub_df = pd.DataFrame({'id':test_data.index,
                      'loan_paid_back':test_preds})
sub_df


sub_df.to_csv('submission file.csv', index=False)




