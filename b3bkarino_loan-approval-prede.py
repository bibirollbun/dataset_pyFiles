# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


train.info()


test.info()


submission.info()


x=train.drop(columns=['id','loan_status'])
y=train['loan_status']


# Distribution of status
import matplotlib.pyplot as plt
sns.countplot(x='loan_status', data=train)
plt.show()


# Categorical variable 
X_encoded=pd.get_dummies(x)
test_encoded=pd.get_dummies(test.drop(columns=['id']))


#  Calculate correlation between loan_status and other variables
correlation_with_loan_status = train.corr(numeric_only=True)['loan_status'].sort_values(ascending=False)


#  Correlation visualization
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(correlation_with_loan_status), annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Correlation with Loan Status')
plt.show()

# Correlation Output
print(correlation_with_loan_status)


# Select top 10 variables excluding id column
top_10_features = train.columns[1:11]

# Select only numeric variables
numeric_features = train[top_10_features].select_dtypes(include=np.number)

# Visualize correlation with heatmap
correlation_matrix = numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Top 10 Features (excluding ID)')
plt.show()


# Numerical Variable Distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Age group distribution (Age)
train['age_group'] = (train['person_age'] // 10) * 10
sns.countplot(x='age_group', data=train, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Age Groups')
axes[0, 0].set_xlabel('Age Group (10-year intervals)')
axes[0, 0].set_ylabel('Count')

# Income distribution (Annual income)
income_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
income_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
train['income_category'] = pd.cut(train['person_income'], bins=income_bins, labels=income_labels, right=False)
sns.countplot(x='income_category', data=train, order=income_labels, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Person Income ')
axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Count')

# Employment length distribution (Years of service)
emp_length_bins = [-1, 0, 5, 10, 15, 20, float('inf')]
emp_length_labels = ['<1 year', '1-5 years', '6-10 years', '11-15 years', '16-20 years', '20+ years']
train['emp_length_category'] = pd.cut(train['person_emp_length'], bins=emp_length_bins, labels=emp_length_labels, right=False)
sns.countplot(x='emp_length_category', data=train, order=emp_length_labels, ax=axes[0, 2])
axes[0, 2].set_title('Distribution of Employment Length ')
axes[0, 2].set_xlabel('Employment Length')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_xticklabels(emp_length_labels, rotation=45)

# Loan amount distribution (Loan amount)
loan_amnt_bins = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
loan_amnt_labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']
train['loan_amnt_category'] = pd.cut(train['loan_amnt'], bins=loan_amnt_bins, labels=loan_amnt_labels, right=False)
sns.countplot(x='loan_amnt_category', data=train, order=loan_amnt_labels, ax=axes[1, 0])
axes[0, 3].set_title('Distribution of Loan Amount ')
axes[0, 3].set_xlabel('Loan Amount Category')
axes[0, 3].set_ylabel('Count')

# cb_person_cred_hist_length (credit retention period [years])
sns.histplot(train['cb_person_cred_hist_length'], bins=20, ax=axes[0, 3])
axes[1, 0].set_title('Distribution of cb_person_cred_hist_length')
axes[1, 0].set_xlabel('cb_person_cred_hist_length (Years)')
axes[1, 0].set_ylabel('Count')


# Loan interest rate distribution (Loan interest rate)
sns.histplot(train['loan_int_rate'], bins=20, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Loan Interest Rate')
axes[1, 1].set_xlabel('Loan Interest Rate')
axes[1, 1].set_ylabel('Count')

# Loan percent of income distribution (Loan to Income Ratio)
sns.histplot(train['loan_percent_income'], bins=20, ax=axes[1, 2])
axes[1, 2].set_title('Distribution of Loan Amount Percentage of Income')
axes[1, 2].set_xlabel('Loan Amount Percentage of Income')
axes[1, 2].set_ylabel('Count')

# Hide the last subplot (if unused)
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()


# Categorical variable distribution

import matplotlib.pyplot as plt
categorical_cols = train.select_dtypes(include='object').columns

num_plots = len(categorical_cols)
rows = (num_plots + 3) // 4
cols = min(num_plots, 4)


plt.figure(figsize=(16, rows * 4))

for i, col in enumerate(categorical_cols):
  plt.subplot(rows, cols, i + 1)
  sns.countplot(x=train[col])
  plt.title(f'Distribution of {col}')
  plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# Categorical Variable Identification

# person_home_ownership (Loan ownership status): The ratio of rent and home mortgage is the highest, and the ratio of owning a home is the lowest

# loan_intent (Loan purpose): The most common purpose is education

# loan_grade (Loan grade): The ratio of A and B is the highest

# cb_person_default_on_file (Loan repayment delinquency): Whether there is a delinquency record / Y: There is a delinquency record, N: There is no delinquency record, the ratio of no delinquency is high


test_encoded.info()


test_encoded=test_encoded.reindex(columns=X_encoded.columns,fill_value=0)


#  Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_encoded,y,test_size=0.2,stratify=y)


# SMOTE application
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Modify model definition
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1,max_depth=10,n_estimators=100),
    'Support Vector Classifier': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Xgboost Classifier': XGBClassifier(n_jobs=-1, eval_metric='auc',max_depth=10,n_estimators=100)
}

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    
    return {
        'Model': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Accuracy Score': accuracy,
        'AUC Score': auc_score
    }

# Training and evaluating models with parallel processing
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate)(
        name, model, X_train_scaled, y_train_smote, X_test_scaled, y_test
    ) for name, model in models.items()
)

results_df = pd.DataFrame(results)
print(results_df)


# Draw ROC curve
plt.figure(figsize=(10, 8))
for name, model in models.items():
    model.fit(X_train_scaled, y_train_smote)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()


def train_and_evaluate_xgb(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    return model, auc_score

xgb = XGBClassifier(n_jobs=-1, eval_metric='auc', max_depth=10, n_estimators=100)
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_xgb)(
        xgb, X_train_smote, y_train_smote, X_test_scaled, y_test
    ) for _ in range(1)
)

best_model, auc_score = results[0]
print(f"AUC-ROC scores of XGBoost models: {auc_score:.4f}")

# 제출 데이터 예측
submission_pred = best_model.predict(test_encoded)
submission_df = pd.DataFrame({'id': test['id'], 'loan_status': submission_pred})
submission_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()




