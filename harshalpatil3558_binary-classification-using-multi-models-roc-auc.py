import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


test.head()


print(train.shape,test.shape)


train.info()


test.info()


train.describe(include='all')


train.describe(include='all')


train.duplicated().sum()


test.duplicated().sum()


train.isnull().sum()


test.isnull().sum()


(train == 0).sum()


(test == 0).sum()


train = train.drop(columns=['id','poutcome'],axis=1)
test = test.drop(columns=['id','poutcome'],axis=1)


categorical_columns = train.columns[train.dtypes == 'O']
print(categorical_columns)


numerical_columns = train.columns[train.dtypes != 'O']
print(numerical_columns)


import matplotlib.pyplot as plt
import seaborn as sns

for col in categorical_columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data=train, x=col, discrete=True)
    plt.title(f'{col} Distribution in training set')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=55)
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

for col in categorical_columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data=train, x=col, discrete=True, hue=train['y'])
    plt.title(f'{col} Distribution in training set')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=55)
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_columns:
    plt.figure(figsize=(8,6))
    sns.distplot(train[col])
    plt.title(f'{col} Distribution in training set')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=55)
    plt.show()


train['y'].value_counts()


plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.show()


train[numerical_columns].hist(figsize=(20,15), bins=30, edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


# Columns with 'unknown values'
print(train['contact'].value_counts(),"\n")
print(train['education'].value_counts(),"\n")
print(train['job'].value_counts())


## Contact feature
# Find the mode (most frequent value) of the column
mode_val_train = train['contact'].mode()[0]
mode_val_test = test['contact'].mode()[0]


# Replace 'unknown' with the mode value
train['contact'] = train['contact'].replace('unknown', mode_val_train)
test['contact'] = test['contact'].replace('unknown', mode_val_test)


## Education feature
# Find the mode (most frequent value) of the column
mode_val_train = train['education'].mode()[0]
mode_val_test = test['education'].mode()[0]


# Replace 'unknown' with the mode value
train['education'] = train['education'].replace('unknown', mode_val_train)
test['education'] = test['education'].replace('unknown', mode_val_test)


## Job feature
# Find the mode (most frequent value) of the column
mode_val_train = train['job'].mode()[0]
mode_val_test = test['job'].mode()[0]


# Replace 'unknown' with the mode value
train['job'] = train['job'].replace('unknown', mode_val_train)
test['job'] = test['job'].replace('unknown', mode_val_test)


print('Categorical Columns value counts')
for col in categorical_columns:
    print(train[col].value_counts())
    print()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder

order_encode = OrdinalEncoder()

# Fit and transform the 'job' column (note the double square brackets to keep it 2D)
train['job'] = order_encode.fit_transform(train[['job']])
test['job'] = order_encode.fit_transform(test[['job']])

# train['job'] = train['job'].replace('job':{'management','blue-collar','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student'})
train['marital'] = train['marital'].replace({'single': 0, 'married': 1, 'divorced': -1})
test['marital'] = test['marital'].replace({'single': 0, 'married': 1, 'divorced': -1})

train['month'] = train['month'].replace({
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
})
test['month'] = test['month'].replace({
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
})

train['education'] = train['education'].replace({'primary': 1, 'secondary': 2, 'tertiary': 3})
test['education'] = test['education'].replace({'primary': 1, 'secondary': 2, 'tertiary': 3})

train['default'] = train['default'].replace({'no':0,'yes':1})
test['default'] = test['default'].replace({'no':0,'yes':1})

train['housing'] = train['housing'].replace({'no':0,'yes':1})
test['housing'] = test['housing'].replace({'no':0,'yes':1})

train['loan'] = train['loan'].replace({'no':0,'yes':1})
test['loan'] = test['loan'].replace({'no':0,'yes':1})

train['contact'] = train['contact'].replace({'cellular':1,'telephone':0})
test['contact'] = test['contact'].replace({'cellular':1,'telephone':0})


X = train.drop(columns=['y'])
y = train['y']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# test data
test_scaled = scaler.transform(test)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


regressor = LogisticRegression(max_iter=1000, random_state=42)  # Recommended to increase max_iter and set random_state

# Fit the model
regressor.fit(X_train_scaled, y_train)

# Predict on training data
y_train_pred = regressor.predict(X_train_scaled)
# Calculate training accuracy (true labels first)
y_train_acc = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {y_train_acc:.4f}')

# Predict on test data
y_test_pred = regressor.predict(X_test_scaled)
# Calculate test accuracy
y_test_acc = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {y_test_acc:.4f}')

# Optional: Full classification report on test set
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# Compute predicted probabilities for positive class
y_test_prob = regressor.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc:.4f}')

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


rf = RandomForestClassifier(random_state=42)

# Fit the model
rf.fit(X_train_scaled, y_train)

# Predict on training data
y_train_pred = rf.predict(X_train_scaled)
# Calculate training accuracy (true labels first)
y_train_acc = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {y_train_acc:.4f}')

# Predict on test data
y_test_pred = rf.predict(X_test_scaled)
# Calculate test accuracy
y_test_acc = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {y_test_acc:.4f}')

# Optional: Full classification report on test set
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


# Compute predicted probabilities for positive class
y_test_prob = rf.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc:.4f}')

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dtc = DecisionTreeClassifier(random_state=42)

# Fit the model
dtc.fit(X_train_scaled, y_train)

# Predict on training data
y_train_pred = dtc.predict(X_train_scaled)
# Calculate training accuracy (true labels first)
y_train_acc = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {y_train_acc:.4f}')

# Predict on test data
y_test_pred = dtc.predict(X_test_scaled)
# Calculate test accuracy
y_test_acc = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {y_test_acc:.4f}')

# Optional: Full classification report on test set
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


# Compute predicted probabilities for positive class
y_test_prob = dtc.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc:.4f}')

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# We will test the test data with Logistic Regression

test_pred = regressor.predict(test_scaled)



test_pred





!pip install imbalanced-learn


# # =========== LIBRARIES & READING ===========
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings

# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
# from imblearn.over_sampling import SMOTE  # For class balancing

# warnings.simplefilter('ignore')

# # Load datasets
# train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# train = train.drop(columns=['id', 'poutcome'], axis=1)
# test = test.drop(columns=['id', 'poutcome'], axis=1)

# categorical_cols = train.columns[train.dtypes == 'O']
# numerical_cols = train.columns[train.dtypes != 'O']

# # =========== HANDLE MISSING CATEGORICAL ('unknown') ===========
# def fill_unknown_with_mode(df, column):
#     mode_val = df[column].mode()[0]
#     df[column] = df[column].replace('unknown', mode_val)
#     return df

# for col in ['contact', 'education', 'job']:
#     train = fill_unknown_with_mode(train, col)
#     test = fill_unknown_with_mode(test, col)

# # =========== CATEGORICAL ENCODING ===========
# job_encoder = OrdinalEncoder()
# train['job'] = job_encoder.fit_transform(train[['job']])
# test['job'] = job_encoder.transform(test[['job']])  # use transform only!

# marital_map = {'single': 0, 'married': 1, 'divorced': -1}
# month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
#              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
# education_map = {'primary': 1, 'secondary': 2, 'tertiary': 3}
# default_map = {'no': 0, 'yes': 1}
# housing_map = {'no': 0, 'yes': 1}
# loan_map = {'no': 0, 'yes': 1}
# contact_map = {'cellular': 1, 'telephone': 0}

# for col, mapping in zip(
#     ['marital', 'month', 'education', 'default', 'housing', 'loan', 'contact'],
#     [marital_map, month_map, education_map, default_map, housing_map, loan_map, contact_map]
# ):
#     train[col] = train[col].replace(mapping)
#     test[col] = test[col].replace(mapping)

# # =========== SPLIT FEATURES/TARGET ===========
# X = train.drop(columns=['y'])
# y = train['y']

# # =========== TRAIN/VALIDATION SPLIT (Stratified) ===========
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # =========== SCALING ===========
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# test_scaled = scaler.transform(test)

# # =========== HANDLE IMBALANCE (SMOTE) ===========
# smote = SMOTE(random_state=42, sampling_strategy='auto')   # Default: upsample minority
# X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
# print("After balancing, class distribution:", np.bincount(y_train_bal))

# # =========== CROSS-VALIDATION & HYPERPARAMETER TUNING (LOGISTIC REG) ===========
# lr_params = {
#     'C': [0.01, 0.1, 1, 10], 
#     'solver': ['lbfgs'], 
#     'class_weight': [None, 'balanced']
# }
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# lr_grid = GridSearchCV(
#     LogisticRegression(max_iter=1000, random_state=42),
#     lr_params,
#     cv=skf,
#     scoring='roc_auc',
#     n_jobs=-1
# )
# lr_grid.fit(X_train_bal, y_train_bal)
# print("Best Logistic Regression parameters:", lr_grid.best_params_)

# # =========== FINAL MODEL EVALUATION ===========
# def evaluate_model(model, X_tr, y_tr, X_val, y_val, name='Model'):
#     # Training
#     model.fit(X_tr, y_tr)
#     # Predictions
#     val_pred = model.predict(X_val)
#     val_prob = model.predict_proba(X_val)[:,1]
#     train_pred = model.predict(X_tr)
#     print(f'\n{name} Training Accuracy: {accuracy_score(y_tr, train_pred):.4f}')
#     print(f'{name} Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}')
#     print(f'{name} ROC AUC (Validation): {roc_auc_score(y_val, val_prob):.4f}')
#     print(f"{name} Classification Report (Validation):\n",classification_report(y_val, val_pred))
#     # ROC Curve Plot
#     fpr, tpr, _ = roc_curve(y_val, val_prob)
#     plt.figure(figsize=(8,6))
#     plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc_score(y_val, val_prob):.4f})')
#     plt.plot([0,1],[0,1],'--', c='grey')
#     plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{name} ROC Curve')
#     plt.legend(); plt.show()

# # Use the best logistic regression
# evaluate_model(lr_grid.best_estimator_, X_train_bal, y_train_bal, X_val_scaled, y_val, 'Logistic Regression (Tuned + Balanced)')

# # =========== OPTIONAL: RANDOM FOREST GRIDSEARCH ===========
# rf_params = {'n_estimators': [100, 200], 'max_depth': [8, 15, 30], 'class_weight': [None, 'balanced']}
# rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=skf, scoring='roc_auc', n_jobs=-1)
# rf_grid.fit(X_train_bal, y_train_bal)
# print("Best RF params:", rf_grid.best_params_)
# evaluate_model(rf_grid.best_estimator_, X_train_bal, y_train_bal, X_val_scaled, y_val, 'Random Forest (Tuned + Balanced)')

# # =========== FEATURE IMPORTANCE (Tree models) ===========
# feat_imps = rf_grid.best_estimator_.feature_importances_
# feat_names = X.columns
# imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': feat_imps}).sort_values(by='Importance', ascending=False)
# plt.figure(figsize=(10,5))
# sns.barplot(x='Importance', y='Feature', data=imp_df)
# plt.title('Feature Importances (Random Forest)')
# plt.show()

# # =========== TEST SET PREDICTIONS (Submission: probabilistic) ===========
# test_pred_probs = lr_grid.best_estimator_.predict_proba(test_scaled)[:, 1]
# # For competition submission:
# pd.DataFrame({'y': test_pred_probs}).to_csv('submission.csv', index=False)

# # If you had an id column, add it back for submission:
# # pd.DataFrame({'id': test_ids, 'y': test_pred_probs}).to_csv('submission.csv', index=False)



!pip install scikit-learn==1.2.2



# =========== LIBRARIES & READING ===========
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score

warnings.simplefilter('ignore')

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

train = train.drop(columns=['id', 'poutcome'], axis=1)
test = test.drop(columns=['id', 'poutcome'], axis=1)

categorical_cols = train.columns[train.dtypes == 'O']
numerical_cols = train.columns[train.dtypes != 'O']

# =========== HANDLE MISSING CATEGORICAL ('unknown') ===========
def fill_unknown_with_mode(df, column):
    mode_val = df[column].mode()[0]
    df[column] = df[column].replace('unknown', mode_val)
    return df

for col in ['contact', 'education', 'job']:
    train = fill_unknown_with_mode(train, col)
    test = fill_unknown_with_mode(test, col)

# =========== CATEGORICAL ENCODING ===========
job_encoder = OrdinalEncoder()
train['job'] = job_encoder.fit_transform(train[['job']])
test['job'] = job_encoder.transform(test[['job']])  # use transform only!

marital_map = {'single': 0, 'married': 1, 'divorced': -1}
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
education_map = {'primary': 1, 'secondary': 2, 'tertiary': 3}
default_map = {'no': 0, 'yes': 1}
housing_map = {'no': 0, 'yes': 1}
loan_map = {'no': 0, 'yes': 1}
contact_map = {'cellular': 1, 'telephone': 0}

for col, mapping in zip(
    ['marital', 'month', 'education', 'default', 'housing', 'loan', 'contact'],
    [marital_map, month_map, education_map, default_map, housing_map, loan_map, contact_map]
):
    train[col] = train[col].replace(mapping)
    test[col] = test[col].replace(mapping)

# =========== SPLIT FEATURES/TARGET ===========
X = train.drop(columns=['y'])
y = train['y']

# =========== TRAIN/VALIDATION SPLIT (Stratified) ===========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========== SCALING ===========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)

# =========== CROSS-VALIDATION & HYPERPARAMETER TUNING (LOGISTIC REG) ===========
lr_params = {
    'C': [0.01, 0.1, 1, 10], 
    'solver': ['lbfgs'], 
    'class_weight': [None, 'balanced']
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)
# Fit GridSearchCV using original training data 
lr_grid.fit(X_train_scaled, y_train)
print("Best Logistic Regression parameters:", lr_grid.best_params_)

# =========== FINAL MODEL EVALUATION ===========
def evaluate_model(model, X_tr, y_tr, X_val, y_val, name='Model'):
    # Training
    model.fit(X_tr, y_tr)
    # Predictions
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:,1]
    train_pred = model.predict(X_tr)
    print(f'\n{name} Training Accuracy: {accuracy_score(y_tr, train_pred):.4f}')
    print(f'{name} Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}')
    print(f'{name} ROC AUC (Validation): {roc_auc_score(y_val, val_prob):.4f}')
    print(f"{name} Classification Report (Validation):\n",classification_report(y_val, val_pred))
    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_val, val_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc_score(y_val, val_prob):.4f})')
    plt.plot([0,1],[0,1],'--', c='grey')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{name} ROC Curve')
    plt.legend(); plt.show()

# Use the best logistic regression
evaluate_model(lr_grid.best_estimator_, X_train_scaled, y_train, X_val_scaled, y_val, 'Logistic Regression (Tuned)')


# =========== OPTIONAL: RANDOM FOREST GRIDSEARCH ===========
rf_params = {'n_estimators': [100, 200], 'max_depth': [8, 15, 30], 'class_weight': [None, 'balanced']}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=skf, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
print("Best RF params:", rf_grid.best_params_)
evaluate_model(rf_grid.best_estimator_, X_train_scaled, y_train, X_val_scaled, y_val, 'Random Forest (Tuned)')

# =========== FEATURE IMPORTANCE (Tree models) ===========
feat_imps = rf_grid.best_estimator_.feature_importances_
feat_names = X.columns
imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': feat_imps}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=imp_df)
plt.title('Feature Importances (Random Forest)')
plt.show()

# =========== TEST SET PREDICTIONS (Submission: probabilistic) ===========
test_pred_probs = lr_grid.best_estimator_.predict_proba(test_scaled)[:, 1]
# For competition submission:
pd.DataFrame({'y': test_pred_probs}).to_csv('submission.csv', index=False)

# If you had an id column, add it back for submission:
# pd.DataFrame({'id': test_ids, 'y': test_pred_probs}).to_csv('submission.csv', index=False)





