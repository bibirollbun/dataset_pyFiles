import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


list(train.columns)


list(test.columns)


len(train)


train.head()


x = train.drop(['loan_paid_back'], axis=1)
y = train['loan_paid_back']


sns.scatterplot(x=train.index, y='annual_income', data=train)
plt.show()


sns.countplot(train, x='loan_paid_back')


sns.countplot(train, x="employment_status", hue='loan_paid_back')


sns.countplot(train, x="marital_status", hue='loan_paid_back')


sns.countplot(train, x="education_level", hue='loan_paid_back')


purpose_plot = sns.countplot(train, x="loan_purpose", hue='loan_paid_back')
purpose_plot.set_xticklabels(purpose_plot.get_xticklabels(), rotation=90)


print("Observing independent variables based on the loan_paid_back Column:")

fig, axes = plt.subplots(3,2, figsize=(24,20))

sns.boxplot(data=train, y='credit_score', x ='loan_paid_back',  ax=axes[0,0])
axes[0,0].set_title('Credit Score Distribution by Paid Back Status')

sns.boxplot(data=train, y='loan_amount', x ='loan_paid_back', ax=axes[0,1])
axes[0,1].set_title('Loan Amount Distribution by Paid Back Status')
axes[0,1].tick_params(axis='x', rotation=45)

sns.boxplot(data=train, y='debt_to_income_ratio', x ='loan_paid_back', ax=axes[1,0])
axes[1,0].set_title('Debt to Income Ratio Distribution by Paid Back Status')
axes[1,0].tick_params(axis='x', rotation=45)

sns.boxplot(data=train, y='annual_income', x ='loan_paid_back', ax=axes[1,1])
axes[1,1].set_title('Annual Income Distribution by Paid Back Status')

sns.boxplot(data=train, y='interest_rate', x= 'loan_paid_back', ax=axes[2,0])
axes[2,0].set_title('Interest Rate Distribution by Paid Back Status')

axes[2,1].axis('off')
plt.show()


plt.figure(figsize=(12, 5))
sns.histplot(data=train, x='annual_income', hue='loan_paid_back', kde=True)
plt.title('Distribution of Annual Income based on the loan_paid_back column')
plt.show()


feature = 'annual_income'  

# Plot before transformation
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
train[feature].hist(bins=30)
plt.title(f"Before Log Transform: {feature}")

# Apply log transform safely
train[feature + '_log'] = np.log1p(train[feature])   # log(1+x) handles zeros

# Plot after transformation
plt.subplot(1,2,2)
train[feature + '_log'].hist(bins=30)
plt.title(f"After Log Transform: {feature}_log")

plt.tight_layout()
plt.show()


train[feature] = np.log1p(train[feature])


numerical= ['debt_to_income_ratio',
           'credit_score',
           'annual_income',
           'interest_rate']
categorical = ['marital_status',
               'education_level',
               'employment_status',
               'loan_purpose']


from sklearn.preprocessing import OneHotEncoder


# encode multiple column and files

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoder.fit(train[categorical])


train_encoded = pd.DataFrame(encoder.transform(train[categorical]),
                             columns=encoder.get_feature_names_out(categorical))

test_encoded = pd.DataFrame(encoder.transform(test[categorical]),
                            columns=encoder.get_feature_names_out(categorical))


# Combine numerical + encoded categorical
X_train = pd.concat([train[numerical].reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
X_test  = pd.concat([test[numerical].reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

# Target variable
y_train = train['loan_paid_back']


X_test = X_test[X_train.columns]   # reorder test columns to match train


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)


np.unique(X_test).size


print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


clf = RandomForestClassifier(n_estimators=100, random_state=42)


clf.fit(X_train,y_train)


y_val_proba_rf = clf.predict_proba(X_val)[:, 1]  # probability of class 1 (loan repaid)


from sklearn.metrics import roc_curve, roc_auc_score

fpr_rf, tpr_rf, thresholds = roc_curve(y_val, y_val_proba_rf)
auc_score_rf = roc_auc_score(y_val, y_val_proba_rf)

print("ROC AUC Score:", auc_score_rf)


importance = clf.feature_importances_


names = encoder.get_feature_names_out(categorical)

all_features = list(names) + numerical

# create DataFrame
features_df = pd.DataFrame(all_features, columns=['feature_name'])



#create dataframe adding feature name and importace in the same df
features_df = pd.DataFrame({
    'feature_name': all_features,
    'importance': importance
})
features_df = features_df.sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature_name', data=features_df)
plt.title("Feature Importances for Random Forest Clf")
plt.show()


log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)


y_val_proba_log_reg = log_reg.predict_proba(X_val)[:, 1]


fpr_logr, tpr_logr, thresholds = roc_curve(y_val, y_val_proba_log_reg)
auc_score_logr = roc_auc_score(y_val, y_val_proba_log_reg)

print("ROC AUC Score:", auc_score_logr)


from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
from lightgbm import LGBMClassifier


lgbm = LGBMClassifier(random_state=42)


param_dist = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 1000]
}


random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,   
    n_iter=5,                         # number of random combos
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)


X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


random_search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],    # validation set for early stopping
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


# parameter extraction
best_params = random_search.best_params_
print(best_params)


lgbm_model = LGBMClassifier(
    **best_params,          
    random_state=42         
)


lgbm_model.fit(X_train, y_train)


y_val_proba_lgbm = lgbm_model.predict_proba(X_val)[:, 1]


fpr_lgbm, tpr_lgbm, thresholds = roc_curve(y_val, y_val_proba_lgbm)
auc_score_lgbm = roc_auc_score(y_val, y_val_proba_lgbm)

print("ROC AUC Score:", auc_score_lgbm)


plt.plot(fpr_logr, tpr_logr, label=f"Logistic Regression(AUC = {auc_score_logr:.2f})")
plt.plot(fpr_lgbm, tpr_lgbm, label=f"LGBMClassifier (AUC = {auc_score_lgbm:.2f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
test_preds = lgbm_model.predict_proba(X_test)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'loan_paid_back': test_preds_proba
})

submission.to_csv('submission.csv', index=False)
print("submission.csv is ready")




