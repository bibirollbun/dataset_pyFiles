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


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay



train_path = '/kaggle/input/playground-series-s5e11/train.csv'
test_path = '/kaggle/input/playground-series-s5e11/test.csv'
sample_submission_path = '/kaggle/input/playground-series-s5e11/sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)




print("Train data shape:", train.shape)
print(train.head())
print(train.info())
print(train.isnull().sum())


print("Test data shape:", test.shape)
print(test.head())
print(test.info())
print(test.isnull().sum())





print(train.describe())


train['loan_paid_back'].value_counts(normalize=True) * 100
sns.countplot(data=train, x='loan_paid_back', palette='Set2')
plt.title('Loan Payback Distribution')


train['loan_paid_back'].value_counts()


numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('loan_paid_back')
for col in numerical_cols:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.subplot(1,2,2)
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10,4))
    sns.countplot(x=col, data=train, order=train[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Frequency of {col}')
    plt.show()
    
    # Target proportion by category (stacked bar)
    ct = pd.crosstab(train[col], train['loan_paid_back'], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Loan Paid Back proportion by {col}')
    plt.show()


# Numerical features vs target
num_cols = numerical_cols  # from previous step

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='loan_paid_back', y=col, data=train)
    plt.title(f'{col} distribution by Loan Paid Back')
    plt.show()
    
    mean_vals = train.groupby('loan_paid_back')[col].mean()
    print(f'Mean of {col} by loan_paid_back:\n{mean_vals}\n')


# Categorical features vs target
cat_cols = categorical_cols  # from previous step

for col in cat_cols:
    plt.figure(figsize=(12,5))
    ct = pd.crosstab(train[col], train['loan_paid_back'], normalize='index')
    ct.plot(kind='bar', stacked=True, figsize=(12,5))
    plt.title(f'Proportion of Loan Paid Back by {col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.show()

    # Chi-square test for independence
    contingency = pd.crosstab(train[col], train['loan_paid_back'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f'Chi-square test for {col}: p-value = {p:.4g}\n')


#  Correlation matrix heatmap
plt.figure(figsize=(12,8))
corr = train[numerical_cols.to_list() + ['loan_paid_back']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# ----------------------------------------------------

# Credit Score vs DTI (Debt-to-Income) vs Loan Repaid
# ----------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=train, 
    x='credit_score', 
    y='debt_to_income_ratio', 
    hue='loan_paid_back', 
    alpha=0.6
)
plt.title("Credit Score vs Debt-to-Income Ratio by Loan Repayment")
plt.xlabel("Credit Score")
plt.ylabel("Debt-to-Income Ratio")
plt.legend(title="Loan Paid Back", loc="upper right")
plt.show()


# ----------------------------------------------------

# Interest Rate across Grade_Subgrade categories
# ----------------------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(
    data=train,
    x='grade_subgrade',
    y='interest_rate',
    hue='loan_paid_back'
)
plt.title("Interest Rate Distribution across Grade_Subgrade and Repayment Status")
plt.xlabel("Grade_Subgrade")
plt.ylabel("Interest Rate")
plt.legend(title="Loan Paid Back")
plt.show()



# ----------------------------------------------------

#  Employment Status vs Loan Purpose
# ----------------------------------------------------
cross_tab = pd.crosstab(train['employment_status'], train['loan_purpose'], 
                        values=train['loan_paid_back'], aggfunc='mean').fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(cross_tab, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Average Repayment Rate: Employment Status vs Loan Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Employment Status")
plt.show()


# ----------------------------------------------------
# Loan Amount to Income Ratio and Repayment
# ----------------------------------------------------
train['loan_to_income_ratio'] = train['loan_amount'] / (train['annual_income'] + 1)

plt.figure(figsize=(8,6))
sns.boxplot(
    data=train, 
    x='marital_status', 
    y='loan_to_income_ratio', 
    hue='loan_paid_back'
)
plt.title("Loan-to-Income Ratio vs Marital Status and Repayment")
plt.xlabel("Marital Status")
plt.ylabel("Loan-to-Income Ratio")
plt.legend(title="Loan Paid Back")
plt.show()


target = 'loan_paid_back'
id_col = 'id'

num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


train = train.drop(columns=[id_col], errors='ignore')


X = train.drop(columns=[target])
y = train[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# Label encode 'grade_subgrade
label_enc = LabelEncoder()
X_train['grade_subgrade'] = label_enc.fit_transform(X_train['grade_subgrade'])
X_test['grade_subgrade'] = label_enc.transform(X_test['grade_subgrade'])

# One-Hot Encode other categorical features
cat_to_encode = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_to_encode)
], remainder='passthrough')


logreg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_log = logreg_pipeline.predict(X_test)
y_proba_log = logreg_pipeline.predict_proba(X_test)[:, 1]

# ----------------------------------------------------
# Evaluate Logistic Regression
# ----------------------------------------------------
print("ğŸ”¹ Logistic Regression Results")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))


rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nğŸ”¹ Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))



from xgboost import XGBClassifier
xgb_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)
y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]


# Evaluate XGBoost

print("\nğŸ”¹ XGBoost Results")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))


def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_log),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_xgb)
    ]
})
results = results.round(4)
display(results)



xgb_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier(
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])  # handle imbalance
    ))
])


param_grid = {
    'model__n_estimators': [200, 300, 400, 500],
    'model__max_depth': [5, 7, 9],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.2],
    'model__min_child_weight': [1, 3, 5]
}


print("ğŸ”� Running RandomizedSearchCV for XGBoost... (this may take a few minutes)")

xgb_random = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_grid,
    n_iter=25,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)

print("\nâœ… Best Parameters Found:")
print(xgb_random.best_params_)


best_xgb = xgb_random.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]

print("\nğŸ”¹ XGBoost Tuned Model Performance\n")
print(classification_report(y_test, y_pred_best))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba_best), 4))



plt.figure(figsize=(8,6))
RocCurveDisplay.from_predictions(y_test, y_proba_log, name="Logistic Regression", color='blue')
RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", color='green')
RocCurveDisplay.from_predictions(y_test, y_proba_best, name="Tuned XGBoost", color='darkorange')
plt.title("ROC Curve Comparison After Tuning")
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.legend()
plt.show()


len(best_xgb.named_steps['model'].feature_importances_)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


feature_importance = best_xgb.named_steps['model'].feature_importances_


num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']


ohe = best_xgb.named_steps['preprocess'].named_transformers_['cat']


ohe_feature_names = list(ohe.get_feature_names_out(cat_features))

# Combine numeric + encoded + remainder (grade_subgrade)
all_features = num_features + ohe_feature_names + ['grade_subgrade']

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importance[:len(all_features)]
}).sort_values(by='Importance', ascending=False)

# Display Top Features
print("\nğŸ“Š Top 15 Most Important Features:")
print(importance_df.head(15))


plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'].head(15)[::-1], importance_df['Importance'].head(15)[::-1])
plt.title("Top 15 Feature Importances - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()




test_copy = test.copy()


test_copy['loan_to_income_ratio'] = test_copy['loan_amount'] / (test_copy['annual_income'] + 1)


test_copy['grade_subgrade'] = label_enc.transform(test_copy['grade_subgrade'])


test_pred_proba = best_xgb.predict_proba(test_copy)[:, 1]


submission_df = pd.DataFrame({
    'id': test_copy['id'],
    'loan_paid_back': test_pred_proba
})

# 5ï¸�Save submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


import joblib
joblib.dump(best_xgb, "best_xgb_model.pkl")
print("âœ… Model saved successfully as best_xgb_model.pkl")





