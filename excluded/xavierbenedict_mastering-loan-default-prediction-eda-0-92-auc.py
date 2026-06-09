# Data description
import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions


# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from scipy.stats import mannwhitneyu
from scipy.stats import uniform, randint, loguniform
# Explainability library
import shap
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
# Logistic Regression model
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, brier_score_loss, roc_auc_score, f1_score, accuracy_score
# Random Forest model
from sklearn.ensemble import RandomForestClassifier
# XGBoost model
from xgboost import XGBClassifier
# LightGBM model
import lightgbm as lgb
from lightgbm import LGBMClassifier
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
# CatBoost model
from catboost import CatBoostClassifier
# Neural Networks
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
# Suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')


# Load the training data
df_train = pd.read_csv('train.csv')
print("Training data shape:", df_train.shape)
df_train.head()


# Load the test data
df_test = pd.read_csv('test.csv')
print("Test data shape:", df_test.shape)
df_test.head()


# Data overview
print(f'Data info on training data: {df_train.info()}\n')
print(f'Data info on test data: {df_test.info()}')


# Check Null Values
df_train.isnull().sum()


# Check zero values
print(f'Zero values in training data:\n{df_train.isin([0]).sum()}\n')
print(f'Zero values in test data:\n{df_test.isin([0]).sum()}')



# Check negative values
numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('loan_paid_back')
print(f'Negative values in training data:\n{(df_train[numerical_cols] < 0).sum()}\n')
print(f'Negative values in test data:\n{(df_test[numerical_cols] < 0).sum()}')


# Change data types for loan_paid_back
df_train['loan_paid_back'] = df_train['loan_paid_back'].astype('int')


# Drop ID column in training data
df_train.drop('id', axis=1, inplace=True)


# Check loan_paid_back distribution
print(df_train['loan_paid_back'].value_counts())
# Paid back ratio
paid_back_ratio = df_train['loan_paid_back'].mean()
print(f'Paid back ratio: {paid_back_ratio:.2%}')
# Visualize loan_paid_back distribution

sns.countplot(x='loan_paid_back', data=df_train)
plt.title('Loan Paid Back Distribution')
plt.show()


# Split the data into training and validation sets
X = df_train.drop('loan_paid_back', axis=1)
y = df_train['loan_paid_back']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Paid back ratio in training set: {:.2%}".format(y_train.mean()))
print("Paid back ratio in validation set: {:.2%}".format(y_val.mean()))


# Separate numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


# Combine X_train and y_train for easier processing
Xy_train = pd.concat([X_train, y_train], axis=1)
# Correlation matrix for numerical features
plt.figure(figsize=(12, 10))
corr_matrix = Xy_train[numerical_cols + ['loan_paid_back']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, vmax=1.0, vmin=-1.0, center=0)
plt.title('Correlation Matrix for Numerical Features')
plt.show()


# Plot all the numerical columns of X_train to understand their distribution
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(X_train[col], bins=30, kde=True, color='blue', alpha=0.6)
plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Calculate skewness for numerical features
skewed_features = X_train[numerical_cols].apply(lambda x: skew(x.dropna()))
print("Skewness of numerical features:\n", skewed_features)


# Plot all the categorical columns of X_train to understand their distribution
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    stats.probplot(X_train[col].dropna(), dist="norm", plot=plt)
    plt.title(f'Normal Q-Q Plot of {col}')
    plt.legend([f'Skewness: {skew(X_train[col]):.2f}'])
plt.tight_layout()
plt.show()


# Plot all the categorical columns of X_train to understand their distribution
plt.figure(figsize=(10, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(y=X_train[col], order=X_train[col].value_counts().index, palette='viridis', hue=X_train[col], legend=False)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Visualise the relationship between numerical features and the target variable to understand their impact on the target outcome
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=Xy_train['loan_paid_back'], y=Xy_train[col], palette='viridis', hue=Xy_train['loan_paid_back'], legend=False)
    plt.title(f'Box Plot of {col} by Loan Paid Back')
    plt.xticks([0, 1], ['Not Paid Back', 'Paid Back'])
plt.tight_layout()
plt.show()
# Z-Test for numerical features between paid back and not paid back groups
for col in numerical_cols:
    print(Xy_train.groupby('loan_paid_back')[col].agg(['count', 'mean', 'median', 'std']).sort_values(by = 'mean', ascending = False))
    group1 = Xy_train[Xy_train['loan_paid_back'] == 1][col].dropna()
    group0 = Xy_train[Xy_train['loan_paid_back'] == 0][col].dropna()
    stat, p = stats.ttest_ind(group1, group0, equal_var=False)
    print(f'Z-Test for {col}: statistic={stat:.4f}, p-value={p:.4f}')
    if p < 0.05:
        print(f'  -> Significant difference in {col} between paid back and not paid back groups.\n')
    else:
        print(f'  -> No significant difference in {col} between paid back and not paid back groups.\n')
    


# Visualise the relationship between categorical features and the target variable to understand their impact on the target outcome
plt.figure(figsize=(12, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    # order categories by frequency (you can change to order by mean repayment if preferred)
    order = Xy_train[col].value_counts().index
    sns.barplot(x=Xy_train[col], y=Xy_train['loan_paid_back'], palette='viridis', hue=Xy_train[col], legend=False, errorbar=None, order=order)
    plt.title(f'Loan repayment likelihood by {col}')
    plt.xlabel(col)
    plt.ylabel('Loan Repayment Likelihood')
    plt.hlines(y=Xy_train['loan_paid_back'].mean(), xmin=-0.5, xmax=len(order)-0.5, color='gray', linestyle='--')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chi-Squared Test for categorical features against the target variable
from scipy.stats import chi2_contingency
for col in categorical_cols:
    print(Xy_train.groupby(col)['loan_paid_back'].agg(['count', 'mean', 'std']).sort_values(by='mean', ascending=False))
    contingency_table = pd.crosstab(Xy_train[col], Xy_train['loan_paid_back'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'{col}: Chi2={chi2}, p-value={p}')
    if p < 0.05:
        print(f'  -> Significant association between {col} and Default (p < 0.05)\n')
    else:
        print(f'  -> No significant association between {col} and Default (p >= 0.05)\n')


# Drop very weakly impactful features based on statistical tests
weak_impact_features = ['marital_status']
X_train.drop(columns=weak_impact_features, inplace=True)
X_val.drop(columns=weak_impact_features, inplace=True)


# Rank grade_subgrade by mapping letters to numbers
grade_mapping = {grade: idx for idx, grade in enumerate(sorted(X_train['grade_subgrade'].unique()), 1)}
print("Grade mapping:", grade_mapping)
# Map the grades to numerical values
X_train['grade_subgrade_num'] = X_train['grade_subgrade'].map(grade_mapping)
X_val['grade_subgrade_num'] = X_val['grade_subgrade'].map(grade_mapping)



# Subcategory encoding for grade_subgrade
def mapping_grade_subgrade(grade):
    if grade.startswith('A'):
        return 'very_low_risk'
    elif grade.startswith('B'):
        return 'low_risk'
    elif grade.startswith('C'):
        return 'moderate_risk'
    elif grade.startswith('D'):
        return 'high_risk'
    elif grade.startswith('E'):
        return 'very_high_risk'
    else:
        return 'extreme_risk'
X_train['grade_risk_category'] = X_train['grade_subgrade'].apply(mapping_grade_subgrade).astype('category')
X_val['grade_risk_category'] = X_val['grade_subgrade'].apply(mapping_grade_subgrade).astype('category')
# Drop original grade_subgrade column
X_train.drop(columns=['grade_subgrade'], inplace=True)
X_val.drop(columns=['grade_subgrade'], inplace=True)


# Define a function feature_engineering to preprocess the data
def feature_engineering(X):
    X_processed = X.copy()
    
    # 1. Interest Burden Ratio
    X_processed['interest_burden_ratio'] = (X_processed['loan_amount'] * X_processed['interest_rate']) / X_processed['annual_income']
    # 2. Credit Risk Score with respect to loan amount
    X_processed['credit_risk_score'] = X_processed['credit_score'] / X_processed['loan_amount']
    # 3. Measure of financial stress
    X_processed['financial_stress_index'] = X_processed['loan_amount'] * X_processed['debt_to_income_ratio'] / (X_processed['annual_income'])
    # 4. Financial stress score
    X_processed['financial_stress_score'] = X_processed['loan_amount'] / (X_processed['annual_income'] * (1 - X_processed['debt_to_income_ratio'])).clip(lower=1e-5)
    # 5. Loan Burden Score
    X_processed['loan_burden_score'] = X_processed['debt_to_income_ratio'] * X_processed['interest_rate']
    # 6. Credit Risk Score with respect to loan amount
    X_processed['credit_risk_score'] = X_processed['credit_score'] / X_processed['loan_amount']
    # 7. Credit Score to Debt-to-Income Ratio
    X_processed['credit_to_dti_ratio'] = X_processed['credit_score'] / (X_processed['debt_to_income_ratio'] + 1e-5)
    # 8. Loan Affordability Index
    X_processed['loan_affordability_index'] = X_processed['annual_income'] / X_processed['loan_amount']
    # 9. Low credit score flag
    X_processed['low_credit_score_flag'] = (X_processed['credit_score'] < 600).astype(int)
    # 10. High debt-to-income ratio flag
    X_processed['high_dti_flag'] = (X_processed['debt_to_income_ratio'] > X_train['debt_to_income_ratio'].median()).astype(int)
    # 11. Safe Index
    X_processed['safe_index'] = (X_processed['credit_score'] / 850) * (1 - X_processed['debt_to_income_ratio']) * (1 - X_processed['interest_rate'] / 100)
    # 12. Normalize credit score
    X_processed['normalized_credit_score'] = (X_processed['credit_score'] - X_processed['credit_score'].mean()) / np.std(X_processed['credit_score'])
    # 13. High interest burden flag
    X_processed['high_interest_burden_flag'] = (X_processed['interest_burden_ratio'] > 0.3).astype(int)
    return X_processed


# Apply feature engineering to training and validation sets
X_train = feature_engineering(X_train)
X_val = feature_engineering(X_val)


# Categorical variables
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns after feature engineering:", categorical_cols)


# Encode categorical variables using one-hot encoding
X_train_encoded = pd.get_dummies(X_train, dtype=int)
X_train_encoded = X_train_encoded.rename(columns=lambda x: x.replace(' ', '_'))
X_val_encoded = pd.get_dummies(X_val, dtype=int)
X_val_encoded = X_val_encoded.rename(columns=lambda x: x.replace(' ', '_'))
print("Encoded training set shape:", X_train_encoded.shape)
print("Encoded validation set shape:", X_val_encoded.shape)
# Align the columns of the validation set with the training set
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# Preparing the data for modeling
X_train_logistic = X_train_encoded.copy()
X_val_logistic = X_val_encoded.copy()


# Log-transform highly skewed numerical features
numerical_cols_update = X_train_logistic.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Compute skewness for those numeric columns
skewed_features = X_train_logistic[numerical_cols_update].apply(lambda x: skew(x.dropna()))
skewed_cols = skewed_features[skewed_features.abs() > 1].index.tolist()
print("Highly skewed columns to be log-transformed:", skewed_cols)
for col in skewed_cols:
    X_train_logistic[col] = np.log1p(X_train_logistic[col])
    X_val_logistic[col] = np.log1p(X_val_logistic[col])


# Scale numerical features
cols_to_scale = X_train_logistic.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X_train_logistic[cols_to_scale] = scaler.fit_transform(X_train_logistic[cols_to_scale])
X_val_logistic[cols_to_scale] = scaler.transform(X_val_logistic[cols_to_scale])


X_train_logistic


# Using statsmodels for Logistic Regression
# Add constant and fit the logistic regression model
X_train_sm_1 = sm.add_constant(X_train_logistic)
# Logistic regression model
logit_model_1 = sm.Logit(y_train, X_train_sm_1)
# Fit the model
result = logit_model_1.fit(disp=False, maxiter=100)
print(result.summary())


# VIF Calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
X_train_vif_1 = X_train_sm_1.drop(columns=['const'])
vif_data["feature"] = X_train_vif_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif_1.values, i) for i in range(X_train_vif_1.shape[1])]
vif_data.sort_values(by="VIF", ascending=False)


# Reduce multicollinearity by dropping features with high VIF
X_train_sm_2 = X_train_sm_1.drop(columns=[
    "const", "gender_Other", "education_level_Bachelor's", "employment_status_Unemployed",
    "loan_purpose_Medical", "grade_risk_category_very_high_risk", "credit_score", 
    "debt_to_income_ratio", "interest_rate","gender_Male",
    "loan_amount", "annual_income", "financial_stress_index",
    "interest_burden_ratio", "low_credit_score_flag", "education_level_Other",
    "high_interest_burden_flag", "loan_purpose_Education", "grade_risk_category_low_risk",
    "financial_stress_score", "normalized_credit_score", "credit_risk_score",
    "loan_burden_score", "grade_risk_category_very_low_risk", "credit_to_dti_ratio",
    "employment_status_Employed"
])
# Refit the model
X_train_sm_2 = sm.add_constant(X_train_sm_2)
logit_model_2 = sm.Logit(y_train, X_train_sm_2)
result2 = logit_model_2.fit()
print(result2.summary())


# Calculate VIF for each feature
X_train_vif = X_train_sm_2.drop(columns=['const'])
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
# Display VIF results
print(vif_data.sort_values(by="VIF", ascending=False))


# Keep only columns present in VIF analysis
column_to_keep = X_train_vif.columns.tolist()
X_train_logistic_final = X_train_logistic[column_to_keep]
X_val_logistic_final = X_val_logistic[column_to_keep]
# Predict using the final logistic regression model
y_train_pred_lr = result2.predict(sm.add_constant(X_train_logistic_final))
y_val_pred_lr = result2.predict(sm.add_constant(X_val_logistic_final))
# ROC-AUC scores
train_roc_auc_lr = roc_auc_score(y_train, y_train_pred_lr)
val_roc_auc_lr = roc_auc_score(y_val, y_val_pred_lr)
print("Logistic Regression Model -  Train set ROC-AUC:", train_roc_auc_lr)
print("Logistic Regression Model -  Validation set ROC-AUC:", val_roc_auc_lr)


# Evaluate calibration curve
title = 'Calibration Curve for Logistic Regression Model'
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_lr, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_lr, n_bins=10) 
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label='Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title(title)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_lr = brier_score_loss(y_train, y_train_pred_lr)
brier_score_val_lr = brier_score_loss(y_val, y_val_pred_lr)
print(f"Logistic Brier Score for Logistic Regression - Train set: {brier_score_train_lr:.4f}")
print(f"Logistic Brier Score for Logistic Regression - Validation set: {brier_score_val_lr:.4f}")


# Precision-Recall Curve
precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, y_train_pred_lr)
precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, y_val_pred_lr)
plt.figure(figsize=(8, 6))
plt.plot(recall_train, precision_train, label='Train')
plt.plot(recall_val, precision_val, label='Validation')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve)
average_precision_train_lr = average_precision_score(y_train, y_train_pred_lr)
average_precision_val_lr = average_precision_score(y_val, y_val_pred_lr)
print(f"Logistic Regression - Train set Average Precision Score: {average_precision_train_lr:.4f}")
print(f"Logistic Regression - Validation set Average Precision Score: {average_precision_val_lr:.4f}")


# Pick threshold based on business needs (maximizing F1-score)
f1_scores_logistic = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
best_threshold_index = np.argmax(f1_scores_logistic)
best_threshold_logistic = thresholds_val[best_threshold_index]
print(f"Best threshold for Logistic Regression based on F1-score: {best_threshold_logistic:.4f}")
# Plot Precision-Recall Curve with best threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_val, precision_val[:-1], label='Precision')
plt.plot(thresholds_val, recall_val[:-1], label='Recall')
plt.axvline(x=best_threshold_logistic, color='red', linestyle='--', label='Best Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best Threshold')
plt.legend()
plt.show()


# Evaluate model at the best threshold
y_train_pred_logistic = (y_train_pred_lr >= best_threshold_logistic).astype(int)
y_val_pred_logistic = (y_val_pred_lr >= best_threshold_logistic).astype(int)

print("Logistic Regression Model Evaluation at Best Threshold:")
print(f"Train Classification Report:\n{classification_report(y_train, y_train_pred_logistic)}")
print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred_logistic)}")



X_train_tree = X_train_encoded.copy()
X_val_tree = X_val_encoded.copy()
print("X_train_tree shape:", X_train_tree.shape)
print("X_val_tree shape:", X_val_tree.shape)


# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=100, n_jobs=-1)
rf_model.fit(X_train_tree, y_train)
# Predict probabilities
y_train_pred_prob_rf = rf_model.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_rf = rf_model.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
train_roc_auc_rf = roc_auc_score(y_train, y_train_pred_prob_rf)
val_roc_auc_rf = roc_auc_score(y_val, y_val_pred_prob_rf)
print("Random Forest Model - Train set ROC-AUC:", train_roc_auc_rf)
print("Random Forest Model - Validation set ROC-AUC:", val_roc_auc_rf)


'''
# Use Randomized Search CV for Hyperparameter Tuning of Random Forest
param_dist = {
    'n_estimators': [200, 250],
    'max_depth': [8, 10],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [8, 12],
    'max_features': ['sqrt', 0.5],
    'class_weight': ['balanced'],
    'max_samples': [0.7, 0.8],
    'oob_score': [True],
    'bootstrap': [True]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
rf = RandomForestClassifier(random_state=100, n_jobs=-1)
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    random_state=100,
    n_jobs=-1
)
rf_random_search.fit(X_train_tree, y_train)
# Best parameters from Randomized Search
best_params_rf = rf_random_search.best_params_
print("Best Parameters from Randomized Search:", best_params_rf)
#----------------------------------------------------------------
Fitting 3 folds for each of 15 candidates, totalling 45 fits
Best Parameters from Randomized Search: 
{'oob_score': True, 
 'n_estimators': 250, 
 'min_samples_split': 15, 
 'min_samples_leaf': 12, 
 'max_samples': 0.8, 
 'max_features': 0.5, 
 'max_depth': 10, 
 'class_weight': 'balanced', 
 'bootstrap': True}
'''


# Final Random Forest model with tuned hyperparameters
best_params_rf = {
    'oob_score': True, 
    'n_estimators': 250, 
    'min_samples_split': 15, 
    'min_samples_leaf': 12, 
    'max_samples': 0.8, 
    'max_features': 0.5, 
    'max_depth': 10, 
    'class_weight': 'balanced', 
    'bootstrap': True
    }
rf_model_tuned = RandomForestClassifier(
    **best_params_rf,
    random_state=100,
    n_jobs=-1
)
# Fit the tuned model
rf_model_tuned.fit(X_train_tree, y_train)


# Predict probabilities
y_train_pred_prob_rf_tuned = rf_model_tuned.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_rf_tuned = rf_model_tuned.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
train_roc_auc_rf_tuned = roc_auc_score(y_train, y_train_pred_prob_rf_tuned)
val_roc_auc_rf_tuned = roc_auc_score(y_val, y_val_pred_prob_rf_tuned)
print(f"Tuned Random Forest Model -  Train set ROC-AUC: {train_roc_auc_rf_tuned:.4f}")
print(f"Tuned Random Forest Model -  Validation set ROC-AUC: {val_roc_auc_rf_tuned:.4f}")


# Evaluate calibration curve
title = 'Calibration Curve for Tuned Random Forest Model'
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob_rf_tuned, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_prob_rf_tuned, n_bins=10) 
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Calibration curve - Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label='Calibration curve - Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title(title)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_rf = brier_score_loss(y_train, y_train_pred_prob_rf_tuned)
brier_score_val_rf = brier_score_loss(y_val, y_val_pred_prob_rf_tuned)
print(f"Tuned Random Forest - Train set Brier Score: {brier_score_train_rf:.4f}")
print(f"Tuned Random Forest - Validation set Brier Score: {brier_score_val_rf:.4f}")


# Precision-Recall Curve
precision_train_rf, recall_train_rf, thresholds_train_rf = precision_recall_curve(y_train, y_train_pred_prob_rf_tuned)
precision_val_rf, recall_val_rf, thresholds_val_rf = precision_recall_curve(y_val, y_val_pred_prob_rf_tuned)
plt.figure(figsize=(8, 6))
plt.plot(recall_train_rf, precision_train_rf, label='Precision-Recall curve - Train')
plt.plot(recall_val_rf, precision_val_rf, label='Precision-Recall curve - Validation')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve of Random Forest Model')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve)
average_precision_train_rf = average_precision_score(y_train, y_train_pred_prob_rf_tuned)
average_precision_val_rf = average_precision_score(y_val, y_val_pred_prob_rf_tuned)
print(f"Tuned Random Forest - Train set Average Precision Score: {average_precision_train_rf:.4f}")
print(f"Tuned Random Forest - Validation set Average Precision Score: {average_precision_val_rf:.4f}")


# Pick threshold based on business needs (maximizing F1-score)
f1_scores_rf = 2 * (precision_val_rf * recall_val_rf) / (precision_val_rf + recall_val_rf + 1e-10)
best_threshold_index_rf = np.argmax(f1_scores_rf)
best_threshold_rf = thresholds_val_rf[best_threshold_index_rf]
print(f"Best threshold for Random Forest based on F1-score: {best_threshold_rf:.4f}")
# Plot Precision-Recall Curve with best threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_val_rf, precision_val_rf[:-1], label='Precision')
plt.plot(thresholds_val_rf, recall_val_rf[:-1], label='Recall')
plt.axvline(x=best_threshold_rf, color='red', linestyle='--', label='Best Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best Threshold for Random Forest')
plt.legend()
plt.show()


# Evaluate model at the best threshold
y_train_pred_rf_best = (y_train_pred_prob_rf_tuned >= best_threshold_rf).astype(int)
y_val_pred_rf_best = (y_val_pred_prob_rf_tuned >= best_threshold_rf).astype(int)

print("Random Forest Model Evaluation at Best Threshold:")
print(f"Train Classification Report:\n{classification_report(y_train, y_train_pred_rf_best)}")
print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred_rf_best)}")


# Feature importance from Tuned Random Forest
importances = rf_model_tuned.feature_importances_
feature_names = X_train_tree.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(8, 12))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis', hue='Importance', legend=False)
plt.title('Top 20 Feature Importances from Tuned Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# XGBoost model
xgb_model = XGBClassifier(random_state=100, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_tree, y_train)
# Predict probabilities
y_train_pred_prob_xgb = xgb_model.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
train_roc_auc_xgb = roc_auc_score(y_train, y_train_pred_prob_xgb)
val_roc_auc_xgb = roc_auc_score(y_val, y_val_pred_prob_xgb)
print("XGBoost Model - Train set ROC-AUC:", train_roc_auc_xgb)
print("XGBoost Model - Validation set ROC-AUC:", val_roc_auc_xgb)


# Calculate scale_pos_weight for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Scale Pos Weight: {scale_pos_weight:.2f}")


# Use Randomized Search CV for Hyperparameter Tuning of XGBoost
xgb = XGBClassifier(
    objective='binary:logistic',
    random_state=100, 
    n_jobs=-1, 
    use_label_encoder=False, 
    eval_metric='auc'
)
param_dist_xgb = {
    'n_estimators': randint(500, 1201),
    'learning_rate': uniform(0.001, 1.0 - 0.001),
    'max_depth': randint(3, 11),
    'min_child_weight': randint(5, 11),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.6, 0.4),
    'scale_pos_weight': uniform(scale_pos_weight - 0.1, 0.2),
    'gamma': uniform(0.2, 1.0 - 0.2),
    'reg_alpha': loguniform(1e-8, 1.0),
    'reg_lambda': loguniform(1e-6, 1.0)
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)

xgb_random_search = RandomizedSearchCV(estimator=xgb, 
                                       param_distributions=param_dist_xgb, 
                                       n_iter=15, 
                                       scoring='roc_auc', 
                                       cv=cv, 
                                       random_state=100, 
                                       n_jobs=-1)
xgb_random_search.fit(X_train_tree, y_train)

# Best parameters from Randomized Search
best_params_xgb = xgb_random_search.best_params_
print("Best Parameters from Randomized Search for XGBoost:", best_params_xgb)


# Train XGBoost with best parameters
xgb_model_tuned = XGBClassifier(**best_params_xgb, 
                         random_state=100, 
                         n_jobs=-1, 
                         use_label_encoder=False, 
                         eval_metric='auc')
xgb_model_tuned.fit(X_train_tree, y_train)


# Predict and evaluate the tuned XGBoost model
y_train_pred_prob_xgb_tuned = xgb_model_tuned.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_xgb_tuned = xgb_model_tuned.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
roc_auc_train_xgb_model_tuned = roc_auc_score(y_train, y_train_pred_prob_xgb_tuned)
roc_auc_val_xgb_model_tuned = roc_auc_score(y_val, y_val_pred_prob_xgb_tuned)
print(f"XGBoost - Train set ROC-AUC Score: {roc_auc_train_xgb_model_tuned:.4f}")
print(f"XGBoost - Validation set ROC-AUC Score: {roc_auc_val_xgb_model_tuned:.4f}")


# Evaluate calibration curve
title = 'Calibration Curve for Tuned XGBoost Model'
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob_xgb_tuned, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_prob_xgb_tuned, n_bins=10) 
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label=' Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label=' Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title(title)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_xgb = brier_score_loss(y_train, y_train_pred_prob_xgb_tuned)
brier_score_val_xgb = brier_score_loss(y_val, y_val_pred_prob_xgb_tuned)
print(f"XGBoost - Train set Brier Score: {brier_score_train_xgb:.4f}")
print(f"XGBoost - Validation set Brier Score: {brier_score_val_xgb:.4f}")


# Precision-Recall Curve
precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, y_train_pred_prob_xgb_tuned)
precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, y_val_pred_prob_xgb_tuned)
plt.figure(figsize=(8, 6))
plt.plot(recall_train, precision_train, label='Train')
plt.plot(recall_val, precision_val, label='Validation')
plt.title('Precision-Recall Curve for Tuned XGBoost Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve)
average_precision_train_xgb = average_precision_score(y_train, y_train_pred_prob_xgb_tuned)
average_precision_val_xgb = average_precision_score(y_val, y_val_pred_prob_xgb_tuned)
print(f"XGBoost - Train set Average Precision Score: {average_precision_train_xgb:.4f}")
print(f"XGBoost - Validation set Average Precision Score: {average_precision_val_xgb:.4f}")


# Pick threshold based on business needs (maximizing F1-score)
f1_scores = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds_val[best_threshold_index]
print(f"Best Threshold (F1-score): {best_threshold:.4f}")
# Plot Precision-Recall Curve with best threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_val, precision_val[:-1], label='Precision')
plt.plot(thresholds_val, recall_val[:-1], label='Recall')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best Threshold')
plt.title('Precision-Recall Curve for Tuned XGBoost Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# Evaluate model at the best threshold
y_train_pred_best = (y_train_pred_prob_xgb_tuned >= best_threshold).astype(int)
y_val_pred_best = (y_val_pred_prob_xgb_tuned >= best_threshold).astype(int)
print(f"Train Classification Report (Best Threshold):\n{classification_report(y_train, y_train_pred_best)}")
print(f"Validation Classification Report (Best Threshold):\n{classification_report(y_val, y_val_pred_best)}")


# Feature importance from Tuned XGBoost
importances = xgb_model_tuned.feature_importances_
feature_names = X_train_tree.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(8, 12))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis', hue='Importance', legend=False)
plt.title('Top 20 Feature Importances from Tuned XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# LightGBM model
lgbm_model = LGBMClassifier(random_state=100, n_jobs=-1)
lgbm_model.fit(X_train_tree, y_train)
# Predict probabilities
y_train_pred_prob_lgbm = lgbm_model.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_lgbm = lgbm_model.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
train_roc_auc_lgbm = roc_auc_score(y_train, y_train_pred_prob_lgbm)
val_roc_auc_lgbm = roc_auc_score(y_val, y_val_pred_prob_lgbm)
print("LightGBM Model - Train set ROC-AUC:", train_roc_auc_lgbm)
print("LightGBM Model - Validation set ROC-AUC:", val_roc_auc_lgbm)


import logging
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Use Randomized Search CV for Hyperparameter Tuning of LightGBM
lgbm = LGBMClassifier(
    objective='binary',
    random_state=100, 
    n_jobs=-1, 
    metric='auc',
    importance_type='split'
)
param_dist_lgbm = {
    'n_estimators': randint(700, 2501),
    'num_leaves': randint(15, 64),
    'learning_rate': uniform(0.03, 1.0 - 0.03),
    'max_depth': randint(5, 9),
    'min_child_samples': randint(20, 51),
    'reg_alpha': uniform(0.001, 1.0 - 0.001),
    'reg_lambda': uniform(0.001, 1.0 - 0.001),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

lgbm_random_search = RandomizedSearchCV(estimator=lgbm,
                                        param_distributions=param_dist_lgbm,
                                        n_iter=20,
                                        scoring='roc_auc',
                                        cv=cv,
                                        random_state=100,
                                        n_jobs=-1,
                                        verbose=1)

lgbm_random_search.fit(
    X_train_tree, y_train,
    eval_set=[(X_val_tree, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(100), log_evaluation(50)]
)
# Best parameters from Randomized Search
best_params_lgbm = lgbm_random_search.best_params_
print("Best Parameters from Randomized Search for LightGBM:", best_params_lgbm)


import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
# Predict and evaluate the tuned LightGBM model
#best_params_lgbm1 = {'learning_rate': 0.14131103464830524, 'max_depth': 6, 'min_child_samples': 22, 'n_estimators': 753, 'num_leaves': 16, 'reg_alpha': 0.9666430847928362, 'reg_lambda': 0.9570555877524454}
lgbm_model_tuned = LGBMClassifier(
    **best_params_lgbm,
    objective='binary',
    random_state=100, 
    n_jobs=-1, 
    metric='auc',
    importance_type='split' 
)
# Fit the tuned model
lgbm_model_tuned.fit(X_train_tree, y_train)


# Predict probabilities
y_train_pred_prob_lgbm_tuned = lgbm_model_tuned.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_lgbm_tuned = lgbm_model_tuned.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
roc_auc_train_lgbm_tuned = roc_auc_score(y_train, y_train_pred_prob_lgbm_tuned)
roc_auc_val_lgbm_tuned = roc_auc_score(y_val, y_val_pred_prob_lgbm_tuned)
print(f"LightGBM - Train set ROC-AUC: {roc_auc_train_lgbm_tuned:.4f}")
print(f"LightGBM - Validation set ROC-AUC: {roc_auc_val_lgbm_tuned:.4f}")


# Evaluate calibration curve
title = 'Calibration Curve for Tuned LightGBM Model'
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob_lgbm_tuned, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_prob_lgbm_tuned, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label='Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title(title)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_lgbm = brier_score_loss(y_train, y_train_pred_prob_lgbm_tuned)
brier_score_val_lgbm = brier_score_loss(y_val, y_val_pred_prob_lgbm_tuned)
print(f"Tuned LightGBM - Train set Brier Score: {brier_score_train_lgbm:.4f}")
print(f"Tuned LightGBM - Validation set Brier Score: {brier_score_val_lgbm:.4f}")


# Precision-Recall Curve
precision_train_lgbm, recall_train_lgbm, thresholds_train_lgbm = precision_recall_curve(y_train, y_train_pred_prob_lgbm_tuned)
precision_val_lgbm, recall_val_lgbm, thresholds_val_lgbm = precision_recall_curve(y_val, y_val_pred_prob_lgbm_tuned)
plt.figure(figsize=(8, 6))
plt.plot(recall_train_lgbm, precision_train_lgbm, label='Train')
plt.plot(recall_val_lgbm, precision_val_lgbm, label='Validation')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve of LightGBM Model')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve)
average_precision_train_lgbm = average_precision_score(y_train, y_train_pred_prob_lgbm_tuned)
average_precision_val_lgbm = average_precision_score(y_val, y_val_pred_prob_lgbm_tuned)
print(f"Tuned LightGBM - Train set Average Precision Score: {average_precision_train_lgbm:.4f}")
print(f"Tuned LightGBM - Validation set Average Precision Score: {average_precision_val_lgbm:.4f}")


# Pick threshold based on business needs (maximizing F1-score)
f1_scores_lgbm = 2 * (precision_val_lgbm * recall_val_lgbm) / (precision_val_lgbm + recall_val_lgbm + 1e-10)
best_threshold_index_lgbm = np.argmax(f1_scores_lgbm)
best_threshold_lgbm = thresholds_val_lgbm[best_threshold_index_lgbm]
print(f"Best threshold for LightGBM based on F1-score: {best_threshold_lgbm:.4f}")
# Plot Precision-Recall Curve with best threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_val_lgbm, precision_val_lgbm[:-1], label='Precision')
plt.plot(thresholds_val_lgbm, recall_val_lgbm[:-1], label='Recall')
plt.axvline(x=best_threshold_lgbm, color='red', linestyle='--', label='Best Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best Threshold for LightGBM')
plt.legend()
plt.show()


# Evaluate model at the best threshold
y_train_pred_lgbm_tuned = (y_train_pred_prob_lgbm_tuned >= best_threshold_lgbm).astype(int)
y_val_pred_lgbm_tuned = (y_val_pred_prob_lgbm_tuned >= best_threshold_lgbm).astype(int)
print("LightGBM Model Evaluation at Best Threshold:")
print(f"Train Classification Report:\n{classification_report(y_train, y_train_pred_lgbm_tuned)}")
print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred_lgbm_tuned)}")


# Important feature from Tuned LightGBM
importances = lgbm_model_tuned.feature_importances_
feature_names = X_train_tree.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(8, 12))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', hue='Importance', legend=False)
plt.title('Feature Importance from Tuned LightGBM')
plt.show()


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# Stacking Classifier with Tuned Models
# Define base learners
estimators = [
    ('rf', rf_model_tuned),
    ('xgb', xgb_model_tuned),
    ('lgbm', lgbm_model_tuned)
]
# Meta-model
meta_model = LogisticRegression(max_iter=2000, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    n_jobs=-1,
    cv=cv
)
# Fit stacking classifier
stacking_clf.fit(X_train_tree, y_train)


# Predict and evaluate the Stacking Classifier
y_train_pred_prob_stack = stacking_clf.predict_proba(X_train_tree)[:, 1]
y_val_pred_prob_stack = stacking_clf.predict_proba(X_val_tree)[:, 1]
# ROC-AUC scores
train_roc_auc_stack = roc_auc_score(y_train, y_train_pred_prob_stack)
val_roc_auc_stack = roc_auc_score(y_val, y_val_pred_prob_stack)
print(f"Stacking Classifier - Train set ROC-AUC: {train_roc_auc_stack:.4f}")
print(f"Stacking Classifier - Validation set ROC-AUC: {val_roc_auc_stack:.4f}")


# Calibration curve for stacking classifier
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob_stack, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_prob_stack, n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label='Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve - Stacking Classifier')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_stack = brier_score_loss(y_train, y_train_pred_prob_stack)
brier_score_val_stack = brier_score_loss(y_val, y_val_pred_prob_stack)
print(f"Stacking Classifier - Train set Brier Score: {brier_score_train_stack:.4f}")
print(f"Stacking Classifier - Validation set Brier Score: {brier_score_val_stack:.4f}")


# Precision-Recall Curve for stacking classifier
precision_train_stack, recall_train_stack, thresholds_train_stack = precision_recall_curve(y_train, y_train_pred_prob_stack)
precision_val_stack, recall_val_stack, thresholds_val_stack = precision_recall_curve(y_val, y_val_pred_prob_stack)
plt.figure(figsize=(10, 6))
plt.plot(recall_train_stack, precision_train_stack, label='Train')
plt.plot(recall_val_stack, precision_val_stack, label='Validation')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Stacking Classifier')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve) for stacking classifier
average_precision_train_stack = average_precision_score(y_train, y_train_pred_prob_stack)
average_precision_val_stack = average_precision_score(y_val, y_val_pred_prob_stack)
print(f"Stacking Classifier - Train set Average Precision Score: {average_precision_train_stack:.4f}")
print(f"Stacking Classifier - Validation set Average Precision Score: {average_precision_val_stack:.4f}")


# Pick threshold based on business needs (maximizing F1-score) for stacking classifier
f1_scores_stack = 2 * (precision_val_stack * recall_val_stack) / (precision_val_stack + recall_val_stack + 1e-10)
best_threshold_index_stack = np.argmax(f1_scores_stack)
best_threshold_stack = thresholds_val_stack[best_threshold_index_stack]
print("Best threshold for stacking classifier based on F1-score:", best_threshold_stack)
# Plot Precision-Recall Curve with best threshold for stacking classifier
plt.figure(figsize=(10, 6))
plt.plot(thresholds_val_stack, precision_val_stack[:-1], label='Precision')
plt.plot(thresholds_val_stack, recall_val_stack[:-1], label='Recall')
plt.axvline(x=best_threshold_stack, color='r', linestyle='--', label='Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Curve with Best Threshold - Stacking Classifier')
plt.legend()
plt.show()


# Evaluate model at the best threshold for stacking classifier
y_train_pred_stack = (y_train_pred_prob_stack >= best_threshold_stack).astype(int)
y_val_pred_stack = (y_val_pred_prob_stack >= best_threshold_stack).astype(int)
print("Stacking Classifier Evaluation at Best Threshold:")
print(f"Train Classification Report:\n{classification_report(y_train, y_train_pred_stack)}")
print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred_stack)}")


# Prepare data for neural network
X_train_nn = X_train_logistic.copy()
X_val_nn = X_val_logistic.copy()
print("X_train_nn shape:", X_train_nn.shape)
print("X_val_nn shape:", X_val_nn.shape)
print(X_train_nn.info())
print(X_val_nn.info())


# Check nan and infinite values
print("Checking for NaN and infinite values in X_train_nn:")
print("NaN values:\n", X_train_nn.isna().sum())
print("Infinite values:\n", np.isinf(X_train_nn).sum())
print("Checking for NaN and infinite values in X_val_nn:")
print("NaN values:\n", X_val_nn.isna().sum())
print("Infinite values:\n", np.isinf(X_val_nn).sum())


# Ensure that the validation set has the same columns as the training set
X_val_nn = X_val_nn.reindex(columns=X_train_nn.columns, fill_value=0)


# Clean previous Keras session
tf.keras.backend.clear_session()
# Build Neural Network model
nn_model = Sequential(
    [
        tf.keras.Input(shape=(int(X_train_nn.shape[1]),)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),        
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),        
        Dense(1, activation='sigmoid')
    ]
)
# Optimizer with initial learning rate (use a float so ReduceLROnPlateau can modify it)
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
# Compile the model
metric = tf.keras.metrics.AUC(name='auc')
nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metric])
# Callback for early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=10, mode='max', min_lr=1e-6, verbose=1)
# Train the model
history = nn_model.fit(
    X_train_nn, y_train,
    validation_data=(X_val_nn, y_val),
    epochs=100,
    batch_size=1024,
    callbacks=[early_stopping_callback, rlr],
    verbose=1
)


# Predict probabilities
y_train_pred_prob_nn = nn_model.predict(X_train_nn).ravel()
y_val_pred_prob_nn = nn_model.predict(X_val_nn).ravel()
# ROC-AUC scores
train_roc_auc_nn = roc_auc_score(y_train, y_train_pred_prob_nn)
val_roc_auc_nn = roc_auc_score(y_val, y_val_pred_prob_nn)
print(f"Neural Network - Train set ROC-AUC: {train_roc_auc_nn:.4f}")
print(f"Neural Network - Validation set ROC-AUC: {val_roc_auc_nn:.4f}")


# Calibration curve for neural network
prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob_nn, n_bins=10)
prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred_prob_nn, n_bins=10)
plt.figure(figsize=(10, 6))
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Train')
plt.plot(prob_pred_val, prob_true_val, marker='o', label='Validation')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve - Neural Network')
plt.legend()
plt.show()


# Quantify calibration using Brier Score
brier_score_train_nn = brier_score_loss(y_train, y_train_pred_prob_nn)
brier_score_val_nn = brier_score_loss(y_val, y_val_pred_prob_nn)
print(f"Neural Network - Train set Brier Score: {brier_score_train_nn:.4f}")
print(f"Neural Network - Validation set Brier Score: {brier_score_val_nn:.4f}")


# Precision-Recall Curve for neural network
precision_train_nn, recall_train_nn, thresholds_train_nn = precision_recall_curve(y_train, y_train_pred_prob_nn)
precision_val_nn, recall_val_nn, thresholds_val_nn = precision_recall_curve(y_val, y_val_pred_prob_nn)
plt.figure(figsize=(10, 6))
plt.plot(recall_train_nn, precision_train_nn, label='Train')
plt.plot(recall_val_nn, precision_val_nn, label='Validation')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Neural Network')
plt.legend()
plt.show()


# Average Precision Score (Area under Precision-Recall Curve)
average_precision_train_nn = average_precision_score(y_train, y_train_pred_prob_nn)
average_precision_val_nn = average_precision_score(y_val, y_val_pred_prob_nn)
print(f"Neural Network - Train set Average Precision Score: {average_precision_train_nn:.4f}")
print(f"Neural Network - Validation set Average Precision Score: {average_precision_val_nn:.4f}")


# Pick threshold based on business needs (maximizing F1-score) for neural network
f1_scores_nn = 2 * (precision_val_nn * recall_val_nn) / (precision_val_nn + recall_val_nn + 1e-10)
best_threshold_index_nn = np.argmax(f1_scores_nn)
best_threshold_nn = thresholds_val_nn[best_threshold_index_nn]
print("Best threshold for neural network based on F1-score:", best_threshold_nn)
# Plot Precision-Recall Curve with best threshold for neural network
plt.figure(figsize=(10, 6))
plt.plot(thresholds_val_nn, precision_val_nn[:-1], label='Precision')
plt.plot(thresholds_val_nn, recall_val_nn[:-1], label='Recall')
plt.axvline(x=best_threshold_nn, color='r', linestyle='--', label='Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold - Neural Network')
plt.legend()
plt.show()


# Evaluate model at the best threshold for neural network
y_train_pred_nn = (y_train_pred_prob_nn >= best_threshold_nn).astype(int)
y_val_pred_nn = (y_val_pred_prob_nn >= best_threshold_nn).astype(int)
print("Neural Network Evaluation at Best Threshold:")
print(f"Train Classification Report:\n{classification_report(y_train, y_train_pred_nn)}")
print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred_nn)}")


models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Neural Network', 'Stacking Classifier']
model_val_results = {
    'ROC-AUC': [val_roc_auc_lr, val_roc_auc_rf_tuned, roc_auc_val_xgb_model_tuned, roc_auc_val_lgbm_tuned, val_roc_auc_nn, val_roc_auc_stack],
    'Brier Score': [brier_score_val_lr, brier_score_val_rf, brier_score_val_xgb, brier_score_val_lgbm, brier_score_val_nn, brier_score_val_stack],
    'Average Precision (AUC-PR)': [average_precision_val_lr, average_precision_val_rf, average_precision_val_xgb, average_precision_val_lgbm, average_precision_val_nn, average_precision_val_stack],
}
model_comparison_df = pd.DataFrame(model_val_results, index=models)
print("Model Comparison on Validation Set:")
print(model_comparison_df.sort_values(by='ROC-AUC', ascending=False))


# Plot model comparison
model_comparison_df.plot(kind='bar', figsize=(12, 8))
plt.title("Model Comparison on Validation Set")
# Horizontal line for average of ROC-AUC
plt.axhline(y=model_comparison_df['ROC-AUC'].mean(), color='b', linestyle='--', label='Average ROC-AUC')
plt.axhline(y=model_comparison_df['Average Precision (AUC-PR)'].mean(), color='g', linestyle='--', label='Average AUC-PR')
plt.axhline(y=model_comparison_df['Brier Score'].mean(), color='r', linestyle='--', label='Average Brier Score')
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Porfolios high ROC-AUC and Average Precision Score
p_lighgbm = y_val_pred_prob_lgbm_tuned
p_stack = y_val_pred_prob_stack
p_xgb = y_val_pred_prob_xgb_tuned
best = None
np.random.seed(100)
for _ in range(3000):
    weights = np.random.dirichlet(np.ones(3), size=1)[0]
    p_ensemble = (
        (weights[0] * p_lighgbm) + \
        (weights[1] * p_xgb) + \
        (weights[2] * p_stack)
    )
    roc_auc = roc_auc_score(y_val, p_ensemble)
    avg_precision = average_precision_score(y_val, p_ensemble)
    brier = brier_score_loss(y_val, p_ensemble)
    if best is None or roc_auc > best[0]:
        best = (roc_auc, avg_precision, brier, weights)
best_auc, best_ap, best_brier, best_weights = best

print(f"Best Ensemble Weights (Random Search) - LightGBM: {best_weights[0]:.2f}, XGBoost: {best_weights[1]:.2f}, Stacking: {best_weights[2]:.2f}")
print(f"Ensemble ROC-AUC: {best_auc:.4f}, Average Precision: {best_ap:.4f}, Brier Score: {best_brier:.4f}")


# Load the test data
df_test = pd.read_csv('test.csv')
print("Test data shape:", df_test.shape)
df_test.head()


# Check the structure of the test data
df_test.info()


# Check for missing values in test data
missing_values_test = df_test.isnull().sum()
print("Missing values in test data:\n", missing_values_test)


# Subcategory encoding for grade_subgrade
df_test['grade_risk_category'] = df_test['grade_subgrade'].apply(mapping_grade_subgrade).astype('category')
# Drop original grade_subgrade column
df_test = df_test.drop(columns=['grade_subgrade'])


# Apply feature engineering on test data
df_test_fe = feature_engineering(df_test)


# Categorial encoding on test data
catagorical_cols = df_test_fe.select_dtypes(include=['object', 'category']).columns.tolist()
df_test_encoded = pd.get_dummies(df_test_fe, columns=catagorical_cols, dtype=int)
df_test_encoded = df_test_encoded.rename(columns=lambda x: x.strip().replace(' ', '_').replace('-', '_'))
# Align test data columns with training data columns
df_test_encoded = df_test_encoded.reindex(columns=X_train_logistic.columns, fill_value=0)
print("Encoded test data shape:", df_test_encoded.shape)


df_test_encoded.head()


# Final Ensemble Weights (Random Search) - LightGBM: 0.47, XGBoost: 0.03, Stacking: 0.50
p_lighgbm_test = lgbm_model_tuned.predict_proba(df_test_encoded)[:, 1]
p_xgb_test = xgb_model_tuned.predict_proba(df_test_encoded)[:, 1]
p_stack_test = stacking_clf.predict_proba(df_test_encoded)[:, 1]
p_ensemble_test = (0.47 * p_lighgbm_test) + (0.03 * p_xgb_test) + (0.50 * p_stack_test)



# Probability DataFrame for test data
p_df = pd.DataFrame({
    'id': df_test['id'],
    'lightgbm_probability': p_lighgbm_test,
    'xgboost_probability': p_xgb_test,
    'stacking_probability': p_stack_test,
    'loan_repaid_probability': p_ensemble_test
})
p_df.head()


# Final prediction dataframe
prediction_df = pd.DataFrame({
    'id': df_test['id'],    
    'loan_repaid_probability': p_ensemble_test
})
print(f'Shape of prediction dataframe: {prediction_df.shape}')
print(prediction_df.head(10))


# Save predictions to CSV
prediction_df.to_csv('loan_repaid_probability_predictions.csv', index=False)
print("Predictions saved to 'loan_repaid_probability_predictions.csv'")
submission = pd.read_csv('loan_repaid_probability_predictions.csv')
print("Submission file preview:")
print(submission.head(10))

