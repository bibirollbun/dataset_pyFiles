import numpy as np 
import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


# Configuration
class Config:
    """Configuration class for hyperparameters and settings"""
    TARGET = 'loan_paid_back'

config = Config()


# Separate numerical and categorical columns
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('id')

if config.TARGET in numerical_cols:
    numerical_cols.remove(config.TARGET)

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

# Combine all feature names
all_features = numerical_cols + categorical_cols
total_features   = len(all_features)
print(f"\nTotal features: {total_features}")


from sklearn.preprocessing import StandardScaler

# Store target encoding mappings globally
target_encoding_maps = {}

def transform(df):
    global target_encoding_maps
    
    ### Label Encoding On Ordinal Data
    
    # Binning - Extract first letter part from grade_subgrade to represent subgrade 
    df['grade_subgrade'] = df['grade_subgrade'].str[0]
    education_level_mapping = {"High School":0, "Bachelor's":1, "Master's":2, "PhD":3, "Other":4}
    df['education_level'] = df['education_level'].map(education_level_mapping)
    
    ## Performing One Hot Encoding On Categorical Features
    df = pd.get_dummies(df, columns=categorical_cols)

    ## To achieve Normal Distribution for skewed features
    df['annual_income']        = np.log(df['annual_income']+1)
    df['debt_to_income_ratio'] = df['debt_to_income_ratio']**(1/5)

    ## Feature Scaling ##
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df


train_en = transform(train.copy())
train_en


test_en = transform(test.copy())
test_en


columns_to_test_model = train_en.select_dtypes(include=[np.number, 'float64', 'int64', 'bool']).columns.tolist()
columns_to_test_model.remove(config.TARGET)
columns_to_test_model.remove('id')

print(f"Total Features After Transformation: {len(columns_to_test_model)}")


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler

# Separate features and target in train_en
X = train_en[columns_to_test_model]
y = train_en[config.TARGET]

# L1 Logistic Regression for feature selection
lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=5)
lasso.fit(X, y)

# Select features where coef_ != 0
model = SelectFromModel(lasso, prefit=True)
X_lasso_selected = model.transform(X)
selected_features_lasso = X.columns[model.get_support()]

print("Features selected by Lasso:")
print(selected_features_lasso)

# # Recursive Feature Elimination with Logistic Regression
# log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
# rfe = RFE(estimator=log_reg, n_features_to_select=5)  # Adjust number of features as needed
# rfe.fit(X, y)
# selected_features_rfe = X.columns[rfe.support_]

# print("Features selected by RFE:")
# print(selected_features_rfe)


# Assuming your dataset is in a DataFrame df with target column 'loan_paid_back'
majority_class = train_en[train_en['loan_paid_back'] == 1]
minority_class = train_en[train_en['loan_paid_back'] == 0]

# Randomly sample 20% from the majority class
majority_sample = majority_class.sample(frac=0.20, random_state=42)

# Combine the sampled majority with all of minority class
balanced_df = pd.concat([majority_sample, minority_class])


balanced_df['loan_paid_back'].value_counts()


train_en = balanced_df


from sklearn.model_selection import train_test_split

X = train_en[selected_features_lasso]
# X = train_en[selected_features_rfe]
y = train_en[config.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, solver='lbfgs')

model.fit(X_train, y_train)

# Predict on scaled test data
y_pred = model.predict(X_test)


from sklearn.metrics import roc_auc_score

# Predict probabilities for positive class
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC score with Logistic Regression:", roc_auc)


y_pred = model.predict(test_en[columns_to_test_model])


test_en.loc[:, 'id'] = test['id']


output = pd.DataFrame({'id': test_en['id'], 'loan_paid_back': y_pred})
output.to_csv('sample_submission.csv', index=False)


output['loan_paid_back'].value_counts()


from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Initialize model with AUC as eval metric
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

# Fit with eval set for early stopping and metric evaluation
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Predict probabilities (needed for ROC curve)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {auc_score:.4f}")

# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


y_pred = model.predict(test_en[columns_to_test_model])


test_en.loc[:, 'id'] = test['id']


output = pd.DataFrame({'id': test_en['id'], 'loan_paid_back': y_pred})
output.to_csv('sample_submission.csv', index=False)


output['loan_paid_back'].value_counts()




