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


# Data analysis and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew


# Preprocessing and feature engineering
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Evaluation metrics
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# Load the training and test datasets
data_path = "/kaggle/input/playground-series-s4e10/"
df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

print(f"Train dataset: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
print(f"Test dataset: {df_test.shape[0]} rows, {df_test.shape[1]} columns")


# Display the first 3 rows to understand the structure
df_train.head(3)


# Inspect the types of feature columns
print("\nData Info:")
df_train.info()


# Check distribution of the target variable (loan_status)
sns.set_style("whitegrid") 
sns.countplot(x='loan_status', data=df_train)
plt.title('Distribution of loan_status (1 = Approved, 0 = Not Approved)')
plt.xlabel('loan_status')
plt.ylabel('Count')
plt.show()


print("\nMissing Values in Training Set:")
print(df_train.isnull().sum())

# Define features
num_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']


for i, col in enumerate(num_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='loan_status', y=col, data=df_train)
    plt.title(f'Boxplot of {col} by loan_status')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, hue='loan_status', data=df_train)
    plt.title(f'Countplot of {col} by loan_status')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Before Transformation
print(skew(df_train['person_income'])) # highly skewed
print(skew(df_train['loan_amnt'])) # moderately skewed


# Log-transform skewed features
skewed_features = ['person_income', 'loan_amnt']
for col in skewed_features:
    df_train[f'log_{col}'] = np.log1p(df_train[col])
    df_test[f'log_{col}'] = np.log1p(df_test[col])
num_features.extend(['log_person_income', 'log_loan_amnt'])


#After Transformation

print(skew(df_train['log_person_income'])) # much closer to symmetric

print(skew(df_train['log_loan_amnt'])) # much closer to symmetric


X = df_train.drop(columns=['id', 'loan_status'])
y = df_train['loan_status']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# Before Preprocessing: Check the state of the data
print("person_income ranges from", df_train['person_income'].min(), "to", df_train['person_income'].max(), "(large scale).")
print("person_home_ownership has values like", df_train['person_home_ownership'].unique(), "(strings).")


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])


# Apply preprocessing to the training data to demonstrate the transformation
X_train_processed = preprocessor.fit_transform(X_train)

# For numerical features (after scaling)
print("person_income is scaled to mean=0, std=1.")
# For categorical features (after one-hot encoding)
encoded_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_features)
print("person_home_ownership is encoded into columns like", 
      [name for name in encoded_feature_names if 'person_home_ownership' in name], 
      "with 0/1 values.")


models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}


# Train and Evaluate Models with Cross-Validation
results = []
kf = KFold(n_splits=3, shuffle=True, random_state=42) 

for model_name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, y_val_prob)
    
    # Store results
    results.append({
        'Model': model_name,
        'Cross-Validation ROC-AUC (Mean)': cv_scores.mean(),
        'Cross-Validation ROC-AUC (Std)': cv_scores.std(),
        'Validation ROC-AUC': val_roc_auc
    })


# Display results in a table
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)


# best model based on Validation ROC-AUC
best_model_name = results_df.loc[results_df['Validation ROC-AUC'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")


# Train the best model on the full training data
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
pipeline.fit(X_train, y_train)

# Make Predictions on Test Data
X_test_final = df_test.drop(columns=['id'])
test_predictions = pipeline.predict(X_test_final)


# Create DataFrame for submission with columns 'Id' and 'Status'
submission = pd.DataFrame({'id': df_test['id'], 'loan_status': test_predictions})
submission.to_csv('/kaggle/working/submission.csv', index=False)

submission_df = pd.read_csv('/kaggle/working/submission.csv')
print(submission_df.head())
print(submission_df.shape)

