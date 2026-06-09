pip show scikit-learn


#pip install --upgrade scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


train_df=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


train_df.head(20)


train_df.isnull().sum()


train_df.describe()


train_df.info()


train_df.shape


train_df.corr(numeric_only=True)


train_df=train_df.drop('id', axis=1)
test_id=test_df['id']
test_df=test_df.drop('id', axis=1)


train_df=train_df.drop(columns=['person_age','cb_person_cred_hist_length'], axis=1)
test_df=test_df.drop(columns=['person_age','cb_person_cred_hist_length'], axis=1)


# Distribution of loan amounts
sns.histplot(train_df['loan_amnt'], bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.show()


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Count plot for loan status
sns.countplot(x='loan_status', data=train_df)
plt.title('Loan Status Count')
plt.show()


# Create a new feature for income to loan amount ratio
train_df['income_to_loan_ratio'] = train_df['person_income'] / train_df['loan_amnt']
test_df['income_to_loan_ratio'] = test_df['person_income'] / test_df['loan_amnt']


#Loan Amount per Year of Employment
train_df['loan_per_emp_year'] = train_df['loan_amnt'] / (train_df['person_emp_length'] + 1)
test_df['loan_per_emp_year'] = test_df['loan_amnt'] / (test_df['person_emp_length'] + 1)


train_df.head()


preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']),
        ('label', OrdinalEncoder(), ['loan_grade']),
        ('scaler', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64']))
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

X = train_df.drop(columns=['loan_status']) 
y = train_df['loan_status']

X_preprocessed = pipeline.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Original class distribution:")
print(y_train.value_counts())

print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts())


# Train a Random Forest model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# Initialize the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# Fit the model on the resampled data
gb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_gb))
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb)}')


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')


# Fit GridSearchCV
grid_search.fit(X_resampled, y_resampled)

# Best parameters from grid search
print("Best parameters:", grid_search.best_params_)

# Use the best model found
best_gb_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_gb_model.predict(X_test)

# Evaluate the best model
print(classification_report(y_test, y_pred_best))
print(f'Accuracy: {accuracy_score(y_test, y_pred_best)}')


X_test_preprocessed = pipeline.transform(test_df)

y_pred = best_gb_model.predict(X_test_preprocessed)


# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_id,  # Replace test_id with your actual test ID array
    'loan_status': None  # Placeholder for predictions
})

# Now, make predictions using your model
y_pred = best_gb_model.predict(X_test_preprocessed)

# Assign predictions to the submission DataFrame
submission['loan_status'] = y_pred


submission


model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'gradient_boosting_model2.pkl')


# Save the trained Gradient Boosting model
joblib.dump(best_gb_model, 'gradient_boosting_model.pkl')

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')




