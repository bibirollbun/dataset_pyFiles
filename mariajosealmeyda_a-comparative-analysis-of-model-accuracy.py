# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import StackingClassifier


# Load the necessary data
train_data = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


# Summary and preview of the data 
def resumetable(df):
    summary = df.describe(include='all').transpose()
    summary['missing_values'] = df.isnull().sum()  
    summary['unique_values'] = df.nunique()       
    return summary
train_summary = resumetable(train_data)
print("train_data summary", train_summary)

test_summary = resumetable(test_data)
print("train_data summary", train_summary)



# Create a boxplot to verify the distribution of the data
variables = ['person_age', 'person_emp_length']
for var in variables: 
    plt.figure(figsize=(8, 6))  
    sns.boxplot(data=train_data, x=var)  
    plt.title(f'Boxplot of {var}', fontsize=16)
    plt.show()


# Filter the DataFrame to keep only the records where person_age < 100
train_data = train_data[train_data['person_age'] < 100]
train_data = train_data[train_data['person_emp_length'] < 70]

# Verify the data after filtering
print(train_data['person_age'].describe())
print(train_data['person_emp_length'].describe())
print(f"Number of records after filtering: {len(train_data)}")



columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

def encode_categorical_columns(dataframe, columns):
    for col in columns:
        dataframe[col] = pd.Categorical(dataframe[col]).codes  
    return dataframe


train_data_2 = encode_categorical_columns(train_data, columns)
test_data_2 = encode_categorical_columns(test_data, columns)


# Categories for income 
def categorize_income(income):
    if income <= 50000:
        return "Low Income"
    elif 50000 < income <= 500000:
        return "Middle Income"
    else:
        return "High Income"

# Apply the function
train_data_2["income_category"] = train_data_2["person_income"].apply(categorize_income)
test_data_2["income_category"] = test_data_2["person_income"].apply(categorize_income)

columns= ['income_category']
def encode_categorical_columns(dataframe, columns):
    for col in columns:
        dataframe[col] = pd.Categorical(dataframe[col]).codes  
    return dataframe


train_data_2 = encode_categorical_columns(train_data_2, columns)
test_data_2 = encode_categorical_columns(test_data_2, columns)


print(train_data_2[["person_income", "income_category"]].head())


# Combine train_data_2 and test_data_2
combined_data = pd.concat([train_data_2, test_data_2], axis=0, ignore_index=True)

def calculate_probabilities(data):
    # Calculate the probability of positive loan_status for each loan_grade
    probabilities = data.groupby('loan_grade')['loan_status'].mean().reset_index()
    probabilities.rename(columns={'loan_status': 'probability_positive'}, inplace=True)
    return probabilities

loan_grade_probabilities = calculate_probabilities(combined_data)
combined_data = combined_data.merge(loan_grade_probabilities, on='loan_grade', how='left')

# Split into train and test again
train_v3 = combined_data[~combined_data['loan_status'].isna()].copy()
test_v3= combined_data[combined_data['loan_status'].isna()].copy()

print(train_v3.head())



Correlation = train_v3.corr()  
correlation_target = Correlation['loan_status'].sort_values(ascending=False)

# Create the heatmap
plt.figure(figsize=(10, 8))  
sns.heatmap(Correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

print(correlation_target)


# Data Setup
train_df = train_v3
test_df = test_v3
target_column = 'loan_status'
drop_columns = ['loan_status']

# Prepare the target variable and split into training and test sets
X = train_df.drop(columns=drop_columns)
y = train_df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configure and train the KNN model
k = 100  
knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', p=2)
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test_scaled)  
y_pred_prob = knn_model.predict_proba(X_test_scaled)[:, 1]  

# Model evaluation
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.2f}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Make predictions on the test set (new test_df)
X_test_new_scaled = scaler.transform(test_df[X_train.columns])  
test_predictions_prob = knn_model.predict_proba(X_test_new_scaled)[:, 1]
test_predictions = knn_model.predict(X_test_new_scaled)


output_test = pd.DataFrame({
    'Predicted Probabilities': test_predictions_prob,
    'Predicted Values': test_predictions
})
print("\n--- Test Set Predictions ---")
print(output_test.head())

# Save predictions to CSV files
output_test.to_csv("test_knn_predictions.csv", index=False)
print("Test set predictions saved in test_knn_predictions.csv")



# Optimal hyperparameters selection with Random Search
# Base model of LightGBM
lightgbm_model = LGBMClassifier(
    random_state=42,
    verbose=-1  
)

# Hyperparameter search space
param_distributions = {
    'n_estimators': [500, 1000, 2000],  
    'learning_rate': [0.01, 0.05, 0.1],  
    'max_depth': [4, 6, 8, 10],  
    'num_leaves': [20, 31, 50, 100],  
    'min_child_samples': [10, 20, 30],  
    'reg_alpha': [0, 0.1, 1, 10],  
    'reg_lambda': [0, 0.1, 1, 10],  
    'subsample': [0.6, 0.8, 1.0],  
    'colsample_bytree': [0.6, 0.8, 1.0]  
}

# RandomizedSearchCV for LightGBM
random_search = RandomizedSearchCV(
    estimator=lightgbm_model,
    param_distributions=param_distributions,
    n_iter=10,  
    scoring='neg_mean_squared_error',  
    cv=3,  
    random_state=42,
    verbose=1  
)

# Fit the model
random_search.fit(X_train, y_train)

print("Best hyperparameters:", random_search.best_params_)


# LightGBM Model Configuration (Classification)
lightgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    verbose=-1,
    subsample=0.8,
    reg_lambda=10, 
    reg_alpha=10, 
    num_leaves=50, 
    min_child_samples=10, 
    colsample_bytree=0.8
)

# Train the model on the training set
lightgbm_model.fit(X_train, y_train)

# Predictions (classes) and probabilities on the test set
y_pred = lightgbm_model.predict(X_test)  
y_pred_prob = lightgbm_model.predict_proba(X_test)[:, 1]  

# Model evaluation
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.2f}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Predictions on the test set
X_test_new = test_df[X_train.columns]  
test_predictions_prob = lightgbm_model.predict_proba(X_test_new)[:, 1]
test_predictions = lightgbm_model.predict(X_test_new)

output_test = pd.DataFrame({
    'Predicted Probabilities': test_predictions_prob,
    'Predicted Values': test_predictions
})

print("\n--- Test Set Predictions ---")
print(output_test.head())


# Save predictions to CSV files
output_test.to_csv("test_lightgbm_predictions.csv", index=False)
print("Test set predictions saved in test_lightgbm_predictions.csv")



# Define the baseline model
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)

# Define the range of hyperparameters
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 1, 10],
    'reg_lambda': [0, 1, 10]
}

# Configure Random Search
random_search = RandomizedSearchCV(
    estimator=xgboost_model,
    param_distributions=param_distributions,
    n_iter=50,          
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)


print("Best Hyperparameters:", random_search.best_params_)
print("Best AUC-ROC Score:", random_search.best_score_)


# Create the XGBoost model
xgboost_model = xgb.XGBClassifier(  
    n_estimators=300,         
    learning_rate=0.05,        
    max_depth=6,              
    subsample=1.0,            
    colsample_bytree=0.6,     
    random_state=42,          
    reg_alpha=0,             
    reg_lambda=1,            
    use_label_encoder=False,  
    eval_metric='auc'         
)

# Train the model and make predictions
xgboost_model.fit(X_train, y_train)
y_pred = xgboost_model.predict(X_test)  
y_pred_prob = xgboost_model.predict_proba(X_test)[:, 1]  

# Model evaluation
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.2f}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Predictions on the test set
X_test_new = test_df[X_train.columns]  
test_predictions_prob = xgboost_model.predict_proba(X_test_new)[:, 1]
test_predictions = xgboost_model.predict(X_test_new)

output_test = pd.DataFrame({
    'Predicted Probabilities': test_predictions_prob,
    'Predicted Values': test_predictions
})

print("\n--- Test Set Predictions ---")
print(output_test.head())

# Save the predictions
output_test.to_csv("xgboost_test_predictions.csv", index=False)
print("Test predictions saved in xgboost_test_predictions.csv")



# XGBoost Model Configuration
xgboost_model = xgb.XGBClassifier(  
    n_estimators=300,         
    learning_rate=0.05,        
    max_depth=6,              
    subsample=1.0,            
    colsample_bytree=0.6,     
    random_state=42,          
    reg_alpha=0,             
    reg_lambda=1,            
    use_label_encoder=False,  
    eval_metric='auc'         
)

# LightGBM Model Configuration
lightgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    verbose=-1,
    subsample=0.8,
    reg_lambda=10, 
    reg_alpha=10, 
    num_leaves=50, 
    min_child_samples=10, 
    colsample_bytree=0.8
)

# Meta-Model
meta_model = KNeighborsClassifier(n_neighbors=100, weights='uniform', metric='minkowski', p=2)

# Create the Stacking Model
stacking_model = StackingClassifier(
    estimators=[
        ('xgboost', xgboost_model),
        ('lightgbm', lightgbm_model)
    ],
    final_estimator=meta_model,
    cv=5, 
    n_jobs=-1
)

# Train the Stacking Model
stacking_model.fit(X_train, y_train)

# Predictions with the Stacking Model
y_pred = stacking_model.predict(X_test)
y_pred_prob = stacking_model.predict_proba(X_test)[:, 1]

# Calculate Evaluation Metrics
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc_score:.2f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Predictions on the Test Set
X_test_new = test_df[X_train.columns]  
test_predictions_prob = stacking_model.predict_proba(X_test_new)[:, 1]  
test_predictions = stacking_model.predict(X_test_new)  

output_test = pd.DataFrame({
    'Predicted Probabilities': test_predictions_prob,
    'Predicted Values': test_predictions
})

# Save Predictions to a CSV File
output_test.to_csv("test_stacking_predictions.csv", index=False)

print("\n--- Test predictions saved ---")
print("test_stacking_predictions.csv")



# Load test set predictions
test_predictions = pd.read_csv("test_lightgbm_predictions.csv")
print("\nTest set predictions:")
print(test_predictions.head())



# Create the submission DataFrame with the columns 'id' and 'sales'
submission = test_data[['id']].copy()  

# Assign predictions to the 'submission' DataFrame
submission['loan_status'] = test_predictions['Predicted Probabilities'].values  
print(submission.head())

# Save the CSV file
submission.to_csv('/kaggle/working/final_submission.csv', index=False)
print("Predictions file created.")

