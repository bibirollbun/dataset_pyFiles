import numpy as np
import pandas as pd
import warnings
import datetime
import xgboost as xgb
import seaborn as sns
import itertools
import datetime
import sys
import os
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
%matplotlib inline
warnings.filterwarnings("ignore")





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


# Load the Training dataset for Bank Loan Dataset
MyData = pd.read_csv(r'/kaggle/input/playground-series-s5e12/train.csv') 
MyData.shape



MyData.head(10)


MyData.info()


MyData.describe()


MyData.columns


MyData.head(5)


# Approach 3: One-hot encoding
dummy_columns = ["gender","ethnicity","education_level","income_level","smoking_status","employment_status"]
MyData_ohe = pd.get_dummies(MyData, columns=dummy_columns, dtype=int)
#X_test_ohe  = pd.get_dummies(X_test, columns=dummy_columns)
MyData_ohe.head(5)


MyData_ohe.columns 


columns_rearranged = ['id', 'age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'gender_Female',
       'gender_Male', 'gender_Other', 'ethnicity_Asian', 'ethnicity_Black',
       'ethnicity_Hispanic', 'ethnicity_Other', 'ethnicity_White',
       'education_level_Graduate', 'education_level_Highschool',
       'education_level_No formal', 'education_level_Postgraduate',
       'income_level_High', 'income_level_Low', 'income_level_Lower-Middle',
       'income_level_Middle', 'income_level_Upper-Middle',
       'smoking_status_Current', 'smoking_status_Former',
       'smoking_status_Never', 'employment_status_Employed',
       'employment_status_Retired', 'employment_status_Student',
       'employment_status_Unemployed', 'diagnosed_diabetes']  # Added parentheses around the tuple

MyData_ohe = MyData_ohe[columns_rearranged]
MyData_ohe.columns
##MyData_ohe.head(5)


columns_to_select = ['id', 'age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 
       'gender_Female', 'gender_Male', 'gender_Other', 'ethnicity_Asian',
       'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Other',
       'ethnicity_White', 'education_level_Graduate',
       'education_level_Highschool', 'education_level_No formal',
       'education_level_Postgraduate', 'income_level_High', 'income_level_Low',
       'income_level_Lower-Middle', 'income_level_Middle',
       'income_level_Upper-Middle', 'smoking_status_Current',
       'smoking_status_Former', 'smoking_status_Never',
       'employment_status_Employed', 'employment_status_Retired',
       'employment_status_Student', 'employment_status_Unemployed']

X_train_ohe = MyData_ohe[columns_to_select]
y_train_ohe = MyData_ohe['diagnosed_diabetes']
X_train_ohe.head(5)


y_train_ohe.head(5)


# With random_state for reproducibility and shuffle=True (default)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_ohe, 
    y_train_ohe, 
    test_size=0.2, 
    random_state=42,  # For reproducibility
    shuffle=True       # Shuffle data before splitting (default is True)
)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print("Train:", len(X_train))  # Changed 'length' to 'len' which is the correct Python function to get length


X_train.head(5)


independent_variables = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 
       'gender_Female', 'gender_Male', 'gender_Other', 'ethnicity_Asian',
       'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Other',
       'ethnicity_White', 'education_level_Graduate',
       'education_level_Highschool', 'education_level_No formal',
       'education_level_Postgraduate', 'income_level_High', 'income_level_Low',
       'income_level_Lower-Middle', 'income_level_Middle',
       'income_level_Upper-Middle', 'smoking_status_Current',
       'smoking_status_Former', 'smoking_status_Never',
       'employment_status_Employed', 'employment_status_Retired',
       'employment_status_Student', 'employment_status_Unemployed']
X_train_fit = X_train[independent_variables]
X_val_fit   = X_val[independent_variables]
X_train_fit.head(5)


X_val_fit.head(5)


# Launch a classifier
# XGBoost Training Parameter Reference: 
#   https://xgboost.readthedocs.io/en/latest/parameter.html
classifier = xgb.XGBClassifier(objective="binary:logistic",
                               eval_metric=['logloss'],
                               enable_categorical=True,
                               early_stopping_rounds=10)


classifier


classifier.fit(X_train_fit,
               y_train, 
               eval_set = [(X_train_fit, y_train), (X_val_fit, y_val)])


eval_result = classifier.evals_result()


training_rounds = range(len(eval_result['validation_0']['logloss']))


print(training_rounds)


plt.scatter(x=training_rounds,y=eval_result['validation_0']['logloss'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['logloss'],label='Validation Error')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('LogLoss')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()


y_pred = classifier.predict(X_val_fit)


from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

# Get confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Calculate accuracy manually from confusion matrix
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    accuracy_from_cm = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy (from confusion matrix): {accuracy_from_cm:.4f}")
    print(f"Accuracy (from confusion matrix): {accuracy_from_cm*100:.2f}%")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# For multi-class problems, you might want to normalize the confusion matrix
if cm.shape[0] > 2:
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.show()


# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

# Alternative method using confusion matrix
from sklearn.metrics import confusion_matrix

# Get confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Calculate accuracy manually from confusion matrix
# For binary classification:
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    accuracy_from_cm = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy (from confusion matrix): {accuracy_from_cm:.4f}")
    print(f"Accuracy (from confusion matrix): {accuracy_from_cm*100:.2f}%")


from sklearn.metrics import roc_curve, auc, roc_auc_score

# Get predicted probabilities for the positive class
# For binary classification, we need the probability of the positive class (class 1)
# y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# Calculate AUC
roc_auc = auc(fpr, tpr)
# Alternative: roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Add some threshold annotations
# Select a few thresholds to annotate
threshold_indices = np.linspace(0, len(thresholds) - 1, 5, dtype=int)
for i in threshold_indices:
    plt.annotate(f'{thresholds[i]:.2f}', 
                 xy=(fpr[i], tpr[i]), 
                 xytext=(fpr[i]+0.05, tpr[i]-0.05),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

plt.tight_layout()
plt.show()

# Print AUC value
print(f"AUC: {roc_auc:.4f}")


xgb.plot_importance(classifier)
plt.show()


# Get feature importance as a dictionary
feature_importance = classifier.get_booster().get_score(importance_type='weight')

# Convert to a more readable format (sorted by importance)
import pandas as pd

# Create a DataFrame for better display
importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
})

# Sort by importance in descending order
importance_df = importance_df.sort_values('Importance', ascending=False)

# Display the feature importance
print(importance_df)


probabilities = classifier.predict_proba(X_val_fit)


val_probabilities_df = pd.DataFrame({
    'id': X_val['id'].reset_index(drop=True),
    # Extract the probability for class 1 (second column)
    'y': pd.Series(probabilities[:, 1]).reset_index(drop=True)  # Using column index 1 for positive class probability
    # Alternatively, if you want both probabilities:
    # 'y_class0': pd.Series(probabilities[:, 0]).reset_index(drop=True),
    # 'y_class1': pd.Series(probabilities[:, 1]).reset_index(drop=True)
})
val_probabilities_df.head(10)


# Load the test dataset for Bank Dataset
MyData_test = pd.read_csv(r'/kaggle/input/playground-series-s5e12/test.csv') 
MyData_test.shape


MyData_test.info()


MyData_test.describe()


MyData_test.columns



# Approach 3: One-hot encoding
dummy_columns = ["gender","ethnicity","education_level","income_level","smoking_status","employment_status"]
MyData_test_ohe = pd.get_dummies(MyData_test, columns=dummy_columns, dtype=int)
#X_test_ohe  = pd.get_dummies(X_test, columns=dummy_columns)
MyData_test_ohe.head(5)



MyData_test_ohe.info()


MyData_test_ohe.head(5)


X_test_ohe_fit = MyData_test_ohe[independent_variables]
X_test_ohe_fit.head(5)


y_pred_test = classifier.predict(X_test_ohe_fit)


submission_df = pd.DataFrame({
    'id': MyData_test_ohe['id'].reset_index(drop=True),
    'y': pd.Series(y_pred_test).reset_index(drop=True)
})


submission_df.head(5)


# Count distinct values in column Y
distinct_count = submission_df['y'].value_counts()
print(distinct_count)

# If you just want the number of distinct values
num_distinct = submission_df['y'].nunique()
print(f"Number of distinct values: {num_distinct}")


test_probabilities = classifier.predict_proba(X_test_ohe_fit)



submission_probabilities_df = pd.DataFrame({
    'id': MyData_test_ohe['id'].reset_index(drop=True),
    # Extract the probability for class 1 (second column)
    'diagnosed_diabetes': pd.Series(test_probabilities[:, 1]).reset_index(drop=True)  # Using column index 1 for positive class probability
    # Alternatively, if you want both probabilities:
    # 'y_class0': pd.Series(probabilities[:, 0]).reset_index(drop=True),
    # 'y_class1': pd.Series(probabilities[:, 1]).reset_index(drop=True)
})
submission_probabilities_df.head(10)


MyData_test_ohe.describe()


submission_probabilities_df.describe()


submission_probabilities_df.shape


#submission_probabilities_df.to_csv(r'..\..\results\submission.csv', index=False)
#@print("CSV file has been created successfully!")


# Define the threshold as a variable
threshold = 0.5

# Filter the DataFrame to show only rows where y >= threshold
filtered_df = submission_probabilities_df[submission_probabilities_df['diagnosed_diabetes'] >= threshold]

# Display all filtered rows
print(f"Rows where diagnosed_diabetes >= {threshold}:")
display(filtered_df)

# Count the total number of rows where y >= threshold
count = len(filtered_df)
print(f"\nTotal number of rows where diagnosed_diabetes >= {threshold}: {count}")

# Optional: Calculate the percentage of rows that meet the condition
percentage = (count / len(submission_probabilities_df)) * 100
print(f"Percentage of rows where diagnosed_diabetes >= {threshold}: {percentage:.2f}%")







