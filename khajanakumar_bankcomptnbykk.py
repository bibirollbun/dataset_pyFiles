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


# Load the dataset for Bank Dataset
MyData = pd.read_csv(r'/kaggle/input/playground-series-s5e8/train.csv') 
MyData.shape


MyData.head(7)


MyData.info()


MyData.columns


# First, let's check the actual column names in the DataFrame
print(MyData.columns)

# Then use the correct column names
# Option 1: If column names are the same but with different capitalization or spacing
My_data_columns = ['id', 'age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome']
independent_variables = ['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome']  # Adjust based on actual column names

# Option 2: If columns have completely different names, replace with actual column names
# independent_variables = ["actual_column_1", "actual_column_2", ...]

# Correct syntax for selecting multiple columns - use double square brackets
X = MyData[My_data_columns]
       
y = MyData["y"]  # Adjust if "y" is named differently

# Alternative approach using the independent_variables list
# X = MyData[independent_variables]
# y = MyData["y"]


X.head(5)


X.drop(['day', 'month'], axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print("Train:", len(X_train))  # Changed 'length' to 'len' which is the correct Python function to get length


X_train.head(5)


# Approach 2: Label encoding
X_train_le = X_train.copy()
X_val_le = X_val.copy()
X_train_le = X_train_le[independent_variables]
X_val_le   = X_val_le[independent_variables]
le_dict = {}
for col in ['job', 'marital','education','default','housing','loan','contact','poutcome']:
    le = LabelEncoder()
    X_train_le[col] = le.fit_transform(X_train[col])
    X_val_le[col] = le.transform(X_val[col])
    le_dict[col] = le

# Train with label encoded features
model_le = xgb.XGBClassifier()
#model_le = xgb.XGBClassifier(objective="binary:logistic",
#                               eval_metric=['logloss'],
#                               early_stopping_rounds=10)
model_le.fit(X_train_le, y_train)


y_pred = model_le.predict(X_val_le)
len(model_le.predict(X_val_le))


y_pred_df = pd.DataFrame({
    'id': X_val['id'].reset_index(drop=True),
    'y': pd.Series(y_pred).reset_index(drop=True)
})


y_pred_df.head(5)


# Count distinct values in column Y
distinct_count = y_pred_df['y'].value_counts()
print(distinct_count)

# If you just want the number of distinct values
num_distinct = y_pred_df['y'].nunique()
print(f"Number of distinct values: {num_distinct}")


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


xgb.plot_importance(model_le)
plt.show()


probabilities = model_le.predict_proba(X_val_le)


val_probabilities_df = pd.DataFrame({
    'id': X_train['id'].reset_index(drop=True),
    # Extract the probability for class 1 (second column)
    'y': pd.Series(probabilities[:, 1]).reset_index(drop=True)  # Using column index 1 for positive class probability
    # Alternatively, if you want both probabilities:
    # 'y_class0': pd.Series(probabilities[:, 0]).reset_index(drop=True),
    # 'y_class1': pd.Series(probabilities[:, 1]).reset_index(drop=True)
})
val_probabilities_df.head(10)


# Load the test dataset for Bank Dataset
MyData_test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e8/test.csv') 
MyData_test_df.shape


MyData_test_df.info()


MyData_test_df.describe


X_test = MyData_test_df[My_data_columns]
X_test.head(5)  


X_test_le = X_test.copy()
X_test_le = X_test_le[independent_variables]
le_dict = {}

# Create and fit a separate label encoder for each categorical column
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
    # Create a new label encoder for each column
    le_dict[col] = LabelEncoder()
    
    # Combine training and test data values to fit the encoder on all possible values
    # This ensures the encoder knows about all categories
    all_values = pd.concat([X_train[col], X_test[col]]).unique()
    le_dict[col].fit(all_values)
    
    # Transform the test data using the properly fitted encoder
    X_test_le[col] = le_dict[col].transform(X_test[col])


X_test_le.head(5)


y_test = model_le.predict(X_test_le)


submission_df = pd.DataFrame({
    'id': X_test['id'].reset_index(drop=True),
    'y': pd.Series(y_test).reset_index(drop=True)
})


submission_df.head(5)


# Count distinct values in column Y
distinct_count = submission_df['y'].value_counts()
print(distinct_count)

# If you just want the number of distinct values
num_distinct = submission_df['y'].nunique()
print(f"Number of distinct values: {num_distinct}")


test_probabilities = model_le.predict_proba(X_test_le)


submission_probabilities_df = pd.DataFrame({
    'id': X_test['id'].reset_index(drop=True),
    # Extract the probability for class 1 (second column)
    'y': pd.Series(test_probabilities[:, 1]).reset_index(drop=True)  # Using column index 1 for positive class probability
    # Alternatively, if you want both probabilities:
    # 'y_class0': pd.Series(probabilities[:, 0]).reset_index(drop=True),
    # 'y_class1': pd.Series(probabilities[:, 1]).reset_index(drop=True)
})
submission_probabilities_df.head(10)

