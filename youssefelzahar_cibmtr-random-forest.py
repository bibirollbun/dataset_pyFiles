#pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
#from lifelines.utils import concordance_index



# Load the datasets
train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Display dataset structure
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
train_data.head()



# Check for missing values
print(train_data.isnull().sum())

# Fill or drop missing values
train_data.fillna(-999, inplace=True)

# Encode categorical variables if necessary
train_data = pd.get_dummies(train_data, drop_first=True)



# Split features and target
X = train_data.drop(['efs','efs_time'], axis=1)
y = train_data['efs']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize a simple Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]



# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_val, y_prob))
print("Classification Report:\n", classification_report(y_val, y_pred))



unwanted_columns = ['efs_time', 'efs']  
train_data = train_data.drop(columns=unwanted_columns, errors='ignore')

train_data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)

train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

test_data_aligned = test_data_encoded.reindex(columns=train_data_encoded.columns, fill_value=0)

test_predictions = model.predict(test_data_aligned)

print(test_predictions[:10]) 



# Create a submission file
submission = pd.DataFrame({'ID': test_data['ID'], 'prediction': test_predictions})
submission.to_csv('submission.csv', index=False)



submission

