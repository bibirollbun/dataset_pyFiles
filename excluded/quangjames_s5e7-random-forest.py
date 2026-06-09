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


import pandas as pd
import numpy as np
from IPython.display import display, HTML

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


train_file = '/kaggle/input/playground-series-s5e7/train.csv'
test_file = '/kaggle/input/playground-series-s5e7/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print("-----"*10 + "Overview Train Dataset" + "------"*10)
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Information</span>"))
display(train_data.info())
print("-----"*10 + "Overview Test Dataset" + "------"*10)
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Information</span>"))
display(test_data.info())


print(train_data['Personality'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_data, palette='Set2')
plt.title('Personality Distribution on Train set', fontsize=14, pad=15)
plt.xlabel('Personality', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.savefig('personality_distribution.png')
plt.show()


# Extract features and labels
X = train_data.drop(columns=['id', 'Personality'])
y = train_data['Personality']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# Encode the sorting columns
le = LabelEncoder()
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    # Encode on the train set
    X[col] = le.fit_transform(X[col].astype(str))
    # Encode on the test set, handle unseen values by adding 'unknown'
    test_data[col] = test_data[col].astype(str).map(lambda x: x if x in le.classes_ else 'unknown')
    # Add 'unknown' to LabelEncoder's classes
    if 'unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'unknown')
    X_test[col] = le.transform(test_data[col])

# Personality Label Encoding
y = le.fit_transform(y)

# Handling missing values
# For numeric columns: fill in the average value
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
imputer = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# Split train/test data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

display(HTML("<span style = 'color: red; font-weight:bold;'> Data preprocessing is completed</span>"))
print("Shape of training set:", X_train.shape)
print("Shape of validation set:", X_val.shape)
print("Featues of training:", X_train.columns.tolist())


# Fine-tune Random Forest model with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Use with best model
best_rf_model = grid_search.best_estimator_


# Model evaluation on validation set
y_val_pred = best_rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")


# Prediction on test set
y_test_pred = best_rf_model.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred)

# Create a submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': y_test_pred
})
submission.to_csv('submission.csv', index=False)
display(HTML("<span style = 'color: blue; font-weight:bold;'> File submission.csv was created!</span>"))

# Submission file checking
print(submission.head())

