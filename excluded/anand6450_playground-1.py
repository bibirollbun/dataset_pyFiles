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


# ðŸ“˜ Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ðŸ“˜ Step 2: Load the dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# ðŸ“˜ Step 3: Visualize dataset
print("\nTrain Data Info:\n")
print(train.info())

# Plot distribution of target
plt.figure(figsize=(10,5))
sns.countplot(data=train, y='Fertilizer Name', order=train['Fertilizer Name'].value_counts().index)
plt.title('Distribution of Fertilizer Names')
plt.tight_layout()
plt.show()

# Correlation heatmap (numerical features)
plt.figure(figsize=(12, 8))
sns.heatmap(train.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# ðŸ“˜ Step 4: Preprocess
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

# Label encode categorical features
cat_cols = X.select_dtypes(include='object').columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ðŸ“˜ Step 5: Train the model
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ðŸ“˜ Step 6: Feature importance visualization
importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feat_names[indices])
plt.title('Feature Importances (Random Forest)')
plt.show()

# ðŸ“˜ Step 7: Predict on validation and print accuracy
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# ðŸ“˜ Step 8: Predict probabilities on test
probs = model.predict_proba(X_test)

# Get top 3 predictions
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Top 3 indices per row

# Map back to label names
top_3_labels = target_encoder.inverse_transform(top_3.ravel()).reshape(top_3.shape)
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top_3_labels]
})




submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")

