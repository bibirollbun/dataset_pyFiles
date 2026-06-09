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


# Step 1: Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Step 3: Prepare Data
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)

# ğŸ§  Step 4: One-Hot Encoding (Convert Text to Numbers)
# Step 4.1: Handle Missing Values (NaNs)
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# ğŸ”§ Make sure columns in test match train
X_test = X_test.reindex(columns = X.columns, fill_value=0)

# ğŸ�·ï¸� Encode labels (target classes)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 5: Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Validate Model
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Step 8: Predict on Test
test_preds = model.predict(X_test)
test_preds_labels = le.inverse_transform(test_preds)

# Step 9: Create Submission
submission = sample_submission.copy()
submission['Personality'] = test_preds_labels
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created successfully!")


