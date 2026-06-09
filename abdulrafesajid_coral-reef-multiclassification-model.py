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

train = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/train.csv')
test = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/test.csv')
sample_submission = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission sample:")




X = train.drop(columns=[col for col in train.columns if col.startswith("species_")] + ['region', 'id'])
y = train[[col for col in train.columns if col.startswith("species_")]]



# One-hot encode both train and test
combined = pd.concat([X, test.drop(columns=['id', 'region'])], axis=0)

combined = pd.get_dummies(combined, columns=['substrate_type', 'light_availability', 'marine_protection_status'], drop_first=True)

# Split back
X = combined.iloc[:len(X)]
X_test = combined.iloc[len(X):]

# Scale numerical features (optional)
from sklearn.preprocessing import StandardScaler
numeric_cols = ['depth', 'structural_complexity', 'water_temperature', 'salinity', 'coral_cover', 'algal_cover', 'proximity_to_human_activity']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])



X.isnull().any().sum


# Combine X and y to drop rows together
full = X.copy()
full[y.columns] = y

# Drop rows with any NaN
full = full.dropna()

# Separate again
X = full.drop(columns=y.columns)
y = full[y.columns]



from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, y)



from sklearn.impute import SimpleImputer
import pandas as pd

# Step 1: Impute missing values in X_test
imputer = SimpleImputer(strategy='mean')
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

# Step 2: Predict using the trained model
y_pred = model.predict(X_test_imputed)

# Step 3: Convert predictions to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)


submission = pd.concat([test['id'], y_pred_df], axis=1)


submission.to_csv("submission.csv", index=False)





