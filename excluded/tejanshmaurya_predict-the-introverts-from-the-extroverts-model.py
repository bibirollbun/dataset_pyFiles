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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train_df.head(10)


train_df.isnull().sum()


X = train_df.drop(columns=['id', 'Personality'])
y = train_df['Personality']


y


categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))


imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)





# Prepare test set
X_test = test.drop(columns=['id'])
for col in categorical_cols:
    X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)





# Predict on test set
preds = model.predict(X_test)
pred_labels = target_encoder.inverse_transform(preds)





# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': pred_labels
})
submission.to_csv("submission.csv", index=False)







