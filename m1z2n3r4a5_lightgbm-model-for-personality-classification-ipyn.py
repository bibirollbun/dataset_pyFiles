# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality']) 


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])
    X_test[col] = le_col.transform(X_test[col])  # Ensure same mapping


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


from lightgbm import LGBMClassifier
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))


test_preds = model.predict(X_test)
test_preds_labels = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds_labels
})
submission.to_csv('submission.csv', index=False)


# Get predictions
test_preds = model.predict(X_test)
test_preds_labels = le.inverse_transform(test_preds)  # Get "Introvert"/"Extrovert"

# âœ… Final clean submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds_labels
})

# âœ… Save exactly as Kaggle expects
submission.to_csv('submission.csv', index=False, encoding='utf-8')



print(submission.head())
print(submission.dtypes)
print(submission.isnull().sum())



import os

# Check if file exists
file_path = "submission.csv"
if os.path.exists(file_path):
    print("âœ… Submission file created successfully.")
    
    # Preview content
    df = pd.read_csv(file_path)
    print("ğŸ”� First 5 rows:\n", df.head())
    print("\nğŸ“� Shape:", df.shape)
    print("\nğŸ“‹ Columns:", df.columns.tolist())
else:
    print("â�Œ Submission file NOT found. Please check your save path.")





