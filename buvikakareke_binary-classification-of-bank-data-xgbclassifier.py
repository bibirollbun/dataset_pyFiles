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

train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train_data.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_dataframe(df, target_col, drop_cols=None):
    """
    Preprocess a dataframe by:
    - Dropping ID or unwanted columns
    - Encoding binary, categorical, and numerical features
    - Returning X (features) and y (target)
    """
    df = df.copy()

    # Drop ID or unnecessary columns
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Prepare for encoded output
    X_encoded = pd.DataFrame()

    for col in X.columns:
        if X[col].nunique() == 2:
            # Binary encoding (e.g., Yes/No, Male/Female)
            X_encoded[col] = LabelEncoder().fit_transform(X[col])
        
        elif X[col].dtype == 'object':
            # One-hot encode categorical text features
            dummies = pd.get_dummies(X[col], prefix=col)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
        
        else:
            # Leave numeric features as-is
            X_encoded[col] = X[col]

    return X_encoded, y



X_encoded, y = preprocess_dataframe(train_data, 'y', ['id'])


X_encoded


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(X_train, y_train)

# Step 3: Get feature importances from training
importances = pd.Series(model.feature_importances_, index=X_train.columns)

# Step 4: Drop low-importance features
low_impact_features = importances[importances < 0.01].index

X_train_reduced = X_train.drop(columns=low_impact_features)
X_test_reduced = X_test.drop(columns=low_impact_features)

# Step 5: Retrain on reduced features
model = XGBClassifier()
model.fit(X_train_reduced, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_reduced)
print("Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.metrics import roc_auc_score

# For binary classification, get predicted probabilities for the positive class
y_proba = model.predict_proba(X_test_reduced)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", roc_auc)


def preprocess_test_data(test_df, train_columns, drop_cols=None):
    # Drop ID and other drop_cols temporarily for processing
    id_col = test_df['id'] if 'id' in test_df.columns else None

    if drop_cols:
        test_df = test_df.drop(columns=drop_cols, errors='ignore')

    # One-hot encode or label encode
    X_encoded = pd.DataFrame()

    for col in test_df.columns:
        if test_df[col].nunique() == 2:
            X_encoded[col] = LabelEncoder().fit_transform(test_df[col])
        elif test_df[col].dtype == 'object':
            dummies = pd.get_dummies(test_df[col], prefix=col)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
        else:
            X_encoded[col] = test_df[col]

    # Align test columns with training columns
    X_encoded = X_encoded.reindex(columns=train_columns, fill_value=0)

    # Restore ID column
    if id_col is not None:
        X_encoded.insert(0, 'id', id_col)

    return X_encoded



# Load test CSV
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Apply same preprocessing
X_test_full = preprocess_test_data(test_df, train_columns=X_train.columns.tolist(), drop_cols=['ID'])

# Drop low-importance features
X_test_reduced = X_test_full.drop(columns=low_impact_features, errors='ignore')

# Keep ID
id_column = X_test_reduced['id']
X_test_final = X_test_reduced.drop(columns=['id'])

# Predict probabilities
y_pred_proba = model.predict_proba(X_test_final)[:, 1]  # Probability of class 1

# Output
submission_df = pd.DataFrame({
    'id': id_column,
    'y': y_pred_proba
})

submission_df.to_csv("predictions.csv", index=False)



predictions = pd.read_csv("/kaggle/working/predictions.csv")
predictions.head()

