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


# Bank Customer Churn Prediction - Kaggle Competition

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
train = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-challenge/sample_submission.csv')

# 2. Column List
print("Columns in dataset:", train.columns.tolist())

# 3. Rename Target Column for Consistency
train.rename(columns={'Exited': 'Churn'}, inplace=True)
test_ids = test['id']

# 4. Visualizations with Plotly
fig = px.pie(train, names='Churn', title='Churn Distribution', hole=0.4)
fig.show()

numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
for col in numeric_cols:
    fig = px.histogram(train, x=col, color='Churn', barmode='overlay', title=f'{col} Distribution by Churn')
    fig.show()

cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
for col in cat_cols:
    fig = px.histogram(train, x=col, color='Churn', barmode='group', title=f'{col} vs Churn')
    fig.show()

fig = px.imshow(train.select_dtypes(include=np.number).corr(), text_auto=True, title='Correlation Heatmap')
fig.show()

# 5. Preprocessing
X = train.drop(['id', 'CustomerId', 'Surname', 'Churn'], axis=1)
y = train['Churn']
X_test = test.drop(['id', 'CustomerId', 'Surname'], axis=1)

for col in ['Gender', 'Geography']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. Model Training
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_val_pred))
print("ROC AUC:", roc_auc_score(y_val, y_val_proba))

# 8. Final Predictions
final_preds = model.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({
    'id': test_ids,
    'Exited': final_preds
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved.")


