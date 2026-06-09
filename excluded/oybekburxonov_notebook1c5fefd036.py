# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, classification_report

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
print(" Ma'lumotlar yuklandi.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
display(train_df.head())



train_df = train_df[train_df['Status'].isin(['C', 'CL', 'D'])].copy()
print("\n Faqat C, CL, D sinfli ma'lumotlar qoldirildi.")
print(train_df['Status'].value_counts())



label_encoder = LabelEncoder()
train_df['Status_encoded'] = label_encoder.fit_transform(train_df['Status'])
print("\n LabelEncoder natijasi:")
print(train_df[['Status', 'Status_encoded']].drop_duplicates())


X = train_df.drop(columns=['id', 'Status', 'Status_encoded'])
y = train_df['Status_encoded']
X_test_raw = test_df.drop(columns=['id'])
test_ids = test_df['id']


categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n Sonli ustunlar:", numerical_cols)
print(" Kategorial ustunlar:", categorical_cols)


plt.figure(figsize=(6,4))
sns.countplot(x='Status', data=train_df)
plt.title("Status sinflari taqsimoti")
plt.show()


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Ma'lumotlar ajratildi:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)


model_pipeline.fit(X_train, y_train)



y_val_probs = model_pipeline.predict_proba(X_val)
val_logloss = log_loss(y_val, y_val_probs, labels=[0,1,2])
print(f" Validation log_loss: {val_logloss:.5f}")


y_val_preds = model_pipeline.predict(X_val)
print("\nðŸ“‹ Validation natijasi (Classification report):")
print(classification_report(y_val, y_val_preds, target_names=label_encoder.classes_))


test_probs = model_pipeline.predict_proba(X_test_raw)


submission_df = pd.DataFrame({
    'id': test_ids,
    'Status_C': test_probs[:, label_encoder.transform(['C'])[0]],
    'Status_CL': test_probs[:, label_encoder.transform(['CL'])[0]],
    'Status_D': test_probs[:, label_encoder.transform(['D'])[0]]
})
display(submission_df.head())



submission_df.to_csv('submission.csv', index=False)


