import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())


print("\nMissing Values in Train:")
print(train_df.isnull().sum())
print("\nMissing Values in Test:")
print(test_df.isnull().sum())


plt.figure(figsize=(8, 5))
sns.countplot(x='rainfall', data=train_df)
plt.title('Target Variable Distribution')
plt.show()


plt.figure(figsize=(12, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)



label_encoders = {}
for col in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le


# Splitting data
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_df)



# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print("Model Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))



test_predictions = model.predict_proba(X_test)[:, 1]



submission = pd.read_csv(sample_submission_path)
submission['rainfall'] = test_predictions
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")




