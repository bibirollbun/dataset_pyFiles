import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")


df.head(10)


df.isnull().sum()


df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())


df.isnull().sum()


label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])


X = df.drop(columns=['satisfaction'])
y = df['satisfaction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)

y_pred = gboost.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


print(df['satisfaction'].value_counts(normalize=True))


print(df.describe())
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='satisfaction', data=df)
plt.show()


from sklearn.ensemble import GradientBoostingClassifier
gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)


from sklearn.metrics import classification_report, accuracy_score
y_pred = gboost.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import numpy as np

importances = gboost.feature_importances_
indices = np.argsort(importances)
features = X_train.columns

plt.figure(figsize=(10,6))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


import shap

explainer = shap.Explainer(gboost, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = gboost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Define categorical columns (excluding 'satisfaction')
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Create a dictionary to store encoders for each column
encoders = {}

# Apply label encoding using training data and store encoders
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # df is your train dataset
    encoders[col] = le

# Apply the same encoders to the test dataset
for col in label_cols:
    test_df[col] = encoders[col].transform(test_df[col])



# Fill missing values like training
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())

# Encode categorical features same as training
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in label_cols:
    test_df[col] = le.fit_transform(test_df[col])

# Match test features to training features
X_test_final = test_df[X_train.columns]  # ✅ THIS LINE FIXES YOUR ERROR

# Predict
test_preds = gboost.predict(X_test_final)

# Inverse transform predictions to original labels
test_df['satisfaction'] = le.inverse_transform(test_preds)

# Prepare submission
submission = test_df[['id', 'satisfaction']].copy()
submission.columns = ['ID', 'satisfaction']
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv saved successfully.")
print(submission.head())


