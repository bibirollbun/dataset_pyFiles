import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



import pandas as pd

train = pd.read_csv('/kaggle/input/diabetes-dataset/train.csv')
test = pd.read_csv('/kaggle/input/diabetes-dataset/test.csv')



print(train.head())
print(train.info())


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
                    'smoking_status', 'employment_status', 'family_history_diabetes',
                    'hypertension_history', 'cardiovascular_history']

encoder = LabelEncoder()

for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))



import pandas as pd

train = pd.read_csv(r"/kaggle/input/diabetes-dataset/train.csv")
test = pd.read_csv(r"/kaggle/input/diabetes-dataset/test.csv")
sample_submission = pd.read_csv(r"/kaggle/input/diabetes-dataset/sample_submission.csv")

print(train.shape)
print(train.head())




from sklearn.preprocessing import LabelEncoder

categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
                    'smoking_status', 'employment_status', 'family_history_diabetes',
                    'hypertension_history', 'cardiovascular_history']

encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))



X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
test_scaled = scaler.transform(test.drop('id', axis=1))

model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_valid_scaled)

print("Validation Accuracy:", accuracy_score(y_valid, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))



print(y.value_counts(normalize=True))




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight='balanced'   # ðŸ‘ˆ important fix
)
model.fit(X_train_scaled, y_train)



from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

test_predictions = log_reg.predict_proba(test_scaled)[:, 1]



import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(test_predictions, kde=True, bins=20, color="teal")
plt.title("Predicted Probability Distribution (Diabetes)")
plt.xlabel("Predicted Probability of Being Diabetic")
plt.show()



submission = sample_submission.copy()
submission['diagnosed_diabetes'] = test_predictions

# Save to working directory
submission.to_csv('/kaggle/working/submission.csv', index=False)

print(submission.head())
print(submission.describe())



import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(test_predictions, kde=True, bins=20, color="teal")
plt.title("Predicted Probability Distribution (Diabetes)")
plt.xlabel("Predicted Probability of Being Diabetic")
plt.show()





