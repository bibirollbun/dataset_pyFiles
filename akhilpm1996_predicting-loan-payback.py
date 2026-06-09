import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


train.head()


train.drop('id',axis=1,inplace=True)


train.info()


train.isna().sum()


train.describe()


train.loan_paid_back.value_counts(normalize=True)


# Frequency encoding
freq = train['grade_subgrade'].value_counts()
train['grade_subgrade_freq'] = train['grade_subgrade'].map(freq)
train.drop('grade_subgrade',axis=1,inplace=True)


X=train.drop('loan_paid_back',axis=1)
y=train['loan_paid_back']


# Train test Splits
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


cat_col = X_train.select_dtypes(include='object').columns

# Initialize label encoder
le = LabelEncoder()

encoders = {}
for col in cat_col:                     
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    encoders[col] = le


# Apply label encoding to each column
for col in cat_col:
    le = encoders[col]
    X_test[col] = le.transform(X_test[col])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


model = RandomForestClassifier(class_weight = "balanced")
model.fit(X_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
predsRf = model.predict(X_test_scaled)
cmlf = confusion_matrix(y_test, predsRf)

print("Accuracy:", accuracy_score(y_test, predsRf))
print("F1 Score:", f1_score(y_test, predsRf, average='weighted'))

plt.figure(figsize=(8,5))
sns.heatmap(cmlf,annot=True,fmt=".2f")
plt.tight_layout()
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

IDS = test['id']
test.drop('id',axis=1,inplace=True)

test['grade_subgrade_freq'] = test['grade_subgrade'].map(freq)
test.drop('grade_subgrade',axis=1,inplace=True)

# Apply label encoding to each column
for col in cat_col:
    le = encoders[col]
    test[col] = le.transform(test[col])

test_scaled = scaler.transform(test)


y_pred = model.predict(test_scaled)
subm = pd.DataFrame({'id': IDS,'loan_paid_back':y_pred})
subm.to_csv('submission.csv',index=False)

