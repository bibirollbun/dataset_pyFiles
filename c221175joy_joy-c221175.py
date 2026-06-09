import pandas as pd
import numpy as np


train = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")


train


test = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")


test


train.drop(columns=['Unnamed: 0'], inplace=True)


'Unnamed: 0' in train.columns



test.drop(columns=['Unnamed: 0'], inplace=True)


train['Arrival Delay in Minutes'] = train['Arrival Delay in Minutes'].fillna(train['Arrival Delay in Minutes'].median())



test['Arrival Delay in Minutes'] = test['Arrival Delay in Minutes'].fillna(test['Arrival Delay in Minutes'].median())



submission=test.copy()


test.isnull().sum()



train['is_train'] = 1
test['is_train'] = 0


combined = pd.concat([train, test], axis=0)


combined


from sklearn.preprocessing import LabelEncoder


categorical_cols = combined.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    combined[col] = le.fit_transform(combined[col])


combined


train_processed = combined[combined['is_train'] == 1].drop(columns=['is_train'])


train_processed


test_processed = combined[combined['is_train'] == 0].drop(columns=['is_train', 'satisfaction'])


import seaborn as sns
import matplotlib.pyplot as plt


numerical_cols = train_processed.select_dtypes(include=['int64', 'float64']).drop(columns=['id', 'satisfaction']).columns

train_processed[numerical_cols].hist(figsize=(16, 12), bins=20, edgecolor='black')
plt.suptitle("Numerical Feature Distributions", fontsize=18)
plt.tight_layout()
plt.show()



categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='satisfaction', data=train_processed)
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=45)
    plt.legend(['Dissatisfied', 'Satisfied'])
    plt.tight_layout()
    plt.show()



sns.countplot(x='satisfaction', data=train_processed)
plt.title("Satisfaction Distribution")
plt.xticks([0, 1], ['Dissatisfied', 'Satisfied'])
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split



X = train_processed.drop(columns=['id', 'satisfaction'])
y = train_processed['satisfaction']
X_test_final = test_processed.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))





test_preds = model.predict(X_test_final)


test_preds


test_labels = ['Satisfied' if p == 1 else 'Dissatisfied' for p in test_preds]


submission['satisfaction'] = test_labels


submission.rename(columns={'id': 'ID'}, inplace=True)


submission[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


#submission.to_csv("Submission.csv", index=False)
#print("Submission file saved as 'Submission.csv'")


submission.head()








import pandas as pd
import lightgbm as lgb


train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")



label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le



X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']



imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)



from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
lgb_model.fit(X_train, y_train)



y_pred = lgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f" LightGBM Validation Accuracy: {accuracy * 100:.2f}%")




















