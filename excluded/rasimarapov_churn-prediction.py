import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')



test_ids = test['Customer_ID']

target = 'Churn'

all_data = pd.concat([train.drop(target, axis=1), test], axis=0)

cat_cols = all_data.select_dtypes(include='object').columns

le = LabelEncoder()
for col in cat_cols:
    all_data[col] = le.fit_transform(all_data[col].astype(str))

X_train = all_data.iloc[:len(train)]
X_test = all_data.iloc[len(train):]
y_train = train[target]



X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_tr, y_tr)

val_preds = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))




test_probs = model.predict_proba(X_test)[:, 1]  # вероятность класса 1 (churn)


submission = pd.DataFrame({
    'Customer_ID': test_ids,
    'Churn': test_probs  
})

submission.to_csv('submission.csv', index=False)


