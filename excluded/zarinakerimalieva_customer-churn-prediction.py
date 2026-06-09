import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')


train['is_train'] = 1
test['is_train'] = 0
test['Churn'] = None  


full = pd.concat([train, test], ignore_index=True)

cat_cols = ['Gender', 'Location', 'Subscription_Type', 'Last_Interaction_Type']
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))


full.drop(['Customer_ID'], axis=1, inplace=True)


train = full[full['is_train'] == 1].drop(columns=['is_train'])
test = full[full['is_train'] == 0].drop(columns=['is_train', 'Churn'])

train = train.dropna(subset=['Churn'])


X = train.drop('Churn', axis=1)
y = train['Churn'].astype(int)  


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)
print("Validation F1-score:", f1_score(y_val, val_preds))


test_preds = model.predict(test)


submission = pd.read_csv("/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv")[["Customer_ID"]].copy()
submission["Churn"] = test_preds
submission.to_csv("submission.csv", index=False)



