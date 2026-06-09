# 1. Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# 3. EDA
test.head()


train.head()


print(train.describe())
print(train['rainfall'].value_counts())


# 4. Handle missing values (simple strategy)
imputer = SimpleImputer(strategy='mean')
X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']
X_test = test.drop(['id'], axis=1)

X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# 5. Train/Validation split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 6. Baseline model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# 7. Validation performance
val_preds = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC Score: {auc_score:.4f}")


# 8. Predict on test set
test_preds = model.predict_proba(X_test)[:, 1]


# 9. Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

