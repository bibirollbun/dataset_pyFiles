import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# 2. Confirm that 'y' is already numeric (0/1)
print("Target unique values:", train['y'].unique())

# 3. Preprocess categorical features
categorical_cols = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'poutcome'
]
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# Align test columns with train (fill missing columns with 0)
test = test.reindex(columns=train.columns.drop('y'), fill_value=0)

# 4. Separate features and target
X = train.drop(columns=['id', 'y'])
y = train['y']
X_test = test.drop(columns=['id'])

# 5. Scale numerical features
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 7. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# 8. Evaluate with ROC AUC on training data
y_res_proba = model.predict_proba(X_res)[:, 1]
roc_auc = roc_auc_score(y_res, y_res_proba)
print(f"Training ROC AUC: {roc_auc:.4f}")

# 9. Generate predictions on the test set
y_test_proba = model.predict_proba(X_test)[:, 1]

# 10. Prepare and save the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'y': y_test_proba
})
submission.to_csv('submission.csv', index=False)

