import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print(train.info())
print('='*70)
print(test.info())
print('='*70)
print(sample.info())
print('='*70)


y = train['y']
X = train.drop('y', axis=1)

combined = pd.concat([X, test], ignore_index=True)

categorical_cols = [cname for cname in combined.columns if combined[cname].dtype == 'object']

combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True, dtype=float)

X_final = combined_encoded.iloc[:len(train)]
test_final = combined_encoded.iloc[len(train):]

print("Preprocessing complete.")
print("Shape of final training data:", X_final.shape)
print("Shape of final test data:", test_final.shape)


from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

X_train = X_final.drop('id', axis=1)
test_train = test_final.drop('id', axis=1)

clf1 = CatBoostClassifier(random_state=42, silent=True)
clf2 = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

model = VotingClassifier(estimators=[('catboost', clf1), ('xgb', clf2)], voting='soft')

model.fit(X_train, y)

print("Ensemble model (CatBoost + XGB) training complete.")


predictions = model.predict_proba(test_train)[:, 1]

submission = pd.DataFrame({'id': test['id'], 'y': predictions})

submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
submission.head()

