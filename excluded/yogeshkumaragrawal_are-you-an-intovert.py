import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Drop id
train = train.drop('id', axis=1)
test_id = test['id']
test = test.drop('id', axis=1)


# Separate target
X = train.drop('Personality', axis=1)
y = train['Personality']


# Label encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)


# Categorical columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']
num_cols = [col for col in X.columns if col not in cat_cols]


# Impute numeric with median
imputer_num = SimpleImputer(strategy='median')
X[num_cols] = imputer_num.fit_transform(X[num_cols])
test[num_cols] = imputer_num.transform(test[num_cols])


# Impute categorical with mode ('No')
imputer_cat = SimpleImputer(strategy='constant', fill_value='No')
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
test[cat_cols] = imputer_cat.transform(test[cat_cols])


# Label encode categorical
le_stage = LabelEncoder()
X['Stage_fear'] = le_stage.fit_transform(X['Stage_fear'])
test['Stage_fear'] = le_stage.transform(test['Stage_fear'])

le_drained = LabelEncoder()
X['Drained_after_socializing'] = le_drained.fit_transform(X['Drained_after_socializing'])
test['Drained_after_socializing'] = le_drained.transform(test['Drained_after_socializing'])


# Train-test split for local validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)


# Local accuracy
y_pred = rf.predict(X_val)
print('Local Accuracy:', accuracy_score(y_val, y_pred))


# CV score
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print('CV Accuracy:', cv_scores.mean())


# Fit on full train
rf.fit(X, y)


# Predict on test
test_pred = rf.predict(test)
test_pred = le_target.inverse_transform(test_pred)


# Submission
submission['Personality'] = test_pred
submission.to_csv('submission.csv', index=False)
print('Submission saved!')

