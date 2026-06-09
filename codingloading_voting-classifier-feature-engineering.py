!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0 --quiet



#  Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



#  Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


#  Encode Binary Features
binary_map = {'Yes': 1, 'No': 0}
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = train[col].map(binary_map)
    test[col] = test[col].map(binary_map)



#  Feature Engineering
train['Alone_per_Social'] = train['Time_spent_Alone'] / (train['Social_event_attendance'] + 1)
test['Alone_per_Social'] = test['Time_spent_Alone'] / (test['Social_event_attendance'] + 1)

train['Social_Activity_Index'] = train[['Social_event_attendance', 'Going_outside', 'Friends_circle_size']].mean(axis=1)
test['Social_Activity_Index'] = test[['Social_event_attendance', 'Going_outside', 'Friends_circle_size']].mean(axis=1)

train['Stage_fear_AND_Drained'] = train['Stage_fear'] * train['Drained_after_socializing']
test['Stage_fear_AND_Drained'] = test['Stage_fear'] * test['Drained_after_socializing']


#  Fill Missing Values
features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency', 'Stage_fear',
            'Drained_after_socializing', 'Alone_per_Social',
            'Social_Activity_Index', 'Stage_fear_AND_Drained']


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(train[features], train['Personality'])

print("âœ… Class counts after balancing:", Counter(y_balanced))




for col in features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())



#  Encode Target
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # 0=Extrovert, 1=Introvert


#  Scale Features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(train[features]), columns=features)
X_test = pd.DataFrame(scaler.transform(test[features]), columns=features)
y = train['Personality']


#  Define Base Models
models = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42))
]


#  Voting Classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X, y, cv=skf, scoring='accuracy')
print(f"\nâœ… Voting Classifier CV Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")


#  Fit on full training data
voting_clf.fit(X, y)

#  Predict on test data
preds = voting_clf.predict(X_test)

#  Decode back to original labels
pred_labels = le.inverse_transform(preds)  # Converts 0 â†’ 'Extrovert', 1 â†’ 'Introvert'

#   submission file
submission['Personality'] = pred_labels
submission.to_csv("submission.csv", index=False)

#  Check
print(submission.head())
print(submission['Personality'].value_counts())





