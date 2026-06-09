
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
test_ids = test['id']
X_test = test.drop('id', axis=1)

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

y = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]}

rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
rf_search.fit(X_train, y_train)

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
xgb_search.fit(X_train, y_train)

rf_pred = rf_search.predict(X_val)
xgb_pred = xgb_search.predict(X_val)

print("Random Forest Metrics:")
print(classification_report(y_val, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, rf_pred))

print("\nXGBoost Metrics:")
print(classification_report(y_val, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, xgb_pred))

rf_score = rf_search.score(X_val, y_val)
xgb_score = xgb_search.score(X_val, y_val)
best_model = rf_search if rf_score >= xgb_score else xgb_search
print(f"\nBest Model: {'Random Forest' if rf_score >= xgb_score else 'XGBoost'} (ROC-AUC: {max(rf_score, xgb_score):.4f})")

test_pred = best_model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({'id': test_ids, 'Response': test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
test_ids = test['id']
X_test = test.drop('id', axis=1)

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ('RandomForest', RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}),
    ('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss'), {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]}),
    ('LightGBM', LGBMClassifier(random_state=42, verbose=-1), {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]}),
    ('CatBoost', CatBoostClassifier(random_state=42, verbose=0), {'iterations': [100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1]}),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}),
    ('ExtraTrees', ExtraTreesClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}),
    ('AdaBoost', AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
    ('Bagging', BaggingClassifier(random_state=42), {'n_estimators': [10, 50], 'max_samples': [0.5, 1.0]}),
    ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000), {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}),
    ('KNeighbors', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    ('SVM', SVC(random_state=42), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    ('DecisionTree', DecisionTreeClassifier(random_state=42), {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}),
    ('GaussianNB', GaussianNB(), {}),
    ('BernoulliNB', BernoulliNB(), {'alpha': [0.1, 1.0]}),
    ('LDA', LinearDiscriminantAnalysis(), {}),
    ('QDA', QuadraticDiscriminantAnalysis(), {}),
    ('MLP', MLPClassifier(random_state=42, max_iter=500), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}),
    ('SGD', SGDClassifier(random_state=42, max_iter=1000), {'loss': ['hinge', 'log'], 'alpha': [0.0001, 0.001]}),
    ('Ridge', RidgeClassifier(random_state=42), {'alpha': [0.1, 1.0, 10.0]}),
    ('PassiveAggressive', PassiveAggressiveClassifier(random_state=42), {'C': [0.1, 1.0, 10.0]}),
    ('GaussianProcess', GaussianProcessClassifier(random_state=42), {}),
    ('Dummy', DummyClassifier(random_state=42), {'strategy': ['most_frequent', 'stratified']})
]

best_model = None
best_score = 0
best_name = ''

for name, model, params in models:
    if params:
        search = RandomizedSearchCV(model, params, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        pred = search.predict(X_val)
        score = search.score(X_val, y_val)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) if hasattr(model, 'predict_proba') else model.score(X_val, y_val)
        search = model
    print(f"\n{name} Metrics:")
    print(classification_report(y_val, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, pred))
    print(f"ROC-AUC: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = search
        best_name = name

print(f"\nBest Model: {best_name} (ROC-AUC: {best_score:.4f})")

test_pred = best_model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({'id': test_ids, 'Response': test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, 
                                          QuadraticDiscriminantAnalysis)
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
import warnings
warnings.filterwarnings('ignore')

# Data Loading and Preprocessing
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
test_ids = test['id']
X_test = test.drop('id', axis=1)

# Imputation and Encoding
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
le = LabelEncoder()

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

for col in cat_cols:
    X[col] = le.fit_transform(cat_imputer.fit_transform(X[col].values.reshape(-1, 1)))
    X_test[col] = le.transform(cat_imputer.transform(X_test[col].values.reshape(-1, 1)))

y = le.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers with Hyperparameters
classifiers = {
    'Random Forest': (RandomForestClassifier(), 
                     {'n_estimators': [100, 200, 300],
                      'max_depth': [10, 20, None],
                      'min_samples_split': [2, 5]}),
    
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                {'n_estimators': [100, 200],
                 'max_depth': [3, 6, 9],
                 'learning_rate': [0.01, 0.1, 0.2]}),
    
    'LightGBM': (LGBMClassifier(),
                 {'n_estimators': [100, 200],
                  'max_depth': [5, 10],
                  'learning_rate': [0.01, 0.1]}),
    
    'CatBoost': (CatBoostClassifier(verbose=0),
                 {'iterations': [100, 200],
                  'depth': [4, 6],
                  'learning_rate': [0.01, 0.1]}),
    
    'Logistic Regression': (LogisticRegression(max_iter=1000),
                            {'C': [0.1, 1, 10],
                             'penalty': ['l1', 'l2']}),
    
    'SVM': (SVC(probability=True),
             {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf']}),
    
    'KNN': (KNeighborsClassifier(),
             {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance']}),
    
    'Decision Tree': (DecisionTreeClassifier(),
                       {'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5]}),
    
    'Gradient Boosting': (GradientBoostingClassifier(),
                           {'n_estimators': [100, 200],
                            'learning_rate': [0.01, 0.1],
                            'max_depth': [3, 5]}),
    
    'AdaBoost': (AdaBoostClassifier(),
                  {'n_estimators': [50, 100],
                   'learning_rate': [0.01, 0.1]}),
    
    'Extra Trees': (ExtraTreesClassifier(),
                     {'n_estimators': [100, 200],
                      'max_depth': [10, 20, None]}),
    
    'LDA': (LinearDiscriminantAnalysis(), {}),
    
    'QDA': (QuadraticDiscriminantAnalysis(), {}),
    
    'Gaussian NB': (GaussianNB(), {}),
    
    'MLP': (MLPClassifier(max_iter=1000),
             {'hidden_layer_sizes': [(50,), (100,)],
              'activation': ['relu', 'tanh']}),
    
    'SGD': (SGDClassifier(),
             {'loss': ['hinge', 'log_loss'],
              'penalty': ['l2', 'elasticnet']}),
    
    'Ridge': (RidgeClassifier(),
               {'alpha': [0.1, 1, 10]}),
    
    'Bagging': (BaggingClassifier(),
                 {'n_estimators': [10, 20],
                  'max_samples': [0.5, 1.0]}),
    
    'Gaussian Process': (GaussianProcessClassifier(), {})
}

# Model Training and Evaluation
results = {}
best_score = 0
best_model = None

for name, (clf, params) in classifiers.items():
    print(f"\n=== Training {name} ===")
    
    if params:  # Do hyperparameter tuning if parameters are specified
        search = RandomizedSearchCV(clf, params, n_iter=10, cv=3, 
                                  scoring='roc_auc', n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = clf.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'report': classification_report(y_val, y_pred),
        'confusion': confusion_matrix(y_val, y_pred)
    }
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(results[name]['report'])
    print("\nConfusion Matrix:")
    print(results[name]['confusion'])
    
    # Track best model
    if roc_auc is not None and roc_auc > best_score:
        best_score = roc_auc
        best_model = model

# Ensemble Methods
print("\n=== Training Ensemble Models ===")

# Voting Classifier
vote_clf = VotingClassifier(
    estimators=[(name, results[name]['model']) 
               for name in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']],
    voting='soft'
)
vote_clf.fit(X_train, y_train)
vote_pred = vote_clf.predict(X_val)
vote_proba = vote_clf.predict_proba(X_val)[:, 1]

# Stacking Classifier
stack_clf = StackingClassifier(
    estimators=[(name, results[name]['model']) 
               for name in ['Random Forest', 'XGBoost', 'LightGBM']],
    final_estimator=LogisticRegression()
)
stack_clf.fit(X_train, y_train)
stack_pred = stack_clf.predict(X_val)
stack_proba = stack_clf.predict_proba(X_val)[:, 1]

# Evaluate Ensembles
for name, pred, proba in [('Voting', vote_pred, vote_proba), 
                          ('Stacking', stack_pred, stack_proba)]:
    accuracy = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    roc_auc = roc_auc_score(y_val, proba)
    
    results[name] = {
        'model': vote_clf if name == 'Voting' else stack_clf,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'report': classification_report(y_val, pred),
        'confusion': confusion_matrix(y_val, pred)
    }
    
    print(f"\n{name} Ensemble Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = vote_clf if name == 'Voting' else stack_clf

# Final Model Selection and Submission
print(f"\n=== Best Model: {type(best_model).__name__} with ROC AUC {best_score:.4f} ===")

test_pred = best_model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({'id': test_ids, 'Response': test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

# Results Summary
print("\n=== Performance Summary ===")
summary = pd.DataFrame.from_dict({k: [v['accuracy'], v['f1'], v['roc_auc']] 
                                for k, v in results.items()}, 
                               orient='index', 
                               columns=['Accuracy', 'F1', 'ROC AUC'])
print(summary.sort_values(by='ROC AUC', ascending=False))


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Data Loading and Preprocessing
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
test_ids = test['id']
X_test = test.drop('id', axis=1)

# Imputation and Encoding
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
le = LabelEncoder()

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

for col in cat_cols:
    X[col] = le.fit_transform(cat_imputer.fit_transform(X[col].values.reshape(-1, 1)))
    X_test[col] = le.transform(cat_imputer.transform(X_test[col].values.reshape(-1, 1)))

y = le.fit_transform(y)

# Hyperparameter Tuning for Random Forest (best model)
rf_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=50, cv=5, 
                             scoring='roc_auc', n_jobs=-1, random_state=42)
rf_search.fit(X, y)

# Get best model
best_rf = rf_search.best_estimator_
print(f"Best Parameters: {rf_search.best_params_}")
print(f"Best ROC-AUC Score: {rf_search.best_score_:.4f}")

# Make predictions
test_pred = best_rf.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)

# Save submission
submission = pd.DataFrame({'id': test_ids, 'Response': test_pred_labels})
submission.to_csv('new_submission.csv', index=False)
print("Submission saved as new_submission.csv")




