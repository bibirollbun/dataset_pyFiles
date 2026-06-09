import numpy as np
import pandas as pd

data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
data.head(3)


data.info()


data.describe()


data.select_dtypes(include=object).head()


from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

###########
grade_subgrade = np.sort(data.grade_subgrade.unique())
grade_subgrade = list(grade_subgrade)
grade_subgrade = make_pipeline(
    OrdinalEncoder(categories=[grade_subgrade]),
    MinMaxScaler())
#############
onehot = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

############
preprocessing = ColumnTransformer([
    ('oe', grade_subgrade, ['grade_subgrade']),
    ('ohe', onehot, data.select_dtypes(include='object').columns.drop('grade_subgrade')),
    ('ss', StandardScaler(), data.select_dtypes(exclude='object').columns.drop('loan_paid_back')),
], remainder='drop').set_output(transform='pandas')


#############
X_train, X_val, y_train, y_val = train_test_split(data, data.loan_paid_back, test_size=0.2, random_state=40, stratify=data.loan_paid_back)
X_train_proc = preprocessing.fit_transform(X_train)
X_val_proc = preprocessing.transform(X_val)


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def xgbc(*args, **kwargs):
    xgbc = XGBClassifier(**kwargs)
    xgbc.fit(X_train_proc, y_train)
    
    
    y_proba = xgbc.predict_proba(X_val_proc)[:, 1]
    test_roc_auc = roc_auc_score(y_val, y_proba)
    print(f'Test ROC_AUC: {test_roc_auc}')
    
    y_train_proba = xgbc.predict_proba(X_train_proc)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    print(f'Train ROC_AUC: {train_roc_auc}')

    return xgbc

# estimator = xgbc()


# estimator = xgbc(max_depth=5, max_leaves=25, n_estimators=200)


positive_cases = data.loan_paid_back.sum()
negative_cases = data.shape[0] - positive_cases
neg_to_pos_ratio = negative_cases / positive_cases

# estimator = xgbc(objective='binary:logistic', scale_pos_weight=neg_to_pos_ratio)


estimator = xgbc(objective='binary:logistic', learning_rate=0.1, max_depth=8, max_leaves=10, n_estimators=2000, scale_pos_weight=neg_to_pos_ratio, n_jobs=-1)


from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, estimator.predict(X_val_proc))


# correct_pred = estimator.predict(X_val_proc) == y_val
# well = X_val[correct_pred]
# notwell = X_val[~correct_pred]


# display(X_val.describe())
# display(well.describe())
# display(notwell.describe())


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from numpy.random import random, seed
from scipy.stats import randint, uniform

seed(3)

xgb_classifier = Pipeline([
    ('preproc', preprocessing),
    ('model', XGBClassifier(objective='binary:logistic', n_jobs=-1))
])

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__subsample': uniform(0.8, 0.2),
    'model__colsample_bytree': uniform(0.8, 0.2),
    'model__max_depth': randint(3, 15),
    'model__max_leaves': randint(10, 50),
}

random_search = RandomizedSearchCV(xgb_classifier, param_dist,
                                   scoring='roc_auc', cv=sss,
                                   return_train_score=True,
                                   n_iter=50)


# random_search.fit(data, data.loan_paid_back)
# print(f'Best Score: {random_search.best_score_}')
# print(f'Best Params: {random_search.best_params_}')


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
# model = random_search.best_estimator_
# model.fit(data, target)
# predictions = model.predict_proba(test)[:, 1]

#############
train_proc, target = preprocessing.fit_transform(data), data.loan_paid_back
test_proc = preprocessing.transform(test)

model = estimator
model.fit(train_proc, target)
predictions = model.predict_proba(test_proc)[:, 1]
############

submission = pd.DataFrame({
    'loan_paid_back': predictions
})
submission.index = test.index
submission.to_csv('submission.csv', index=True)

