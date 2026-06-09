!pip install pgmpy --quiet


import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch, BDeu, BayesianEstimator
from sklearn.metrics import accuracy_score


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
personality = pd.read_csv('/kaggle/input/personality-dataset/personality_dataset.csv')


test_ids = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


train = pd.concat([train, personality], axis=0)


def bayesian_model(X):
    hc = HillClimbSearch(X)
    best_model = hc.estimate(scoring_method=BDeu(X))
    model = DiscreteBayesianNetwork(best_model.edges())
    model.fit(X, estimator=BayesianEstimator)
    return model


def handle_missing_values(model, X):
    infer = VariableElimination(model)
    X_imputed = X.copy()
    for idx, row in X.iterrows():
        known_evidence = row.dropna().to_dict()
        missing_vars = row[row.isna()].index
        for var in missing_vars:
            q = infer.query(variables=[var], evidence=known_evidence)
            predicted_value = q.values.argmax()
            X_imputed.at[idx, var] = predicted_value
    return X_imputed


def bayesian_predict(model, X, target):
    infer = VariableElimination(model)
    pred = []
    for _, row in X.iterrows():
        result = infer.query(variables=[target], evidence=row.dropna().to_dict())
        pred.append(result.values.argmax())
    return pd.Series(pred)


def set_binary(val):
    if val == 'Yes' or val == 'Introvert':
        return 1
    else:
        return 0


cat_train_cols = train.select_dtypes(include='object').columns
for col in cat_train_cols:
    train[col] = train[col].apply(set_binary)



cat_test_cols = test.select_dtypes(include='object').columns
for col in cat_test_cols:
    test[col] = test[col].apply(set_binary)


train.drop_duplicates(inplace=True)
train.duplicated().sum()


train.isna().sum()


complete_train = train[train.notna().all(axis=1)]
missing_train = train[train.isna().any(axis=1)]


complete_train.isna().sum()


missing_train.isna().sum()


model_fillna = bayesian_model(complete_train)
filled_missing_train = handle_missing_values(model_fillna, missing_train)


train = pd.concat([complete_train, filled_missing_train], axis=0)


train.isna().sum()


target = 'Personality'
X = train.drop(columns=[target])
y = train[target]


model = bayesian_model(train)
pred_y_train = bayesian_predict(model, X, target)
accuracy_score(y, pred_y_train)


pred_y_test = bayesian_predict(model, test, target)


sub = pd.DataFrame({
    'id': test_ids,
    'Personality': pred_y_test.apply(lambda p: 'Introvert' if p == 1 else 'Extrovert')
})
sub.to_csv('submission.csv', index=False)


