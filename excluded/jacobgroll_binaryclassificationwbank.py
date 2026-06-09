import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

X = train_data.drop(columns=['y'])
y = train_data['y']

X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val  = train_test_split(
    X, y, test_size=0.2, random_state=0
)

rf = RandomForestClassifier(
    n_estimators=50, max_depth=15, n_jobs=-1, random_state=0
)
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X, y, cv=5)
print(scores)
print(rf.score(X_val, y_val))


test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_id = test_data['id']
X_test = test_data.drop(columns=['id'])

X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

predictions = rf.predict(X_test)
print(predictions[:50])


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2, random_state=1
)

rf.fit(X_train_split, y_train_split)
val_preds = rf.predict(X_val_split)

print(accuracy_score(y_val_split, val_preds))


# param_dist = {
#     'n_estimators':[20, 50, 100],
#     'max_depth':[None, 10, 20, 30],
#     'min_samples_split':[2, 5, 10],
#     'min_samples_leaf':[1, 2, 4],
#     'max_features':['sqrt', 'log2'],
#     'bootstrap':[True, False]
# }

# rf = RandomForestClassifier(random_state=0, n_jobs=-1)

# rf_random = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_dist,
#     n_iter=15,
#     cv=3,
#     verbose=2,
#     random_state=0,
#     n_jobs=-1
# )

# rf_random.fit(X, y)

# print(rf_random.best_params_)
# print(rf_random.best_score_)


best_params = {
    'n_estimators':50,
    'max_depth':30,
    'min_samples_split':2,
    'min_samples_leaf':2,
    'max_features':'log2',
    'bootstrap':True
}
final_model = RandomForestClassifier(**best_params)

final_model.fit(X_train, y_train)


y_pred = final_model.predict(X_val)
print(accuracy_score(y_val, y_pred))


test_proba = final_model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'id':test_id, 'y':test_proba})
submission.to_csv('submission.csv', index=False)

