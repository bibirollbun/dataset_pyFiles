import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/train-csv/train.csv')
test = pd.read_csv('/kaggle/input/test-csv/test.csv')

cat_features = ['Geography', 'Gender']

X_train, X_val, y_train, y_val = train_test_split(
    train.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1),
    train['Exited'],
    test_size=0.2,
    random_state=17
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.001,
    depth=6,
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool)

best_iteration = model.get_best_iteration()

full_train_pool = Pool(
    train.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1),
    train['Exited'],
    cat_features=cat_features
)

final_model = CatBoostClassifier(
    iterations=int(best_iteration * 1.2),
    learning_rate=0.001,
    depth=6,
    eval_metric='AUC',
    verbose=100
)

final_model.fit(full_train_pool)

X_test = test.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_predictions = final_model.predict_proba(X_test)[:, 1]

output = pd.DataFrame({
    'id': test['id'],
    'Exited': test_predictions
})

output.to_csv('submission.csv', index=False, header=True)

