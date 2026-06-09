import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(train.shape, test.shape)
train.head()




TARGET_COL = 'diagnosed_diabetes'

y = train[TARGET_COL]
X = train.drop([TARGET_COL, 'id'], axis=1)

test_ids = test['id']
X_test = test.drop(['id'], axis=1)



# Find categorical (non-numeric) columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", cat_cols)



X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=cat_cols,        
    eval_set=(X_val, y_val),
    use_best_model=True
)



val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", val_auc)



full_model = CatBoostClassifier(
    iterations= model.tree_count_,   # reuse best number of trees
    depth=8,
    learning_rate=0.03,
    loss_function='Logloss',
    random_seed=42,
    verbose=200
)

full_model.fit(
    X, y,
    cat_features=cat_cols
)



test_preds = full_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)


