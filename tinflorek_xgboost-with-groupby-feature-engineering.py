import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from xgboost import XGBRegressor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col='id')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


print(f"Train data: {train_data.shape}")

train_data.head()


print(f"Test data: {test_data.shape}")


train_data.isnull().sum()


FEATURES = list(test_data.columns)
TARGET = 'Calories'

STATS = ["mean","std","count","median"]


FOLDS = 5 

kfold_1 = KFold(n_splits=FOLDS, shuffle=True, random_state=123)

oof = np.zeros(len(train_data))
preds = np.zeros(len(test_data))

for fold, (train_index, test_index) in enumerate(kfold_1.split(train_data)):
    print(f"OUTER FOLD {fold+1}")
    
    X_train = train_data.loc[train_index, FEATURES+[TARGET]].reset_index(drop=True).copy()
    y_train = train_data.loc[train_index, TARGET]
    X_valid = train_data.loc[test_index, FEATURES].reset_index(drop=True).copy()
    y_valid = train_data.loc[test_index, TARGET]
    X_test = test_data[FEATURES].reset_index(drop=True).copy()

    X_valid_encoded = X_valid.copy()
    X_test_encoded = X_test.copy()

    kfold_2 = KFold(n_splits=FOLDS, shuffle=True, random_state=123)
    for fold2, (train_index2, test_index2) in enumerate(kfold_2.split(X_train)):
        print(f"INNER Fold {fold2+1} (outer fold {fold+1})")
        X_train2 = X_train.loc[train_index2, FEATURES+[TARGET]].copy()
        X_valid2 = X_train.loc[test_index2, FEATURES].copy()

        for col in FEATURES:

            tmp = X_train2.groupby(col)[TARGET].agg(STATS)
            tmp.columns = [f"TE_{col}_{s}" for s in STATS]

            X_valid2 = X_valid2.merge(tmp, on=col, how="left")

            for c in tmp.columns:
                X_train.loc[test_index2, c] = X_valid2[c].values

            if fold2 == 0:
                X_valid_encoded = X_valid_encoded.merge(tmp, on=col, how="left")
                X_test_encoded = X_test_encoded.merge(tmp, on=col, how="left")

    X_train['Sex'] = X_train['Sex'].astype("category")
    X_valid_encoded['Sex'] = X_valid_encoded['Sex'].astype("category")
    X_test_encoded['Sex'] = X_test_encoded['Sex'].astype("category")

    te_cols = [col for col in X_train.columns if col.startswith('TE_')]
    model_cols = FEATURES + te_cols

    if TARGET in model_cols:
        model_cols.remove(TARGET)

    model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
    
    model.fit(
        X_train[model_cols], 
        y_train, 
        eval_set=[(X_valid_encoded[model_cols], y_valid)], 
        eval_metric='rmsle'
    )

    oof[test_index] = model.predict(X_valid_encoded[model_cols])
    preds += model.predict(X_test_encoded[model_cols])

preds /= FOLDS


sub = pd.DataFrame({'id': test_data.index, 'Calories': preds})
sub.to_csv('submission.csv', index=False)

