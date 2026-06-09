import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


for col in X.columns:
    if X[col].dtype == 'object':
        fill_val = 'missing'
    else:
        fill_val = X[col].median()  

    X[col] = X[col].fillna(fill_val)
    X_test[col] = X_test[col].fillna(fill_val)


for col in X.select_dtypes(include='object').columns:
    all_vals = pd.concat([X[col], X_test[col]], axis=0)
    le = LabelEncoder()
    le.fit(all_vals)

    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])


le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X, y_encoded)


# cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
# print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# preds = model.predict(X_test)
# preds_labels = le_target.inverse_transform(preds)


# submission = pd.DataFrame({
#     'id': test['id'],
#     'Personality': preds_labels
# })


# submission.to_csv('submission.csv', index=False)


# import lightgbm as lgb
# from sklearn.model_selection import cross_val_score


# lgb_model = lgb.LGBMClassifier(
#     n_estimators=500,
#     learning_rate=0.03,
#     max_depth=-1,
#     num_leaves=31,
#     random_state=42,
#     n_jobs=-1
# )


# lgb_model.fit(X, y_encoded)


# cv_scores = cross_val_score(lgb_model, X, y_encoded, cv=5, scoring='accuracy')
# print(f"LightGBM CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# preds = lgb_model.predict(X_test)
# preds_labels = le_target.inverse_transform(preds)


# submission = pd.DataFrame({
#     'id': test['id'],
#     'Personality': preds_labels
# })


# submission.to_csv('submission.csv', index=False)
# submission.head()


oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])
seeds = [0, 42, 2025]
n_splits = 5


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation


for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='accuracy',
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        oof_preds[val_idx] += model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1] / len(seeds)
        test_preds += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1] / (n_splits * len(seeds))



final_preds = (test_preds > 0.5).astype(int)
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': le_target.inverse_transform(final_preds)
})


submission.to_csv('submission.csv', index=False)

