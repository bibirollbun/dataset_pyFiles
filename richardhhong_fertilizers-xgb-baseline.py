import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

SEED = 30


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


def mapk(actual, predicted, k=3):
    total_score = 0.0
    actual = le.inverse_transform(actual)
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            index = p.index(a)
            total_score += 1.0 / (index + 1)
    return total_score / len(actual)


le = LabelEncoder()
le.fit(df_train['Fertilizer Name'])

def make_features(df, test=False):
    df_temp = df.copy()
    df_temp.drop(columns=['id'], inplace=True)
    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].astype('category')

    if not test:
        df_temp['Fertilizer Name'] = le.transform(df_temp['Fertilizer Name'])
    return df_temp


df_train1 = make_features(df_train)


params = {
    'seed': SEED,
    'enable_categorical': True,
    'early_stopping_rounds': 100
}


X = df_train1.drop(columns=['Fertilizer Name'])
y = df_train1['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED) # separate holdout validation model never sees


def cross_val(X, y, params=params, K=10):
    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_scores = []
    fold = 0

    for train_idx, val_idx in kf.split(X):
        fold += 1
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            **params
        )
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

        val_pred = model.predict_proba(X_val)
        val_pred = np.argsort(val_pred, axis=1)[:, -3:][:, ::-1]
        val_pred = [[le.classes_[j] for j in row] for row in val_pred]

        score = mapk(y_val, val_pred)
        fold_scores.append(score)
        print(f'Fold {fold} Mean Average Precision Score: {score}')

    avg_score = np.mean(fold_scores)
    print(f'Average Validation Fold Score: {avg_score}')


cross_val(X_train, y_train)


model = XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


df_test1 = make_features(df_test, test=True)
y_test_pred = model.predict_proba(df_test1)
y_test_pred = np.argsort(y_test_pred, axis=1)[:, -3:][:, ::-1]
y_test_pred = [[le.classes_[j] for j in row] for row in y_test_pred]
y_test_pred = [' '.join(row) for row in y_test_pred]

submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission['Fertilizer Name'] = y_test_pred
submission.to_csv('submission.csv', index=False)
submission.head()

