import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col=['id'])
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col=['id'])


main_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
main_df['y'] = main_df['y'].map({'yes':1, 'no':0})


final_df = pd.concat([train_df, main_df], ignore_index=True)
final_df = final_df.drop_duplicates()


X = final_df.drop('y', axis=1).astype('str')
y = final_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y, random_state=0)


cat_clf = CatBoostClassifier(
    allow_writing_files=False,
    verbose=False,
    cat_features=X.columns.to_list(),
    task_type='GPU',
    n_estimators=10000,
    learning_rate=0.05,
)

cat_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=300, verbose=1000)
y_pred = cat_clf.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)


test_pred = cat_clf.predict_proba(test_df.astype('str'))[:, 1]


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()




