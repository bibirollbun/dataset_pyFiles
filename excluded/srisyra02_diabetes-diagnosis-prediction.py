import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

import catboost as cb

print("S11")



train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub   = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.shape, test.shape



train.head()



y = train["diagnosed_diabetes"]
X = train.drop("diagnosed_diabetes", axis=1)



cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols


train.isna().sum()
test.isna().sum()


for col in cat_cols:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)



from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50
)

model.fit(
    X, y,
    cat_features=cat_cols
)



test_pred = model.predict_proba(test)[:, 1]



from sklearn.metrics import roc_auc_score

train_pred = model.predict_proba(X)[:, 1]
score = roc_auc_score(y, train_pred)
print("Train ROC-AUC:", score)


sub["diagnosed_diabetes"] = test_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv!")



This is my first Kaggle Notebook, so if you found it helpful, I’d really appreciate an upvote. Thank you

