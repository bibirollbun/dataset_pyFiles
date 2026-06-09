import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb


df_orig = (
    pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
    .rename(columns={
        'Personality': 'match_p'
    })
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.isnull().sum()


print(train.info())
print(test.info())


train = train.merge(df_orig, how='left')
test = test.merge(df_orig, how='left')


le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])


combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])


from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
combined_imputed = pd.DataFrame(imputer.fit_transform(combined), columns=combined.columns)


X = combined_imputed.iloc[:len(X)].reset_index(drop=True)
X_test = combined_imputed.iloc[len(X):].reset_index(drop=True)


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


N_SPLITS = 5
N_REPEATS = 1
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50, verbose_eval=False)
    
    oof_preds[val_idx] += model.predict(dval) / N_REPEATS
    test_preds += model.predict(dtest) / (N_REPEATS * N_SPLITS)


ll = log_loss(y, oof_preds)
cv_acc = accuracy_score(y, oof_preds > 0.5)
print(f"Cross-Validation log loss: {ll:.4f}, accuracy: {cv_acc:.4f}")


final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()




