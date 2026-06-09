# import required libraries
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# list data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# read data files
df_sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# this is nice dataset shared on kaggle
df_additional = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
df_train = pd.concat([df_train, df_additional], axis=0, ignore_index=True)

df_train.info()


# delete 'id' column
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

df_train.head()


# check if we need to handle missing values
missing_values_count = df_train.isnull().sum()
missing_values_count


# use ordinal encoder to encode category columns
cat_cols_train = df_train.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'Fertilizer Name']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train[cat_cols_train] = ordinal_encoder.fit_transform(df_train[cat_cols_train].astype(str))
df_test[cat_cols_train] = ordinal_encoder.transform(df_test[cat_cols_train].astype(str))

# use label encoder to encode label values
le = LabelEncoder()
df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])


y = df_train['Fertilizer Name'] 
X = df_train.drop(['Fertilizer Name'],axis=1)


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

discrete_features = X.dtypes != object
mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores
# fertilizer does not have very strong co-relation with any particular field


# predicting using xgboost
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(df_train), len(np.unique(y))))
pred = np.zeros((len(df_test), len(np.unique(y))))

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = df_test.copy()

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 16,
        'learning_rate': 0.03,
        'min_child_weight' : 2,
        'alpha': 0.8, 
        'reg_lambda': 4.0, 
        'colsample_bytree': 0.3,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=200
    )

    pred += model.predict(dtest)

pred /= FOLDS


top_preds = np.argsort(pred, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
score = mapk(actual, top_preds)
print(f"Score: {score:.5f}")

# Score: 0.26839


top_preds = np.argsort(pred, axis=1)[:, -3:][:, ::-1]
top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_labels]
})
submission.to_csv('submission.csv', index=False)

