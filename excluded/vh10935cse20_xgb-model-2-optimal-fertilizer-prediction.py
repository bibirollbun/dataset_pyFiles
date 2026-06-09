import numpy as np
import pandas as pd


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
org=pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


train.shape


test_new=test


test.shape


train.head(3)


test.head(3)


train.isna().sum()


test.isna().sum()


train.columns


test.columns


train=train.drop('id',axis=1)
test=test.drop('id',axis=1)
train_new=pd.concat([train,org],ignore_index=True)


train_new.dtypes


test.dtypes


x=train_new.drop(['Fertilizer Name'],axis=1)
y=train_new['Fertilizer Name']


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


model=LabelEncoder()
y_enc=model.fit_transform(y)


cat_cols=['Soil Type','Crop Type']
x=pd.get_dummies(x,columns=cat_cols)
test=pd.get_dummies(test,columns=cat_cols)


X_train,X_val,y_train,y_val=train_test_split(x,y_enc,test_size=0.2,random_state=42)


from xgboost import XGBClassifier


xgb_best_params = {
   'n_estimators': 3500,
    'max_depth':12,
    'subsample': 0.9,
    'colsample_bytree':0.5,
    'learning_rate':0.03,
    'gamma':0.5,
    'max_delta_step': 5,
    'early_stopping_rounds':50,
    # 'objective':'multi:softprob',
    # 'objective':'rank:map',
    'objective': 'multi:softmax',
    'enable_categorical':True,
    'tree_method':'hist',
    'device':'cuda',
    'reg_alpha':2.7,
    'reg_lambda':1.4,
    'num_parallel_tree': 5,
    # 'disable_default_eval_metric': True,    
    # 'eval_metrics': 'accuracy',
    # 'verbose': 100
}
xgb=XGBClassifier(**xgb_best_params)


xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)


y_probs=xgb.predict_proba(X_val)


def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


# Single-label MAP@K
def mapk_single_label(y_true, y_pred, k=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[:, :k]
    matches = (y_true.reshape(-1, 1) == y_pred)
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, np.inf)
    return np.mean(ranks ** -1)


# Multi-label MAP@K (each instance has one label in a list)
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0
    seen = set()
    actual_set = set(actual)
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            num_hits += 1
            score += num_hits / (i + 1)
            seen.add(p)
    return score / min(len(actual), k)
def mapk(actual, predicted, k=10):
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])
# Evaluate MAP@K for k = 1 to k_max


#Evaluate MAP@K for k = 1 to k_max
k_max = 8
k_values = range(1, k_max)
mapk_single_scores = []
mapk_multi_scores = []

for k in k_values:
    top_k_preds = get_top_k_predictions(y_probs, k)
    mapk_single_scores.append(mapk_single_label(y_val, top_k_preds, k))
    mapk_multi_scores.append(mapk(y_val, top_k_preds, k))


test_proba = xgb.predict_proba(test)


preds = np.argsort(test_proba, axis=1)[:, ::-1]
preds


test_top_3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_top_3


test_top_3_names = model.inverse_transform(test_top_3.ravel())
test_3_picks = test_top_3_names.reshape(test_top_3.shape)

test_3_picks


preds_df = pd.DataFrame({
    'id': test_new['id'],
    'Fertilizer Name': [' '.join(preds) for preds in test_3_picks]
})

preds_df.head(4)


preds_df.to_csv('/kaggle/working/submission.csv', index=False)

