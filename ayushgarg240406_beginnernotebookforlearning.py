%config IPCompleter.greedy=True


import numpy as np
import pandas as pd
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


print(df_train.head())
print(df_test.head())


df_train.info()


df_train.columns = df_train.columns.str.strip().str.lower()
df_test.columns = df_test.columns.str.strip().str.lower()
cat_types = ['soil type', 'crop type', 'fertilizer name']  # also lowercase
for cat in cat_types:
    print(f"\nUnique counts in '{cat}':")
    print(df_train[cat].unique())
    print(df_train[cat].value_counts())      



print(df_train.describe())


%matplotlib inline
import matplotlib.pyplot as plt
df_train.hist(bins = 20,figsize = (20,15))
plt.show()
              


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
from sklearn.preprocessing import OrdinalEncoder
ord_encode = OrdinalEncoder()
from sklearn.compose import ColumnTransformer
ohe_cats_cols = ['soil type', 'crop type']
ord_cats_cols =  ['fertilizer name']
cat_transformed = cat_encoder.fit_transform(df_train[ohe_cats_cols])
cat_test_transformed = cat_encoder.transform(df_test[ohe_cats_cols])



tf_columns = cat_encoder.get_feature_names_out(ohe_cats_cols)
cat_transformed_pd = pd.DataFrame(cat_transformed,columns =tf_columns)
cat_test_transformed_pd = pd.DataFrame(cat_test_transformed,columns =tf_columns)


print(cat_transformed_pd.head())
print(cat_test_transformed_pd.head())



df_train = df_train.drop(columns=ohe_cats_cols, errors='ignore')
df_test = df_test.drop(columns=ohe_cats_cols, errors='ignore')

df_train = pd.concat([df_train.reset_index(drop=True), cat_transformed_pd.reset_index(drop=True)], axis=1)
df_test = pd.concat([df_test.reset_index(drop=True), cat_test_transformed_pd.reset_index(drop=True)], axis=1)



ord_cats_cols_transformed = ord_encode.fit_transform(df_train[ord_cats_cols])
ord_tf_columns = ord_encode.get_feature_names_out(ord_cats_cols)
ord_cats_pd = pd.DataFrame(ord_cats_cols_transformed,columns =ord_tf_columns)
df_train = df_train.drop(columns=ord_cats_cols, errors='ignore')
df_train = pd.concat([df_train.reset_index(drop=True), ord_cats_pd.reset_index(drop=True)], axis=1)


print(df_test.head())
print(df_train.head())



X= df_train.drop(columns=['fertilizer name','id'])
y = df_train['fertilizer name']



corr = df_train.corr()
print(corr)


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(y_true, y_pred, k=3):
    return np.mean([apk([a], p, k) for a, p in zip(y_true, y_pred)])


from sklearn.metrics import make_scorer    
custom_scorer = make_scorer(mapk)


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk_1(y_true, y_pred, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


from sklearn.metrics import make_scorer    
custom_scorer = make_scorer(mapk_1)


#learned parameters from Optuna
from xgboost import XGBClassifier
params_2 = {'learning_rate': 0.08379747466360393, 
'n_estimators': 2500, 
'max_depth': 6, 
'min_child_weight': 5, 
'subsample': 0.6759929713847056, 
'colsample_bytree': 0.5070564951479825,
'colsample_bylevel': 0.969854995161654,
'gamma': 0.007411116428546305, 
'reg_alpha': 0.4656153081306118,
'reg_lambda': 0.2481873217570749,
}
model = XGBClassifier(
**params_2,tree_method='hist',device ='cuda',n_jobs=-1
)
model.fit(X, y)




df_test = df_test.drop(columns = ['id'])


probs = model.predict_proba(df_test)  
top_k_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1] 


df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")




top_3_labels = ord_encode.inverse_transform(top_k_preds.ravel().reshape(-1, 1)).reshape(top_k_preds.shape)

submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")




