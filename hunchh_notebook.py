


import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


x=train.drop(['Fertilizer Name'],axis=1)
y=train['Fertilizer Name']


y


x


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


model=LabelEncoder()
y_enc=model.fit_transform(y)


#le=LabelEncoder()
#cat_cols=['Soil Type','Crop Type']
#for i in cat_cols:
#    x[i]=le.fit_transform(x[i])
#    test[i]=le.transform(test[i])


cat_cols=['Soil Type','Crop Type']
x=pd.get_dummies(x,columns=cat_cols)
test=pd.get_dummies(test,columns=cat_cols)


x.isna().sum()


X_train,X_val,y_train,y_val=train_test_split(x,y_enc,test_size=0.2,random_state=42)


#cat_features = test.select_dtypes(exclude=['number']).columns.tolist()


from lightgbm import LGBMClassifier

model_lgbm = LGBMClassifier(objective='multiclass',
                            n_estimators=500,
                            learning_rate=0.05,
                            num_iterations=500,
                            min_data_in_leaf = 5000,
                            lambda_l2 = 100,
                            verbose=0,
                            random_state=0)


model_lgbm.fit(X_train,y_train).score(X_val,y_val)


from catboost import CatBoostClassifier

model_cat = CatBoostClassifier(learning_rate=0.05,
                               boosting_type='Plain',
                               grow_policy = "Depthwise",
                               min_data_in_leaf=5000,
                               verbose=50)

model_cat.fit(X_train,y_train).score(X_val,y_val)


y_probs=model_cat.predict_proba(X_val)


test_probs=model_cat.predict_proba(test)


def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


top3_preds = get_top_k_predictions(test_probs, k=3)
top3_labels = model.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
top3_labels


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name':[' '.join(preds) for preds in top3_labels]})


submission


submission.to_csv('/kaggle/working/submission.csv',index=False)

