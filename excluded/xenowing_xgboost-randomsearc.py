# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q  matplotlib


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_dataset=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_dataset


train_dataset=train_dataset.dropna()
train_dataset


train_dataset['Fertilizer Name'].value_counts()


train_dataset.columns


cat_cols=[col for col in train_dataset if train_dataset[col].dtype=='object']
num_cols=[col for col in train_dataset if train_dataset[col].dtype!='object']
cat_cols,num_cols


train_dataset['Soil Type'].value_counts()


train_dataset['Crop Type'].value_counts()


train_dataset['Temparature'].value_counts()


train_dataset['Humidity'].value_counts()


train_dataset['Moisture'].value_counts()


train_dataset['Temparature'].value_counts().values,train_dataset['Temparature'].value_counts().index


train_dataset['Temparature'].value_counts()


Y=train_dataset['Temparature'].value_counts().index
X=train_dataset['Temparature'].value_counts().values
plt.figure(figsize=(10,12))
plt.plot(X,Y)
plt.xlabel("Count")
plt.ylabel("Temperature")
plt.title("Tmperature Distribution")
plt.show()


from sklearn.preprocessing import LabelEncoder,StandardScaler

le=LabelEncoder()
train_dataset['label']=le.fit_transform(train_dataset['Fertilizer Name'])
train_dataset['Soil Type']=le.fit_transform(train_dataset['Soil Type'])
train_dataset['Crop Type']=le.fit_transform(train_dataset['Crop Type'])
train_dataset


from sklearn.model_selection import train_test_split

features=[col for col in train_dataset.columns if col not in ['id','Fertilizer Name','label']]
X=train_dataset[features]
y=train_dataset['label']

X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

print("Train Dataset:",X_tr.shape)
print("Validation Dataset:",X_val.shape)


test_dataset=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_dataset['Soil Type']=le.fit_transform(test_dataset['Soil Type'])
test_dataset['Crop Type']=le.fit_transform(test_dataset['Crop Type'])
test_dataset


X_tr


scaler=StandardScaler()
X_tr_scaled=scaler.fit_transform(X_tr)
X_val_scaled=scaler.transform(X_val)
X_test_scaled=scaler.transform(test_dataset[features])


from xgboost import XGBClassifier,DMatrix
import xgboost as xgb

dtrain = xgb.DMatrix(X_tr_scaled, label=y_tr)
dval = xgb.DMatrix(X_val_scaled, label=y_val)

params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_tr)),
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss'
}


xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dval, 'validation')]
)


val_probas=xgb_model.predict(dval)
val_top3=np.argsort(val_probas,axis=1)[:,::-1][:,:3]
val_top3_label=le.inverse_transform(val_top3.flatten()).reshape(val_top3.shape)


def mapk(actual,predicted,k=3):
    def apk(a,p,k):
        score=0.0
        for i,pred in enumerate(p[:k]):
            if pred==a:
                score+=1.0/(i+1.0)
                break
        return score
    return np.mean([apk(a,p,k) for a ,p in zip(actual,predicted)])


y_val_labels=le.inverse_transform(y_val)
val_map3=mapk(y_val_labels,val_top3_label,k=3)
print(f"Validation MAP@3:{val_map3:.4f}")


from sklearn.metrics import make_scorer

def mapk_sklearn(y_true,y_pred_proba,k=3):
    top_k_preds=np.argsort(y_pred_proba,axis=1)[:,::-1][:,:k]
    
    y_true_labels = le.inverse_transform(y_true)
    
    top_k_labels = np.array([[le.inverse_transform([pred])[0] for pred in row] for row in top_k_preds])
    
    return mapk(y_true_labels, top_k_labels, k=k)

mapk_scorer=make_scorer(mapk_sklearn,needs_proba=True,greater_is_better=True)


xgb_skl = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    tree_method='hist',   # best splitter on GPU
    device='cuda',        # send both training & inference to GPU
    eval_metric='mlogloss',
    use_label_encoder=False,
    verbosity=1
)

param_dict = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [4, 6, 8, 10],
    'learning_rate':    [0.01, 0.06, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma':            [0, 0.1, 0.3],
    'reg_alpha':        [0, 0.1, 0.5],
    'reg_lambda':       [1, 1.5, 2]
}



from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_skl,
    param_distributions=param_dict,
    n_iter=30,
    scoring=mapk_scorer,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_tr_scaled, y_tr)


print("Best MAP@3:", random_search.best_score_)
print("Best Params:", random_search.best_params_)


best_model = random_search.best_estimator_

probas_val = best_model.predict_proba(X_val_scaled)
top3_idx  = np.argsort(probas_val, axis=1)[:, ::-1][:, :3]
top3_lbls = np.array([[le.classes_[c] for c in row] for row in top3_idx])
train_dataset['label']=le.fit_transform(train_dataset['Fertilizer Name'])
y_val_lbls = le.inverse_transform(y_val)
print(f"Validation MAP@3: {mapk(y_val_lbls, top3_lbls, k=3):.4f}")


probas_test = best_model.predict_proba(X_test_scaled)
top3_idx     = np.argsort(probas_test, axis=1)[:, ::-1][:, :3]
top3_labels  = np.array([[le.classes_[c] for c in row] for row in top3_idx])
top3_labels


sample_sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
sample_sub


sample_sub['id']=test_dataset['id']
sample_sub['Fertilizer Name']=[' '.join(preds) for preds in top3_labels]
sample_sub


sample_sub.to_csv("submission.csv",index=False)

