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


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb


train_Data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_Data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")



print(f"train shape:{train_Data.shape}")
print(f"test shape: {test_Data.shape}")


test_Data.head()




train_Data.head()


y = train_Data["Fertilizer Name"]
X = train_Data.drop(["id","Fertilizer Name"],axis=1)


y.head()



labelencoder = LabelEncoder()
target = labelencoder.fit_transform(y)


print(target)


X.head()


categorical = X[["Soil Type","Crop Type"]]
categorical.head()


OrdEnc = OrdinalEncoder()
converted = OrdEnc.fit_transform(categorical)
print(converted)


X[["Soil Type","Crop Type"]] = converted


X.head()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,target,test_size= 0.2,random_state = 42)



len(np.unique(y_train))


def train_xgb(X_train,X_test,y_train,y_test):
    train = xgb.DMatrix(X_train,label = y_train,enable_categorical=True)
    validation = xgb.DMatrix(X_test, label = y_test,enable_categorical=True)
    params = {
            "num_class": len(np.unique(y_train)),
            "objective": "multi:softprob",
            "max_depth":"10",
            "learning_rate": 0.01,
            "device": "gpu",
            "seed":42
        }
    model = xgb.train(
        params,
        train,
        num_boost_round=600,
        evals=[(train,'train'),(validation,"validation")],
        early_stopping_rounds=50,
        verbose_eval=50,
        
    )
    return model
    


model = train_xgb(X_train,X_test,y_train,y_test)


cat_cols = test_Data[['Soil Type','Crop Type']]
enc_cat_cols = OrdEnc.fit_transform(cat_cols)
enc_cat_cols


subdata = test_Data.drop('id',axis=1)
subdata[['Crop Type','Soil Type']] = enc_cat_cols
subdata.head()


xgb_subdata = xgb.DMatrix(subdata,enable_categorical=True)


prediction = model.predict(xgb_subdata)



pred = np.argsort(prediction,axis = 1)[:,-3:][:,::-1]
pred_transform = labelencoder.inverse_transform(pred.flatten()).reshape(pred.shape)
labels = [' '.join(labels) for labels in pred_transform]


pd.DataFrame(labels).head()


ids = test_Data['id']
submission = pd.DataFrame(ids)
submission['Fertilizer Name'] = pd.DataFrame(labels)
submission.head()


submission.to_csv('submission.csv',index=False)




