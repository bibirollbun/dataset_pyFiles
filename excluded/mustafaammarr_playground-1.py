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


data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
data



data.isna().sum()


data["Fertilizer Name"] = data["Fertilizer Name"].astype('category')
humidity=data[data["Fertilizer Name"]=='14-35-14'].Humidity.unique()
humidity = np.sort(humidity)
print(humidity)



data.groupby(by=['Soil Type','Crop Type'])['Crop Type'].count()


data.groupby(by='Fertilizer Name')['Moisture'].mean()


data.groupby(by='Fertilizer Name')['Temparature'].mean()


data.groupby(by='Fertilizer Name')['Nitrogen'].mean()


data.groupby(by='Fertilizer Name')['Phosphorous'].mean()



data['Soil Type'] = data['Soil Type'].astype('category')
data['Crop Type'] = data['Crop Type'].astype('category')
data['Fertilizer Name'].shape



from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer

# Define precision@3 function
def precision_at_3(y_true, y_proba):
    top3 = np.argsort(y_proba, axis=1)[:, -3:]
    return np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

# Wrap the scorer
class PrecisionAt3Scorer:
    def __init__(self, scorer_func):
        self.scorer_func = scorer_func

    def __call__(self, estimator, X, y):
        y_proba = estimator.predict_proba(X)
        return self.scorer_func(y, y_proba)

custom_scorer = PrecisionAt3Scorer(precision_at_3)


sk = StratifiedKFold(4)
y=data['Fertilizer Name']
le = LabelEncoder()
data['Soil_encoded']= le.fit_transform(data['Soil Type'])
data['Crop_encoded'] = le.fit_transform(data['Crop Type'])
y_encoded = le.fit_transform(y)
classes =le.classes_
X_encoded= data[['Soil_encoded','Crop_encoded']]
model = XGBClassifier(user_label_encoder=False,eval_metric=custom_scorer,n_jobs=-1,enable_categorical=True)
sk = StratifiedKFold(n_splits=4,random_state=2,shuffle=True)
final_scores = cross_val_score(model,X_encoded,y_encoded,cv=sk,scoring=custom_scorer)
print("MEAN: ",final_scores.mean())
print("STD",final_scores.std())
print("cross validation score: ",final_scores)


test= pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test['Soil Type'] = test['Soil Type'].astype('category')
test['Crop Type'] = test['Crop Type'].astype('category')
X= data[['Soil Type','Crop Type']]
def make_submission(model,X=X,y=y_encoded,test = test):
    model.fit(X,y)
    proba = model.predict_proba(test[["Soil Type","Crop Type"]])
    top3_indices = np.argsort(-proba, axis=1)[:, :3]
    top3_preds = np.take(classes, top3_indices)
    pred_strings = [' '.join(map(str, preds)) for preds in top3_preds]
    submission = pd.DataFrame({
        'id': test['id'],
        'Fertilizer Name': pred_strings
    })
    return submission
sub=make_submission(model,X,y_encoded,test)
print(sub)
sub.to_csv("submission.csv",index=False)


