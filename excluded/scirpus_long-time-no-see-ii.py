import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error
from category_encoders.cat_boost import CatBoostEncoder


train = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv')
train = train.set_index("id")
train["month"] = pd.to_datetime(train.harvest_date).dt.month
train["doy"] = pd.to_datetime(train.harvest_date).dt.day_of_year
train["dow"] = pd.to_datetime(train.harvest_date).dt.day_of_week
del train["harvest_date"]
del train["field_id"]
column_to_move = train.pop("yield_tpha")
train["yield_tpha"] = column_to_move
print(train.shape[0])
test = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv')
test = test.set_index("id")
test["month"] = pd.to_datetime(test.harvest_date).dt.month
test["doy"] = pd.to_datetime(test.harvest_date).dt.day_of_year
test["dow"] = pd.to_datetime(test.harvest_date).dt.day_of_week
del test["harvest_date"]
del test["field_id"]
print(test.shape[0])


cats = []
for c in train.columns:
    if(len(train[c].unique())<32):
        cats.append(c)
        train[cats] = train[cats].astype('category')
        test[cats] = test[cats].astype('category')
print(cats)    
numerics = list(set(train.columns[:-1]).difference(set(cats)))+list(['yield_tpha'])
print(numerics)    


cb = CatBoostEncoder()
cbtrain = cb.fit_transform(train.loc[:,cats],train.yield_tpha)
cbtest = cb.transform(test.loc[:,cats])
cbtrain = pd.concat([cbtrain,train.loc[:,numerics]],axis=1)
cbtest = pd.concat([cbtest,test.loc[:,numerics[:-1]]],axis=1)


#Evolved Code ...
def GP(data):
    # Extract columns as Series (vectorized)
    pu = data["pesticide_usage"]
    fa = data["fertilizer_amount"]
    tr = data["total_rainfall"]

    A = 69.784576416015625
    B = 77.584518432617188
    
    # Fully vectorized expression
    result = (
        4.5
        + (fa - 2 * pu) / (A + pu)
        - 2 * (B + 2 * pu) / (B + tr)
    )
    
    return result


cbtest['yield_tpha'] = GP(cbtest)
cbtest[['yield_tpha']].to_csv('gpsubmission.csv')

