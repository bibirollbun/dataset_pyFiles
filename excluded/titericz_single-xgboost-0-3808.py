import pandas as pd
import numpy as np
import random
import os
import gc
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv( "/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train.shape, test.shape, original.shape


NFOLDS = 10
FOLDS = np.zeros(len(train))
skf = StratifiedKFold(n_splits=NFOLDS, random_state=42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(train, train['Fertilizer Name'])):
    FOLDS[test_index]=i
train['fold'] = FOLDS

FOLDS = np.zeros(len(original))
skf = StratifiedKFold(n_splits=NFOLDS, random_state=42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(original, original['Fertilizer Name'])):
    FOLDS[test_index]=i
original['fold'] = FOLDS


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

train = train.rename(columns={'Temparature': 'Temperature'})
test  = test.rename(columns={'Temparature': 'Temperature'})
original  = original.rename(columns={'Temparature': 'Temperature'})

cat_cols = [col for col in test.select_dtypes(include=['object', 'category']).columns]
for col in cat_cols:
    label_enc = LabelEncoder()
    train[col] = label_enc.fit_transform(train[col])
    test[col] = label_enc.transform(test[col])
    original[col] = label_enc.transform(original[col])

target_label_enc = LabelEncoder()
train["Fertilizer Name"] = target_label_enc.fit_transform(train["Fertilizer Name"])
original["Fertilizer Name"] = target_label_enc.transform(original["Fertilizer Name"])


train['comp_data'] = 0
test['comp_data'] = 1
original['comp_data'] = 2
raw = pd.concat([train, test, original]).reset_index(drop=True)
print(raw.shape)


numerical_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in numerical_features:
    raw[col+'_cat'] =  raw[col].astype(str).astype('category')

numerical_features = ['Soil Type', 'Crop Type']
for col in numerical_features:
    raw[col] =  raw[col].astype(str).astype('category')

train = raw.loc[raw['comp_data']==0].reset_index(drop=True)
test = raw.loc[raw['comp_data']==1].reset_index(drop=True)
original = raw.loc[raw['comp_data']==2].reset_index(drop=True)
del raw
test['comp_data'] = 0


train["NxPxP_1"]   = train['Nitrogen']*43*20   +train['Potassium']*43   +train['Phosphorous']
test["NxPxP_1"]    = test['Nitrogen']*43*20    +test['Potassium']*43    +test['Phosphorous']
original["NxPxP_1"]= original['Nitrogen']*43*20+original['Potassium']*43+original['Phosphorous']

train["NxPxP_2"]   = train['Nitrogen']   +train['Potassium']*39*43   +train['Phosphorous']*39
test["NxPxP_2"]    = test['Nitrogen']    +test['Potassium']*39*43    +test['Phosphorous']*39
original["NxPxP_2"]= original['Nitrogen']+original['Potassium']*39*43+original['Phosphorous']*39

train["NxPxP_3"]   = train['Nitrogen']*20   +train['Potassium']   +train['Phosphorous']*39*20
test["NxPxP_3"]    = test['Nitrogen']*20    +test['Potassium']    +test['Phosphorous']*39*20
original["NxPxP_3"]= original['Nitrogen']*20+original['Potassium']+original['Phosphorous']*39*20

train["SoilxCrop_1"] = train['Soil Type'].astype(int).values+train['Crop Type'].astype(int).values*5
test["SoilxCrop_1"] = test['Soil Type'].astype(int).values+test['Crop Type'].astype(int).values*5
original["SoilxCrop_1"] = original['Soil Type'].astype(int).values+original['Crop Type'].astype(int).values*5

train["SoilxCrop_2"] = train['Soil Type'].astype(int).values*11+train['Crop Type'].astype(int).values
test["SoilxCrop_2"] = test['Soil Type'].astype(int).values*11+test['Crop Type'].astype(int).values
original["SoilxCrop_2"] = original['Soil Type'].astype(int).values*11+original['Crop Type'].astype(int).values


def map3(predicted: np.ndarray, labels: np.ndarray) -> float:
    pred = np.argsort(-1*predicted, 1)
    
    p0 = (labels == pred[:, 0])
    p1 = (labels == pred[:, 1])
    p2 = (labels == pred[:, 2])
    
    return float(np.mean(p0/1 + p1/2 + p2/3))


features = [f for f in train.columns if f not in ['fold', 'target', 'grp', 'Fertilizer Name', 'ids']]
train[features].nunique()


params = {
    "n_estimators": 100000,
    "learning_rate": 0.02,
    "num_class": 7,
    "max_depth": 8,
    "min_child_weight": 0.00023161052323600594,
    "subsample": 0.74,
    "colsample_bytree": 0.39,
    "gamma": 0.48,
    "reg_alpha":0.027,
    "reg_lambda": 0.000255,
    'max_bin': 64,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",  # Faster and memory-efficient
    "enable_categorical": True,
    "early_stopping_rounds": 100,
    "device" :"cuda"
}

NBAGS = 1
dtest = xgb.DMatrix(test[features], enable_categorical=True)
ytrain = np.zeros( (len(train), 7) )
ytest = np.zeros( (len(test), 7) )
for fold in range(NFOLDS):
    print(fold)
    ind_train = (train['fold'] != fold)
    ind_valid = (train['fold'] == fold)

    for bag in range(NBAGS):
        params['seed'] = (bag+1)*11
        params['learning_rate'] = np.random.normal(0.005, 0.001)
        params['colsample_bytree'] = np.random.normal(0.39, 0.005)
        params['subsample'] = np.random.normal(0.74, 0.005)

        K2 = np.random.randint(5, 8)
        X = pd.concat( [train.loc[ind_train, features]] + [original[features]]*K2, axis=0, ignore_index=True)
        y = pd.concat( [train.loc[ind_train, 'Fertilizer Name']] + [original['Fertilizer Name']]*K2, axis=0, ignore_index=True)

        X_valid = train.loc[ind_valid, features].reset_index(drop=True)
        y_valid = train.loc[ind_valid, 'Fertilizer Name'].reset_index(drop=True)
        
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        model = xgb.train(
            params, 
            dtrain, 
            99999, 
            evals=[(dvalid, "validation")],
            verbose_eval=2000,
            callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
        )
        ytrain[ind_valid] += (model.predict(dvalid) / NBAGS)
        print(fold, bag, map3(ytrain[ind_valid], y_valid.values))
        ytest += (model.predict(dtest) / (NFOLDS*NBAGS))
    print()

score = map3(ytrain, train['Fertilizer Name'].values)
print(score)


top_3_preds = np.argsort(ytest, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv(f'submission_{score:.4f}.csv', index=False)
print(f"✅ Submission file saved as 'submission_{score:.4f}.csv'")













