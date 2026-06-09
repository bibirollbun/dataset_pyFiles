import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import base
from sklearn.model_selection import KFold
from tqdm import  tqdm
from scipy.stats import uniform, randint
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
import seaborn as sns

from sklearn.ensemble import (
    StackingClassifier,
    GradientBoostingClassifier,

)

import os
import joblib

from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV



class KFoldTargetEncoderTrain(base.BaseEstimator,
                               base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = True, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)
                                     [self.targetName].mean())
            X[col_mean_name] = X[col_mean_name].fillna(mean_of_target)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    
                   np.corrcoef(X[self.targetName].values,
                               encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        global_mean = self.train[self.encodedName].mean()
        mean_df = (
            self.train
            .groupby(self.colNames)[self.encodedName]
            .mean()
            .rename(self.encodedName)
        )
        X = X.copy()
        X[self.encodedName] = X[self.colNames].map(mean_df)
        X[self.encodedName] = X[self.encodedName].fillna(global_mean).astype(float)
        return X
    




def preprocess_main(data):
    data['Age^2'] = data['Age'] ** 2
    data = data.drop(['CustomerId', 'Surname', 'id', 'GeoGender'], axis=1, errors='ignore')
    data['CreditScore_Balance'] = data['CreditScore'] * data['Balance']

    data = pd.get_dummies(data, columns=["Geography", "Gender"], drop_first=True)
    
    data.columns = data.columns.astype(str).str.replace(r"[\[\]\<\>\(\),]", '_', regex=True)
    return data


def preprocess_label(data):
    kf_surname = KFoldTargetEncoderTrain(
        colnames='Surname',
        targetName='Exited',
        n_fold=5,
        verbosity=True,
        discardOriginal_col=False
    )
    data = kf_surname.transform(data)

 

    return data
def preprocess(data):
    data = preprocess_main(data)
    y = data["Exited"]
    X = data.drop("Exited", axis=1) 
    return X, y
def preprocess_test(data, train):

    kf_surname_test = KFoldTargetEncoderTest(
        train=train,
        colNames='Surname',
        encodedName='Surname_Kfold_Target_Enc'
    )
    data = kf_surname_test.transform(data)
    data = preprocess_main(data)
    return data


def train_models(X_train, X_test, y_train, y_test, random_state=0):
    cv_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=random_state)

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 7),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.8, 0.2),
        'colsample_bytree': uniform(0.8, 0.2),
        'reg_alpha': [0, 1],
        'reg_lambda': [1, 5, 10]
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        verbosity=1,
        random_state=random_state,
        
    )

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        scoring='roc_auc',
        cv=cv_split,
        n_iter=30,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
        random_state=random_state
        
    )

    random_search.fit(X_train, y_train)

    print("Best parameters:", random_search.best_params_)
    print("Best ROC AUC: {:.2f}".format(random_search.best_score_ * 100))
    os.makedirs('/kaggle/working', exist_ok=True)
    name = "xgb_final"
    model_filename = os.path.join('/kaggle/working', f"{name.replace(' ', '_')}.pkl")
    joblib.dump(random_search.best_estimator_, model_filename)
    print(f"ğŸ’¾ Model saved to: {model_filename}")
    return name



from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
def preprocess_surname(data):
    preds_csv_path = '/kaggle/input/surname-predictions/surname_predictions.csv'
    preds = pd.read_csv(preds_csv_path, header=None, names=["Surname", "raw_pred"])

    preds["pred_id"] = preds["raw_pred"].str.extract(r"\((\d+),")[0]
    preds = preds[["Surname", "pred_id"]]
    data['Surname_Coutry'] = pd.NA
    preds_dict = preds.set_index('Surname')["pred_id"].to_dict()
    for i in range(len(data)):
        data.at[i, "Surname_Coutry"] = int(preds_dict[data.at[i, "Surname"]])
    data["Surname_Coutry"] = data["Surname_Coutry"].astype('int')
    data.info()
    return data

data = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")
data_test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
data = preprocess_surname(data)
data_test = preprocess_surname(data_test)
data_with_label = preprocess_label(data)
X,y = preprocess(data_with_label)
test_X = preprocess_test(data_test,data_with_label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

name = train_models(X, X_test, y, y_test)


model = joblib.load(f"/kaggle/working/{name}.pkl")

y_proba = model.predict_proba(test_X)[:, 1]


output = pd.DataFrame({
    "id": data_test["id"],
    "Exited": y_proba
})
output.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… Saved predictions to predictions.csv")

