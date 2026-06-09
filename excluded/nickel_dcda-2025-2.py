import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score


folder =  "/kaggle/input/petfinder-adoption-prediction"
train = pd.read_csv(f"{folder}/train/train.csv")
test = pd.read_csv(f"{folder}/test/test.csv")
test["NewName"] = test["Name"].fillna("").str.len()


target_col = "AdoptionSpeed"
train["NewName"] = train["Name"].fillna("").str.len()
X = train.set_index("PetID").drop(target_col, axis=1).select_dtypes(exclude="O")
y = train.set_index("PetID")[target_col]


a.sum(axis=1)


a = pd.crosstab(pd.qcut(train["NewName"], 10, duplicates="drop"), train[target_col])
a 


candidates = [
    {
        "model": LGBMClassifier,
        "params": [
           
            {
                "num_leaves": 50
            },
             {
                "num_leaves": 100
            }
        ]
    },
    {
        "model": RandomForestClassifier,
        "params": [{}]
    }
]


spliter = KFold(shuffle=True)


resultados = pd.Series()
preds = {}
for candidate in candidates:
    for params in candidate["params"]:
        valid_preds = []
        test_preds = []
        for train_idx, valid_idx in spliter.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            model = candidate["model"](**params)
            model.fit(X_train, y_train)
        
            valid_preds.append(pd.Series(
                model.predict(X_valid),
                index=X_valid.index,
                name=target_col
            ))
        
            test_preds.append(pd.Series(
                model.predict(test.set_index("PetID")[X.columns]),
                index=test.PetID,
                name=target_col
            ))
                
        valid_preds = pd.concat(valid_preds)
        resultados.loc[str(model)] = cohen_kappa_score(y.loc[valid_preds.index], valid_preds, weights='quadratic')
        preds[str(model)] = pd.concat(test_preds, axis=1).median(axis=1)



resultados


preds[resultados.idxmax()].astype(int).rename(target_col).to_csv("submission.csv")


!head submission.csv


# kappa = cohen_kappa_score(y.loc[valid_predictions.index], valid_predictions, weights='quadratic')

