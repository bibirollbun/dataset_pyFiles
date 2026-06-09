import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import ParameterGrid

folder =  "/kaggle/input/petfinder-adoption-prediction"
train = pd.read_csv(f"{folder}/train/train.csv")
test = pd.read_csv(f"{folder}/test/test.csv")


target_col = "AdoptionSpeed"
X = train.set_index("PetID").drop(target_col, axis=1).select_dtypes(exclude="O")
y = train.set_index("PetID")[target_col]


candidates = [
    {
        "model": LGBMClassifier,
        "params": ParameterGrid({
            "num_leaves": [31, 40, 50]  
        })
    },
    {
        "model": RandomForestClassifier,
        "params": ParameterGrid({
        "n_estimators": [100, 150, 200]  
        })
    }
]




spliter = KFold(shuffle=True)

valid_preds = []
test_preds = []

resultados = pd.Series()
score_modelos = []


resultados.loc["model"] = 0

preds = {}

for candidate in candidates:
    for params in candidate["params"]:
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

        valid_preds_params = pd.concat(valid_preds)
        score = cohen_kappa_score(y.loc[valid_preds_params.index], valid_preds_params, weights='quadratic')
        print(score)
        score_modelos.append({"model": candidate["model"](**params),
                           "score": score})
        preds[str(model)] = pd.concat(test_preds, axis=1).median(axis=1)




best_model=str(pd.DataFrame(score_modelos).set_index("model").score.idxmax())


preds[best_model].astype(int).rename(target_col).to_csv('submission.csv')


!head submission.csv


# kappa = cohen_kappa_score(y.loc[valid_predictions.index], valid_predictions, weights='quadratic')

