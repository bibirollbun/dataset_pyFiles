import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

from tqdm.notebook import tqdm


train = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/train/train.csv", index_col="PetID") 
test = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/test/test.csv", index_col="PetID")
ss = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/test/sample_submission.csv", index_col="PetID")

label = "AdoptionSpeed"
X_train = train.drop(label, axis=1).select_dtypes(exclude="O")
y_train = train[label]

X_test = test[X_train.columns]


valid_preds = []
test_preds = []

for train_index, valid_index in tqdm(KFold(n_splits=5, shuffle=True).split(X_train), total=5):
    Xt = X_train.iloc[train_index]
    yt = y_train.iloc[train_index]
    Xv = X_train.iloc[valid_index]
    
    model = LGBMClassifier()
    model.fit(Xt, yt)

    valid_preds.append(
        pd.Series(model.predict(Xv), index=Xv.index, name="AdoptionSpeed")
    )
    test_preds.append(
        pd.Series(model.predict(X_test), index=X_test.index, name="AdoptionSpeed")
    ) 

valid_preds = pd.concat(valid_preds)


cohen_kappa_score(y_train, valid_preds.loc[y_train.index], weights="quadratic")


pd.concat(test_preds, axis=1).median(axis=1).to_csv("submission.csv")


!head submission.csv




