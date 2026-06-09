import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

folder =  "/kaggle/input/petfinder-adoption-prediction"
train = pd.read_csv(f"{folder}/train/train.csv")
test = pd.read_csv(f"{folder}/test/test.csv")


target_col = "AdoptionSpeed"
X = train.set_index("PetID").drop(target_col, axis=1).select_dtypes(exclude="O")
y = train.set_index("PetID")[target_col]


spliter = KFold()
spliter


all_preds = []
all_test_preds = []
for train_idx, valid_idx in spliter.split(X):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    all_preds.append(pd.Series(
        model.predict(X_valid),
        index=valid_idx,
        name=target_col
    ))

    all_test_preds.append(pd.Series(
        model.predict(test.set_index("PetID")[X.columns]),
        index=test.PetID,
        name=target_col
    ))
    
all_preds = pd.concat(all_preds)
all_test_preds = pd.concat(all_test_preds)
kappa = cohen_kappa_score(y.loc[all_preds.index], all_preds, weights='quadratic')
print(kappa)


train_idx


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)


model= LGBMClassifier()
model.fit(X_train,y_train)








all_preds = []
all_test_preds = []
for train_idx, valid_idx in spliter.split(X):
    X_train = ?
    y_train = ?

    X_valid = ?
    y_valid = ?
    
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    preds_valid = pd.Series(
        model.predict(?),
        index=?,
        name=target_col
    )


preds = pd.Series(
    model.predict(test.set_index("PetID")[X.columns]),
    index=test.PetID,
    name=target_col
)
preds


preds.to_csv("submission.csv")


!head submission.csv




