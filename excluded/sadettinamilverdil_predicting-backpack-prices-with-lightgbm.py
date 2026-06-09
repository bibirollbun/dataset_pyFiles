import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import shap
import optuna
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

print(train.shape)
print(test.shape)


missing_values = train.isnull().sum() / len(train) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Missing Values Percentage in Train Set")
plt.show()

# Missing values in test dataset
missing_values = test.isnull().sum() / len(test) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Missing Values Percentage in Test Set")
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(train["Price"], bins=50, kde=True, color="royalblue")
plt.title("Price Distribution")
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(y=train["Color"], x=train["Price"], palette="viridis")
plt.xlabel("Price")
plt.show()


RMV = ["id","Price"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)

print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Price"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Price"]
    x_test = test[FEATURES].copy()

    model_lgb = LGBMRegressor(
        device="gpu", 
        objective="regression",
        verbose=-1, 
        early_stopping_rounds=25,
        max_depth=8,  
        n_estimators=2201,  
        learning_rate=0.025756006454565498,  
        num_leaves=23,  
        subsample=0.7448233049768063,  
        colsample_bytree=0.652378629178929,  
        reg_alpha=0.041519896571496503,  
        reg_lambda=0.001090930963040509,  
        min_child_weight=26
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )
    
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    pred_lgb += model_lgb.predict(x_test)

pred_lgb /= FOLDS


importances = model_lgb.feature_importances_
feature_names = FEATURES 

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"], palette="viridis")
plt.title("Feature Importance of LightGBM Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


sample["Price"] = pred_lgb
sample.to_csv("submission.csv", index=False)
sample.head()




