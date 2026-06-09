import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import shap
import optuna
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder 

from sklearn.model_selection import KFold
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train = pd.concat([train,train_extra],axis=0,ignore_index=True)
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


TE = TargetEncoder(n_folds=5, smooth=20, split_method='random', stat='mean')
train["Weight_Capacity_TE"] = TE.fit_transform(train["Weight Capacity (kg)"], train["Price"])
test["Weight_Capacity_TE"] = TE.transform(test["Weight Capacity (kg)"])

FEATURES.append("Weight_Capacity_TE")


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Price"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Price"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        enable_categorical=True,
        early_stopping_rounds=25,
        max_depth=6,  
        n_estimators=1354,  
        learning_rate=0.03765417587724906,  
        subsample=0.9626163805632845,  
        colsample_bytree=0.9326263462469375,  
        reg_alpha=0.45630715547132716,  
        reg_lambda=0.08886115975900113,  
        min_child_weight=27
    )


    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=200 
    )

    oof_xgb[test_index] = model_xgb.predict(x_valid)
    pred_xgb += model_xgb.predict(x_test)

pred_xgb /= FOLDS
rmse = np.sqrt(mean_squared_error(train['Price'], oof_xgb))
print(rmse)


explainer = shap.TreeExplainer(model_xgb, feature_perturbation="tree_path_dependent", model_output="raw")
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_test)


sample["Price"] = pred_xgb
sample.to_csv("submission.csv", index=False)
sample.head()

