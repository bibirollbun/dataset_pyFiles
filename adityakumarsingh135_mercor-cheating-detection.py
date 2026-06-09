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


import pandas as pd
import json 

#----------
# Load Data 
#----------
train = pd.read_csv('/kaggle/input/mercor-cheating-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-cheating-detection/test.csv')

print("train shape", train.shape)
print("test shape", test.shape)

#----------------------
#Load Meta Data
#----------------------
with open('/kaggle/input/mercor-cheating-detection/feature_metadata.json') as f:
    meta = json.load(f)

print(meta)

features = list(meta.keys())

print(features)


train.head()


features = [f for f in features if f in train.columns]
# it means we are taking only that features that are present in train columns


train["label_clean"] = train["is_cheating"]

train["label_clean"] = train["label_clean"].fillna(1-train["high_conf_clean"])


# now the values that are missing we will replace it with the median value
for f in features:
    if train[f].dtype != "object":
        train[f] = train[f].fillna(train[f].median())
        test[f] = test[f].fillna(train[f].median())

    else:
        train[f] = train[f].fillna(train[f].mode()[0])
        test[f] = test[f].fillna(test[f].mode()[0])


X = train[features]
y = train["label_clean"].astype(int)


from xgboost import XGBClassifier 
from catboost import CatBoostClassifier 
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline 


# splitting into training/testing data 
X = train[features]
y = train["label_clean"].astype(int)


models = {
    "catboost": CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
        l2_leaf_reg=5
    ),
    "XGBoost": XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        colsample_bytree=0.3,
        subsample=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_estimators=10000,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device="cuda"
    )
}

#ensembling the models
models["ensemble_all"] = VotingClassifier(
    estimators=[
        # ("LightGBM", models["LightGBM"]),
        ("CatBoost", models["catboost"]),
        ("XGBoost", models["XGBoost"])
    ],
    voting = 'soft',
    weights=[1,1]
)


# cv 
KFold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 42 )
cv_results={}
scoring= {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc"
}

for name, model in models.items():
    print(name)


    #CV 
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=KFold,
        scoring= scoring,
        n_jobs = -1
    )

    cv_results[name] = {metric: np.mean(scores) for metric,scores in cv_scores.items() if "test_" in metric}

    print(f"Accuracy:  {cv_results[name]['test_Accuracy']:.4f}")
    print(f"Precision: {cv_results[name]['test_Precision']:.4f}")
    print(f"Recall:    {cv_results[name]['test_Recall']:.4f}")
    print(f"F1-score:  {cv_results[name]['test_F1']:.4f}")
    print(f"ROC-AUC:   {cv_results[name]['test_ROC_AUC']:.4f}")


results_df = pd.DataFrame({
    model: {
        "Accuracy": cv_results[model]["test_Accuracy"],
        "Precision": cv_results[model]["test_Precision"],
        "Recall": cv_results[model]["test_Recall"],
        "F1": cv_results[model]["test_F1"],
        "ROC_AUC": cv_results[model]["test_ROC_AUC"]
    } for model in cv_results.keys()
}).T.round(4)


print(results_df)


best_model_name = results_df["ROC_AUC"].idxmax()
best_model = models[best_model_name]

print(best_model_name)

best_model.fit(X, y)


final_pred = best_model.predict_proba(test[features])[:,1]


sample = pd.read_csv("/kaggle/input/mercor-cheating-detection/sample_submission.csv")
sample["prediction"] = final_pred
sample.to_csv("submission.csv", index=False)




