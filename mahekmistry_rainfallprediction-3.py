import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score,  GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint
from sklearn.ensemble import VotingClassifier


from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


train.describe().T


test.describe().T


train.isnull().sum()


test.isnull().sum()


train["rainfall"].value_counts()


plt.figure(figsize =(14,7))
sns.boxplot(train)
plt.show()


corr = train.corr()
plt.figure(figsize =(12,8))
sns.heatmap(corr,cmap ="coolwarm")
plt.show()


def get_season(day):
    if 80 <= day <= 171:
        return "spring"
    elif 172 <= day <= 263:
        return "summer"
    elif 264 <= day <= 354:
        return "fall"
    else:
        return "winter"
        
train["season"] = train["day"].apply(get_season)
test["season"] = test["day"].apply(get_season)

train["temp_range"] = train["maxtemp"] - train["mintemp"]
test["temp_range"] = test["maxtemp"] - test["mintemp"]

train["dew_humidity_ratio"] = train["dewpoint"] / (train["humidity"] + 1e-5)
test["dew_humidity_ratio"] = test["dewpoint"] / (test["humidity"] + 1e-5)

train["temp_dew_diff"] = train["temparature"] - train["dewpoint"]
test["temp_dew_diff"] = test["temparature"] - test["dewpoint"]

train["cloud_sun_ratio"] = train["cloud"] / (train["sunshine"] + 1e-5)
test["cloud_sun_ratio"] = test["cloud"] / (test["sunshine"] + 1e-5)

train["low_sun"] = (train["sunshine"] < 1).astype(int)
test["low_sun"] = (test["sunshine"] < 1).astype(int)

train["cloud_humidity"] = train["humidity"] * train["cloud"]
test["cloud_humidity"] = test["humidity"] * test["cloud"]

train["temp_humidity"] = train["humidity"] * train["temp_dew_diff"]
test["temp_humidity"] = test["humidity"] * test["temp_dew_diff"]

season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}

train["season_num"] = train["season"].map(season_map)
test["season_num"] = test["season"].map(season_map)

train["cloud_sun_season"] = train["cloud_sun_ratio"] * train["season_num"]
test["cloud_sun_season"] = test["cloud_sun_ratio"] * test["season_num"]

bool_cols = train.select_dtypes(include='bool').columns

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)
    
train = train.drop(["season"],axis = 1)
test = test.drop(["season"], axis = 1)


test["winddirection"].fillna(test["winddirection"].mean(), inplace=True)


X = train.drop(["id", "rainfall"], axis=1)
y = train["rainfall"]

X_test = test.drop(["id"],axis = 1)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")


model.fit(X_scaled , y)
y_test_pred = model.predict_proba(X_test_scaled)[:,1]


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission.to_csv("submission_logreg.csv", index=False)


model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)


model.fit(X_scaled ,y)
y_test_pred_xgb = model.predict_proba(X_test_scaled)[:,1]
submission_xgb = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred_xgb
})
submission_xgb.to_csv("submission_xgb.csv", index=False)



params = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}


xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)


xgb_model.fit(X_scaled, y)
y_test_pred_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

submission_xgb_final = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred_xgb
})
submission_xgb_final.to_csv("submission_xgb_final.csv", index=False)


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_scaled, y)


cat_model = CatBoostClassifier(
    iterations = 300,
    depth = 4,
    learning_rate= 0.03,
    verbose = 0,
    random_state = 42
)
cat_model.fit(X_scaled,y)


# Best AUC: 0.8954545454545455
# Best Params: {'bagging_temperature': 0.2, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 7, 'learning_rate': 0.01, 'random_strength': 2}


final_cat = CatBoostClassifier(
    bagging_temperature=0.2,
    depth=6,
    iterations=700,
    l2_leaf_reg=7,
    learning_rate=0.03,
    random_strength=2,  
    eval_metric="AUC",
    verbose=0,
    random_seed=42
)
final_cat.fit(X_scaled,y)


y_test_pred = final_cat.predict_proba(X_test_scaled)[:, 1]

submission_cat = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission_cat.to_csv("submission_catboost.csv", index=False)



voting = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('cat', cat_model)
    ],
    voting='soft'
)
voting.fit(X_scaled, y)



y_vote_pred = voting.predict_proba(X_test_scaled)[:, 1]


for name, model in voting.named_estimators_.items():
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    print(f"{name} top 5 possibilities from predictions: {y_pred[:5]}")



voting = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('cat', cat_model)],
    voting='soft'
)



submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_voting.csv", index=False)





