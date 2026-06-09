import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score,  GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

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

train["cloud_sun_intersect"] = train["cloud"] * train["sunshine"]
test["cloud_sun_intersect"] = test["cloud"] * test["sunshine"]

train["cloud_humidity_intersect"] = train["cloud"] * train["humidity"]
test["cloud_humidity_intersect"] = test["cloud"] * test["humidity"]

train["cloud_sun_intersect"] = train["cloud"] / (train["sunshine"] + 1e-3)
test["cloud_sun_intersect"] = test["cloud"] / (test["sunshine"] + 1e-3)

train["humidity_dewpoint_intersect"] = train["humidity"] * train["dewpoint"]
test["humidity_dewpoint_intersect"] = test["humidity"] * test["dewpoint"]

train["sun_wind_intersect"] = train["sunshine"] / (train["windspeed"] + 1e-3)
test["sun_wind_intersect"] = test["sunshine"] / (test["windspeed"] + 1e-3)

train["cloud_low_sun_intersect"] = train["cloud"] * train["low_sun"]
test["cloud_low_sun_intersect"] = test["cloud"] * test["low_sun"]

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


NB_model = GaussianNB().fit(X_scaled,y)


scores = cross_val_score(NB_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")


y_test_pred = NB_model.predict_proba(X_test_scaled)[:,1]


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission.to_csv("submission_naivebayes.csv", index=False)


l1_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
scores = cross_val_score(l1_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")


l1_model.fit(X_scaled , y)
y_test_pred = l1_model.predict_proba(X_test_scaled)[:,1]


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission.to_csv("submission_logreg.csv", index=False)


lr = LogisticRegression(solver="liblinear", random_state=42)

param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "class_weight": [None, "balanced"],
    "solver" : ["liblinear"]
}

grid = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=10,
    n_jobs=-1,
    verbose=0
)

grid.fit(X_scaled, y)



grid.best_params_


final_lr = LogisticRegression(solver="liblinear",penalty= "l2",class_weight = None,C=0.01, max_iter = 1000, random_state=42)
scores = cross_val_score(final_lr, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")
final_lr.fit(X_scaled,y)


y_test_pred = final_lr.predict_proba(X_test_scaled)[:, 1]

submission_lr = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})
submission_lr.to_csv("submission_logreg_final.csv", index=False)


coef_df = pd.Series(final_lr.coef_[0], index=X.columns).sort_values()

plt.figure(figsize=(12, 6))
coef_df.plot(kind='barh')
plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Weight")
plt.grid(True)
plt.show()



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



# best params :{'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 1.0}


xgb_tuned_model = XGBClassifier(
    n_estimators = 200,
    colsample_bytree = 0.8,
    gamma = 1,
    learning_rate =0.05,
    max_depth = 5,
    reg_alpha = 0.1,
    reg_lambda = 1,
    subsample = 1,
    random_state = 42
)


xgb_tuned_model.fit(X_scaled, y)
y_test_pred_xgb = xgb_tuned_model.predict_proba(X_test_scaled)[:, 1]
submission_xgb_final = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred_xgb
})
submission_xgb_final.to_csv("submission_xgb_final_model.csv", index=False)


scores = cross_val_score(xgb_tuned_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")


# Best Params (SVC): {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}


svc_model = SVC(C = 1,gamma = 0.001,kernel = "rbf",probability= True)
scores = cross_val_score(svc_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")
svc_model.fit(X_scaled, y)


y_test_pred = svc_model.predict_proba(X_test_scaled)[:, 1]

submission_lr = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})
submission_lr.to_csv("submission_svc_final.csv", index=False)


# Best Params (RF): {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}


rf_model = RandomForestClassifier(n_estimators = 300,max_depth = 10,max_features = "sqrt",min_samples_leaf=4,min_samples_split=2)
scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")
rf_model.fit(X_scaled, y)


y_test_pred = rf_model.predict_proba(X_test_scaled)[:, 1]

submission_cat = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission_cat.to_csv("submission_randomF.csv", index=False)


# best params : {'activation': 'relu','alpha': 0.01,'hidden_layer_sizes': (3, 5), 'solver': 'adam'}


mlpc = MLPClassifier(activation ='relu',alpha= 0.01,hidden_layer_sizes= (3, 5), solver = "adam")


scores = cross_val_score(mlpc, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")
mlpc.fit(X_scaled, y)


y_test_pred = mlpc.predict_proba(X_test_scaled)[:, 1]

submission_cat = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission_cat.to_csv("submission_mlpcs.csv", index=False)


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
        ('xgb', xgb_tuned_model),
        ('cat', cat_model)
    ],
    voting='soft'
)
voting.fit(X_scaled, y)



y_vote_pred = voting.predict_proba(X_test_scaled)[:, 1]


for name, model in voting.named_estimators_.items():
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    print(f"{name} top 5 possibilities from predictions: {y_pred[:5]}")



submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_voting.csv", index=False)



voting_l1_l2_cat = VotingClassifier(
    estimators=[
        ('log_l2', final_lr),
        ('cat', final_cat),
        ('l1', l1_model),
    ],
    voting='soft'
)

voting_l1_l2_cat.fit(X_scaled, y)



y_vote_pred = voting_l1_l2_cat.predict_proba(X_test_scaled)[:, 1]


submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_best_voting.csv", index=False)


voting_nb_l2 = VotingClassifier(
    estimators=[
        ('log_l2', final_lr),
        ("NaiveB",NB_model),
    ],
    voting='soft'
)

voting_nb_l2.fit(X_scaled, y)
y_vote_pred = voting_nb_l2.predict_proba(X_test_scaled)[:, 1]


submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_nb_l2_voting.csv", index=False)


voting_l1_svc_l2 = VotingClassifier(
    estimators=[
        ('log_l2', final_lr),
        ("l1",l1_model),
        ("svc",svc_model)
    ],
    voting='soft'
)

voting_l1_svc_l2.fit(X_scaled, y)
y_vote_pred = voting_l1_svc_l2.predict_proba(X_test_scaled)[:, 1]


submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_l1_l2_svc_voting.csv", index=False)


voting_xgb_svc_l2_ann = VotingClassifier(
    estimators=[
        ('log_l2', final_lr),
        ("ANN",mlpc),
        ("svc",svc_model),
        ("xgb",xgb_tuned_model),
    ],
    voting='soft'
)

voting_xgb_svc_l2_ann.fit(X_scaled, y)
y_vote_pred = voting_xgb_svc_l2_ann.predict_proba(X_test_scaled)[:, 1]


submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_xgb_l2_svc_ann_voting.csv", index=False)


voting_l2_ann = VotingClassifier(
    estimators=[
        ('log_l2', final_lr),
        ("ANN",mlpc),
    ],
    voting='soft'
)

voting_l2_ann.fit(X_scaled, y)
y_vote_pred = voting_l2_ann.predict_proba(X_test_scaled)[:, 1]


submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_l2_ann_voting.csv", index=False)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled) 
X_clustered = X_scaled.copy()
X_clustered = pd.DataFrame(X_scaled)
X_clustered["Cluster"] = clusters   



mlpc = MLPClassifier(hidden_layer_sizes=(3, 5), activation='relu', solver='adam', alpha=0.01, max_iter=1000, random_state=42)
mlpc.fit(X_clustered.values, y)


scores = cross_val_score(mlpc, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f}")


test_clusters = kmeans.predict(X_test_scaled)
X_test_clustered = pd.DataFrame(X_test_scaled)
X_test_clustered["Cluster"] = test_clusters
y_vote_pred = mlpc.predict_proba(X_test_clustered.values)[:, 1]



submission_vote = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_vote_pred
})

submission_vote.to_csv("submission_ann_kmeans.csv", index=False)




