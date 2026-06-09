!pip install edazer


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from edazer import Edazer
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


ROOT_PATH = "/kaggle/input/playground-series-s5e7/"
df = pd.read_csv(ROOT_PATH + "train.csv")


df.head()


df_eda = Edazer(df, backend= "pandas")
df_eda.summarize_df()


def preproceessing(df):
    df.drop(columns=["id"], inplace=True)
    
    yes_no_map = {"Yes": 1, "No": 0}
    for col in ["Drained_after_socializing", "Stage_fear"]:
        df[col] = df[col].map(yes_no_map)
        
    simple_imputer = SimpleImputer(strategy="median").fit(X)
    df = pd.DataFrame(simple_imputer.transform(df), columns=df.columns)
    
    df_eda = Edazer(df= df, backend= "pandas")
    float_cols = df_eda.cols_with_dtype(dtypes=["float"]) # using edazer
    for col in float_cols:
        df[col] = df[col].astype("int32", errors="ignore")
    
    return df





X, y  = df.drop(columns=["Personality"]), df["Personality"].copy()


personality_map = {"Extrovert": 1, "Introvert": 0}
y = y.map(personality_map)


1893/X.shape[0]*100


X = preproceessing(X)


y.value_counts().rename(index={1: "extrovert", 0: "introvert"})


4825/(4825 + 13699) *100


X_train, X_test, y_train, y_test = train_test_split(X, y)


from xgboost import XGBClassifier
xgb_params = {
        "n_estimators": 60,
        "max_depth": 3,    
        "learning_rate": 1e-1, 
        "eval_metric": "error"
        }

xgb_clf = XGBClassifier(**xgb_params).fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)
xgb_clf_test_score = accuracy_score(y_test, xgb_preds)
xgb_clf_test_score





cross_val_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring="accuracy")
xgb_clf_cv_score = cross_val_scores.mean()
xgb_clf_cv_score


feature_importance_df = pd.DataFrame({"feature": xgb_clf.feature_names_in_,
              "importance": xgb_clf.feature_importances_}).sort_values("importance", ascending=False)
plt.figure(figsize=(6,4))
plt.barh(y= feature_importance_df["feature"], width= feature_importance_df["importance"])
plt.xlabel("importance")
plt.ylabel("feature names")
plt.title("XGB feature importance")
plt.gca().invert_yaxis()  
plt.show()


from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(xgb_clf, X_train, y_train, cv= 5, n_jobs=-1, random_state=42,\
                                                        scoring="accuracy")


train_scores


plt.plot(train_sizes, train_scores.mean(axis=1), 'g--', label="train accuracy")
plt.plot(train_sizes, test_scores.mean(axis=1), 'r:', label="test accuracy")
plt.xlabel("train size")
plt.ylabel("accuracy")
plt.legend(loc="upper right")
plt.show()


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(tol=1e-2, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_test_score = accuracy_score(y_test, log_reg.predict(X_test))
log_reg_test_score


log_reg_cv_score = cross_val_score(log_reg, X_train, y_train, scoring="accuracy", cv=5).mean()
log_reg_cv_score


from sklearn.ensemble import VotingClassifier
estimators=[('xgb_clf', XGBClassifier(**xgb_params)), ('log_reg', LogisticRegression(tol=1e-2, random_state=42))]
voting_clf= VotingClassifier(estimators=estimators, voting="hard")


voting_clf.fit(X_train, y_train)


voting_clf_test_score = accuracy_score(y_test, voting_clf.predict(X_test))
voting_clf_test_score


voting_clf_cv_score= cross_val_score(voting_clf, X_train, y_train, cv=5, scoring="accuracy").mean()
voting_clf_cv_score


pd.DataFrame({"model_name": ["xgb_clf", "log_reg", "voting_clf"],
 "cross_val_score": [xgb_clf_cv_score, log_reg_cv_score, voting_clf_cv_score],
 "test_score": [xgb_clf_test_score, log_reg_test_score, voting_clf_test_score]
})


test_df = pd.read_csv(ROOT_PATH+"test.csv")
test = preproceessing(test_df)


models = [log_reg, xgb_clf, voting_clf]
for idx, model in enumerate(models):
    model_idx_map ={0:"log_reg", 1: "xgb", 2:"voting"}
    
    preds = model.predict(test)
    preds_conv = np.where(preds==1,"Extrovert", "Introvert")
    
    data = np.c_[range(18524, 24699), preds_conv]
    pd.DataFrame(data).rename(columns={0: "id", 1:"Personality"}).to_csv(model_idx_map[idx]+"pred1.csv", index=False)

