import os
import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_df.head()


labels = "Personality"

categorical_cols = train_df.select_dtypes(exclude=['number']).columns.tolist()
categorical_cols.remove(labels)

numeric_cols = train_df.select_dtypes(include='number').columns.tolist()


for col in numeric_cols:
    col_max = train_df[col].max()
    col_min = train_df[col].min()
    col_avg = train_df[col].mean()
    print(f"Details of {col}")
    print(f"Max: {col_max}")
    print(f"Min: {col_min}")
    print(f"Avg: {col_avg}")
    print()


import numpy as np

count_introverts = (train_df["Personality"] == "Introvert").sum()
count_extroverts = (train_df["Personality"] == "Extrovert").sum()
nan_personalities = (train_df["Personality"] == np.nan).sum()


import plotly.express as px

fig = px.bar(x=["Introverts", "Extroverts", "NaN"], y=[count_introverts, count_extroverts, nan_personalities], labels={"x": "Category", "y": "Count"})
fig.show()


def feature_engineer(X):
    X["Social_engagement_score"] = X["Social_event_attendance"] + X["Going_outside"] + X["Friends_circle_size"] + X["Post_frequency"] + X["Stage_fear_No"] + X["Drained_after_socializing_No"]
    X["Introvert_score"] = (X["Time_spent_Alone"] + X["Stage_fear_Yes"] + X["Drained_after_socializing_Yes"]) - (X["Social_event_attendance"] + X["Friends_circle_size"] + X["Stage_fear_No"] + X["Drained_after_socializing_No"])
    return X


train_id_col = train_df["id"].copy()
tmp_ = train_df.columns.tolist()
tmp_.remove(labels)
tmp_.remove("id")

X = train_df[tmp_]
del tmp_

X = pd.get_dummies(X, columns=categorical_cols)
X = X.fillna(X.mean())

X = feature_engineer(X)

y = train_df[labels].map({'Introvert': 0, 'Extrovert': 1})

features = X.columns.tolist()


X.head()


y.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb


xgb_params_train = {
    "objective": "binary:logistic",
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.008,
    "n_estimators": 15_000,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "reg_alpha": 0.08,
    "reg_lambda": 0.3,
    "early_stopping_rounds": 50 # this is special here
}

xgb_params_prod = {
    "objective": "binary:logistic",
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.008,
    "n_estimators": 15_000,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "reg_alpha": 0.08,
    "reg_lambda": 0.3,
}


def cross_validation(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    features = X.columns.tolist()
    scores = []
    
    for n_fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        mm_scaler = MinMaxScaler()
        
        X_train = pd.DataFrame(mm_scaler.fit_transform(X_train), columns=mm_scaler.get_feature_names_out())
        X_val = pd.DataFrame(mm_scaler.transform(X_val), columns=mm_scaler.get_feature_names_out())

        model = xgb.XGBClassifier(**xgb_params_train)

        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)

        acc_score = accuracy_score(y_val, preds)
        scores.append(acc_score)
        
        print(f"Fold {n_fold+1}/{n_splits} Accuracy: {acc_score}")

    print(f"Avg. accuracy: {np.mean(scores)}")
    return scores


scores = cross_validation(X, y.to_frame(), n_splits=5)


model = xgb.XGBClassifier(**xgb_params_prod)


mm_scaler = MinMaxScaler()
X = pd.DataFrame(mm_scaler.fit_transform(X), columns=mm_scaler.get_feature_names_out())


model.fit(X, y, verbose=False)


test_id_col = test_df["id"].copy()

test_df = test_df.drop(columns=["id"])

X_test = pd.get_dummies(test_df, columns=categorical_cols)
X_test = X_test.fillna(X_test.mean())

X_test = feature_engineer(X_test)

X_test = pd.DataFrame(mm_scaler.transform(X_test), columns=mm_scaler.get_feature_names_out())


X_test.head()


test_preds = model.predict(X_test)


test_preds


test_labels = ["Introvert" if test_preds[i] == 0 else "Extrovert" for i in range(len(test_preds))]


preds_df = pd.DataFrame({
    "id": test_id_col,
    "Personality": test_labels
})


preds_df.head()


preds_df.to_csv("/kaggle/working/submission.csv", index=False)

