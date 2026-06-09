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


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping


train_df = pd.read_csv("/kaggle/input/playground-series-s3e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e7/sample_submission.csv")


train_df.head()


train_df.info()


train_df.describe()





train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])


train_df.isnull().sum()


X = train_df.drop(columns=["booking_status"])
y = train_df["booking_status"]


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Distribution of Booking Status")
plt.xlabel("Booking Status (0 = Canceled, 1 = Not Canceled)")
plt.ylabel("Count")
plt.show()


num_cols_to_plot = [
    "lead_time",
    "avg_price_per_room",
    "no_of_special_requests",
    "no_of_week_nights",
    "no_of_weekend_nights"
]

plt.figure(figsize=(15,10))

for i, col in enumerate(num_cols_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_df[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="booking_status", y="lead_time", data=train_df)
plt.title("Lead Time vs Booking Status")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="booking_status", y="avg_price_per_room", data=train_df)
plt.title("Average Price per Room vs Booking Status")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(
    x="no_of_special_requests",
    hue="booking_status",
    data=train_df
)
plt.title("Special Requests vs Booking Status")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(
    x="room_type_reserved",
    hue="booking_status",
    data=train_df
)
plt.title("Room Type Reserved vs Booking Status")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(
    x="market_segment_type",
    hue="booking_status",
    data=train_df
)
plt.title("Market Segment vs Booking Status")
plt.show()


plt.figure(figsize=(14,10))
corr = train_df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.show()





train_df.groupby("booking_status").mean()


train_df.groupby("booking_status")["lead_time"].describe()



train_df.groupby("booking_status")["avg_price_per_room"].describe()



pd.crosstab(
    train_df["repeated_guest"],
    train_df["booking_status"],
    normalize="index"
)


pd.crosstab(
    train_df["no_of_previous_cancellations"],
    train_df["booking_status"]
).head(10)


pd.crosstab(
    train_df["no_of_special_requests"],
    train_df["booking_status"],
    normalize="index"
)








train_df[["lead_time", "avg_price_per_room"]].quantile([0.01, 0.99])



for df in [train_df, test_df]:
    df["total_nights"] = (
        df["no_of_week_nights"] + df["no_of_weekend_nights"]
    )


for df in [train_df, test_df]:
    df["is_family"] = (df["no_of_children"] > 0).astype(int)


lead_time_threshold = train_df["lead_time"].median()



for df in [train_df, test_df]:
    df["high_lead_time"] = (df["lead_time"] > lead_time_threshold).astype(int)


price_threshold = train_df["avg_price_per_room"].median()



for df in [train_df, test_df]:
    df["high_price"] = (df["avg_price_per_room"] > price_threshold).astype(int)


for df in [train_df, test_df]:
    df["guest_reliability"] = (
        df["no_of_previous_bookings_not_canceled"]
        - df["no_of_previous_cancellations"]
    )





X = train_df.drop(columns=["booking_status"])
y = train_df["booking_status"]


X_test = test_df.copy()



X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


lg = LogisticRegression(
    max_iter=1000,
    random_state=42
)


lg.fit(X_train, y_train)


y_pred = lg.predict(X_valid)


accuracy = accuracy_score(y_valid, y_pred)
accuracy


print(classification_report(y_valid, y_pred))



cv_scores = cross_val_score(
    lg,
    X,
    y,
    cv=5,
    scoring="accuracy"
)



cv_scores.mean()



rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


rf.fit(X_train, y_train)



y_pred_rf = rf.predict(X_valid)



rf_accuracy = accuracy_score(y_valid, y_pred_rf)
rf_accuracy


print(classification_report(y_valid, y_pred_rf))



rf_cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


rf_cv_scores.mean()



importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)


plt.figure(figsize=(10,6))
sns.barplot(x=importances[:10], y=importances[:10].index)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()





gbc = GradientBoostingClassifier(random_state=42)


gbc.fit(X_train, y_train)


y_pred_gbc = gbc.predict(X_valid)


accuracy_score(y_valid, y_pred_gbc)



scores_gbc = cross_val_score(
        gbc,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )


scores_gbc.mean()


all_scores = {}
all_scores['Logistic Regression'] = cv_scores.mean()
all_scores['Random Forest'] = rf_cv_scores.mean()
all_scores['Gradient Boosting'] = scores_gbc.mean()


plt.figure(figsize=(6,4))
sns.barplot(x=list(all_scores.keys()), y=list(all_scores.values()))
plt.ylabel("Cross-Validation Accuracy")
plt.title("Model Comparison")
plt.show()


best_model_name = max(all_scores, key=all_scores.get)
best_model_name





xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


xgb_model.fit(X_train, y_train)



xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]


xgb_val_pred = xgb_model.predict_proba(X_valid)[:, 1]


print("XGB AUC:", roc_auc_score(y_valid, xgb_val_pred))


xgb_cv_score = cross_val_score(
    xgb_model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
).mean()


xgb_cv_score



importances = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)



plt.figure(figsize=(10,6))
importances[:10].plot(kind="bar")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.show()


lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=42
)


lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50)],
)


lgb_train_pred = lgb_model.predict_proba(X_train)[:, 1]


lgb_val_pred = lgb_model.predict_proba(X_valid)[:, 1]


print("LightGBM AUC:", roc_auc_score(y_valid, lgb_val_pred))






cat_model = CatBoostClassifier(
    iterations=1000,        
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,      
    early_stopping_rounds=50
)


cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)


cat_train_pred = cat_model.predict_proba(X_train)[:, 1]


cat_val_pred = cat_model.predict_proba(X_valid)[:, 1]
print("CatBoost AUC:", roc_auc_score(y_valid, cat_val_pred))














X_train_stack = np.column_stack([
    xgb_train_pred,
    lgb_train_pred,
    cat_train_pred
])


X_val_stack = np.column_stack([
   xgb_val_pred,
    lgb_val_pred,
    cat_val_pred
])





nn_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_stack.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])


nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["AUC"]
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


nn_model.fit(
    X_train_stack, y_train,
    validation_data=(X_val_stack, y_valid),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)


nn_val_pred = nn_model.predict(X_val_stack).ravel()


print("NN AUC:", roc_auc_score(y_valid, nn_val_pred))






# parameters = {
#     "n_estimators": [200, 400],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5],
#     "min_samples_leaf": [1, 2]
# }


# gbc = GradientBoostingClassifier(random_state=42)


# grid_search = GridSearchCV(
#     gbc,
#     parameters,
#     cv=3,
#     scoring="accuracy",
#     n_jobs=-1,
#     verbose=1
# )


# grid_search.fit(X_train, y_train)


# grid_search.best_params_


# best_gbc_model = grid_search.best_estimator_


# y_pred_tuned = best_gbc_model.predict(X_valid)


# accuracy_score(y_valid, y_pred_tuned)



# tuned_cv_score = cross_val_score(
#     best_gbc_model,
#     X,
#     y,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=-1
# ).mean()


# tuned_cv_score


# best_gbc_model.fit(X, y)



# tuned_predictions = best_gbc_model.predict(X_test)





xgb_full = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_full.fit(X, y)


lgb_full = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=42
)
lgb_full.fit(X, y)


cat_full = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)
cat_full.fit(X, y)


xgb_pred_full = xgb_full.predict_proba(X)[:, 1]
lgb_pred_full = lgb_full.predict_proba(X)[:, 1]
cat_pred_full = cat_full.predict_proba(X)[:, 1]


X_stack_full = np.column_stack([xgb_pred_full, lgb_pred_full, cat_pred_full])



early_stop = EarlyStopping(
    monitor='loss',     
    patience=10,
    restore_best_weights=True
)


nn_model.fit(
    X_stack_full, y,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)


xgb_test_pred = xgb_full.predict_proba(X_test)[:, 1]
lgb_test_pred = lgb_full.predict_proba(X_test)[:, 1]
cat_test_pred = cat_full.predict_proba(X_test)[:, 1]


X_test_stack = np.column_stack([xgb_test_pred, lgb_test_pred, cat_test_pred])
nn_test_pred = nn_model.predict(X_test_stack).ravel()









submission = pd.DataFrame({
    "id": sample_submission["id"],
    "booking_status": nn_test_pred
})


submission.to_csv("submission.csv", index=False)




