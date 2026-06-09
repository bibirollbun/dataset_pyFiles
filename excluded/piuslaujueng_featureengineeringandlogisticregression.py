import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


combined_df.head(15)
# view the dataset


combined_df.info()


plt.figure(figsize=(20,15))
for i, feature in enumerate([f for f in combined_df.columns if not f in ["id", "rainfall"]], 1):
    plt.subplot(4,3,i)
    sns.scatterplot(x=combined_df[feature], y=combined_df["rainfall"])
    plt.title(feature)
plt.tight_layout()
plt.show()
        


plt.figure(figsize=(20,15))
for i, feature in enumerate([f for f in combined_df.columns if not f in ["id", "rainfall", "day"]], 1):
    plt.subplot(4,3,i)
    sns.boxplot(y=combined_df[feature]) 
    plt.title(feature)
plt.tight_layout()
plt.show()
        


plt.figure(figsize=(12, 8))
sns.heatmap(combined_df.corr(numeric_only=True), annot=True)


plt.figure(figsize=(20,15))
for i, feature in enumerate(["dewpoint", "temparature", "mintemp"],1):
    plt.subplot(1,3,i)
    sns.scatterplot(x=combined_df[feature],
                    y=combined_df["maxtemp"],
                   hue=combined_df["rainfall"],  
            palette={0: "red", 1: "blue"})
    plt.title(f"{feature} vs maxtemp")
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,15))
for i, feature in enumerate(["temparature", "mintemp"],1):
    plt.subplot(1,2,i)
    sns.scatterplot(x=combined_df[feature],
                    y=combined_df["dewpoint"],
                   hue=combined_df["rainfall"],  
            palette={0: "red", 1: "blue"})
    plt.title(f"{feature} vs dewpoint")
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,15))
for i, feature in enumerate(["maxtemp", "temparature", "mintemp"],1):
    plt.subplot(1,3,i)
    sns.scatterplot(x=combined_df[feature],
                    y=combined_df["pressure"],
                   hue=combined_df["rainfall"],  
            palette={0: "red", 1: "blue"})
    plt.title(f"{feature} vs pressure")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
plt.subplot(1,1,1)
sns.scatterplot(x=combined_df["cloud"],
                y=combined_df["sunshine"],
               hue=combined_df["rainfall"],  
        palette={0: "red", 1: "blue"})
plt.title("cloud vs sunshine")
plt.show()


rainfall_mean_df = train_df.groupby("day")["rainfall"].mean().reset_index()
rainfall_mean_1 = rainfall_mean_df.iloc[:len(rainfall_mean_df) // 3]
rainfall_mean_2 = rainfall_mean_df.iloc[len(rainfall_mean_df) // 3 : 2*(len(rainfall_mean_df) // 3)]
rainfall_mean_3 = rainfall_mean_df.iloc[2*(len(rainfall_mean_df) // 3) :]
plt.figure(figsize=(50,10))
plt.subplot(1,1,1)
sns.barplot(x=rainfall_mean_1["day"],
                y=rainfall_mean_1["rainfall"])
plt.show()


plt.figure(figsize=(50,10))
plt.subplot(1,1,1)
sns.barplot(x=rainfall_mean_2["day"],
                y=rainfall_mean_2["rainfall"])
plt.show()


plt.figure(figsize=(50,10))
plt.subplot(1,1,1)
sns.barplot(x=rainfall_mean_3["day"],
                y=rainfall_mean_3["rainfall"])
plt.show()


combined_df.isnull().sum().sort_values(ascending=False)


# There is one record with winddirection empty
combined_df[combined_df["winddirection"].isna()]


# since winddirection is considered highly correlated with temparature, maxtemp, mintemp, and pressure, I will choose simply two features to determine how to impute winddirection
plt.figure(figsize=(10,8))
plt.subplot(1,1,1)
sns.regplot(x=combined_df["winddirection"], y=combined_df["temparature"], scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
plt.axhline(y=30.6, color="green", linestyle="--", linewidth=2)
plt.title("winddirection vs temparature")
plt.show()


plt.figure(figsize=(10,8))
plt.subplot(1,1,1)
sns.regplot(x=combined_df["winddirection"], y=combined_df["pressure"], scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
plt.axhline(y=1007.8, color="green", linestyle="--", linewidth=2)
plt.title("winddirection vs pressure")
plt.show()


combined_df.loc[combined_df["id"] == 2707]["winddirection"] = (240+260)/2


combined_df["rainyday"] = np.where(
    ((combined_df["day"] >= 47) & (combined_df["day"] <= 63)) |
    ((combined_df["day"] >= 77) & (combined_df["day"] <= 139)) |
    ((combined_df["day"] >= 212) & (combined_df["day"] <= 242)) |
    ((combined_df["day"] >= 292) & (combined_df["day"] <= 333)) |
    ((combined_df["day"] >= 359) & (combined_df["day"] <= 365)),
    1, 
    0
)


# temperature_range
combined_df["temparature_range"] = combined_df["maxtemp"] - combined_df["mintemp"]


# humidity_index
combined_df["humidity_index"] = combined_df["humidity"] * combined_df["temparature"]


# dewpoint_spread
combined_df["dewpoint_spread"] = combined_df["temparature"] - combined_df["dewpoint"]


# cloud_sunshine_index
combined_df["CSI"] = combined_df["cloud"] / (combined_df["sunshine"] + 1)


combined_df.drop(columns=["day"], inplace=True)


train_df = combined_df[combined_df["rainfall"].notna()]
test_df = combined_df[combined_df["rainfall"].isna()]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["rainfall"], inplace=True)


X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=11)


# No categorical data given
numeric_pipeline = Pipeline(steps = [
    ("inpute", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers = [
        ("numeric_pipeline", numeric_pipeline, X_train.columns.tolist())
    ], remainder = "passthrough", n_jobs=-1
)
preprocessor


model_1 = make_pipeline(preprocessor, LogisticRegression())


model_1.fit(X_train, y_train)


model_1_pred = model_1.predict(X_test)


print(f"model 1 accuracy score is {accuracy_score(model_1_pred, y_test)}")


test_pred = model_1.predict(test_df.drop(columns=["id"]))


submission1 = test_df[["id"]]


submission1["rainfall"] = test_pred


submission1.to_csv("submission1.csv", index=None)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)
grid_search.fit(X_train, y_train)


model_2 = make_pipeline(preprocessor, grid_search.best_estimator_)


model_2.fit(X_train, y_train)


model_2_pred = model_2.predict(X_test)


print(f"model 2 accuracy score is {accuracy_score(model_2_pred, y_test)}")


print(classification_report(model_2_pred, y_test))


test_pred = model_2.predict(test_df.drop(columns=["id"]))


submission2 = test_df[["id"]]
submission2["rainfall"] = test_pred
submission2.to_csv("submission2.csv", index=None)


param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': np.arange(3, 10),
    'learning_rate': np.logspace(-3, 0, 100),
    'subsample': np.arange(0.5, 1.01, 0.1),
    'colsample_bytree': np.arange(0.5, 1.01, 0.1),
    'gamma': np.logspace(-3, 1, 100),
    'reg_alpha': np.logspace(-3, 1, 100),
    'reg_lambda': np.logspace(-3, 1, 100)
}


xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)


random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
    scoring='accuracy' #or 'roc_auc', 'f1' etc.
)
random_search.fit(X_train, y_train)


model_3 = make_pipeline(preprocessor, random_search.best_estimator_)


model_3.fit(X_train, y_train)


model_3_pred = model_3.predict(X_test)


print(f"model 3 accuracy score is {accuracy_score(model_3_pred, y_test)}")


print(classification_report(model_3_pred, y_test))


test_pred = model_3.predict(test_df.drop(columns=["id"]))


submission3 = test_df[["id"]]
submission3["rainfall"] = test_pred
submission3.to_csv("submission3.csv", index=None)




