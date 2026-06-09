import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
train.head()


train.info()


train.isna().sum()


train.describe().round(2)


sns.displot(train["BeatsPerMinute"], kde=True, bins=25)
plt.title(f"Distribution of {train['BeatsPerMinute'].name}")
plt.show()


from scipy.stats import skew

skew(train["BeatsPerMinute"])


train.drop(columns=["BeatsPerMinute", "id"]).hist(
    bins=30, figsize=(15, 12), layout=(4, 3)
)
plt.suptitle("Feature Distributions")
plt.show()


corr = train.drop(columns="id").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


X = train.drop(columns=["BeatsPerMinute", "id"])
y = train["BeatsPerMinute"]


for column in X.columns:
    sns.scatterplot(x=X[column], y=y)
    plt.title(f"{column} vs {y.name}")
    plt.show()
    sns.boxplot(x=X[column])
    plt.title(f"Outliers in {column}")
    plt.show()


def remove_outliers(X: pd.DataFrame):
    for column in X.columns:
        if column != "id":
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr * 1.5
            upper_bound = q3 + iqr * 1.5

            X.loc[X[column] < lower_bound, column] = lower_bound
            X.loc[X[column] > upper_bound, column] = upper_bound

            sns.boxplot(x=X[column])
            plt.title(f"Outliers in {column}")
            plt.show()

    return X


X = remove_outliers(X)


X.hist(bins=30, figsize=(15, 12), layout=(4, 3))
plt.show()


def engineer_features(X: pd.DataFrame):
    X["TrackDurationMins"] = X["TrackDurationMs"] / 60000
    X = X.drop(columns="TrackDurationMs")
    X["RhythmEnergy"] = X["RhythmScore"] * X["Energy"]
    return X

X = engineer_features(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

xgb = XGBRegressor(random_state=42)

param_dist = {
    "max_depth": [3, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train_scaled, y_train)

print("Best parameters:", random_search.best_params_)


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


feature_importance = pd.DataFrame(
    {"features": X.columns, "importance": best_model.feature_importances_}
)
sns.barplot(feature_importance, x="features", y="importance")
plt.title("Feature Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print(feature_importance.sort_values(by="importance", ascending=False))


test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test.head()


test.shape


for column in test.columns:
    if column != "id":
        sns.boxplot(x=test[column])
        plt.title(f"Outliers in {column}")
        plt.show()


test = remove_outliers(test)


test = engineer_features(test)
test_scaled = scaler.transform(test.drop(columns="id"))


y_test_pred = best_model.predict(test_scaled)
y_test_pred


submission = pd.DataFrame({"id": test["id"], "BeatsPerMinute": y_test_pred})
submission.head()


from datetime import datetime

timestamp = datetime.now().strftime("%m-%d-%y_%H:%M")

submission.to_csv(f"submission_{timestamp}.csv", index=False)

