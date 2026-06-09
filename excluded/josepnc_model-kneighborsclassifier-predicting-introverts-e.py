from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)


COMPETITION_DATA = Path("/kaggle/input/playground-series-s5e7")
list(COMPETITION_DATA.glob("*.csv"))


train = pd.read_csv(COMPETITION_DATA / "train.csv", index_col="id")
test = pd.read_csv(COMPETITION_DATA / "test.csv", index_col="id")


train.sample(5)


train.info()


train.describe()


pd.DataFrame([train.duplicated().sum(), test.duplicated().sum()], columns=["Duplicates"], index=["Train Data", "Test Data"])


target_column = ["Personality"]

object_columns_train = train.select_dtypes(include="object").columns.drop(target_column) #.drop(ignore_column, errors="ignore")
num_columns_train = train.select_dtypes(include="number").columns #.drop(ignore_column, errors="ignore")

object_columns_test = test.select_dtypes(include="object").columns #.drop(ignore_column, errors="ignore")
num_columns_test = test.select_dtypes(include="number").columns #.drop(ignore_column, errors="ignore")


print(f"Object Columns in train  df --> {object_columns_train.to_list()}")
print(f"Numeric Columns in train df --> {num_columns_train.to_list()}")
print(f"Target Column ----------------> {target_column}")
print()
print(f"Object Columns in test   df --> {object_columns_test.to_list()}")
print(f"Numeric Columns in test  df --> {num_columns_test.to_list()}")


nrows = 2
ncols = object_columns_train.shape[0] + 1
fig = plt.figure(figsize=(ncols * 3.5, nrows * 2))
autopct = "%0.1f"

# Train columns
for idx, col in enumerate(object_columns_train, start=1):
    values = train[col].value_counts()
    ax = plt.subplot(nrows, ncols, idx)
    ax.pie(values, labels=values.index, startangle=90, rotatelabels=False, autopct=autopct)
    ax.set_title(f"Train: {col}", fontweight="bold")
idx += 1
ax = plt.subplot(nrows, ncols, idx)
values = train[target_column].value_counts()
ax.pie(values, labels=values.index, startangle=90, rotatelabels=False, autopct=autopct)
ax.set_title(f"Target column: {target_column[0]}", fontweight="bold")

# Test columns
for idx, col in enumerate(object_columns_test, start=idx+1):
    values = test[col].value_counts()
    ax = plt.subplot(nrows, ncols, idx)
    ax.pie(values, labels=values.index, startangle=90, rotatelabels=False, autopct=autopct)
    ax.set_title(f"Test: {col}", fontweight="bold")

fig.suptitle("Object Columns", fontsize="xx-large", fontweight="bold")
plt.tight_layout()


df = pd.concat(
    [
        train[num_columns_train],
        test[num_columns_test]
    ],
    axis=1,
    keys=["train_data", "test_data"]
).melt()


df.variable_0 = df.variable_0.astype("category")
df.variable_1 = df.variable_1.astype("category")
df = df.dropna(ignore_index=True)
df = df.rename(columns={"variable_0" : "data", "variable_1" : "features"})


ncols = 2
nrows = (df.features.cat.categories.shape[0] + 1) // 2

plt.figure(figsize=(ncols * 5, nrows * 3))
idx = 1
for feature in df.features.cat.categories:
    ax = plt.subplot(nrows, ncols, idx)
    sns.kdeplot(
        data=df.loc[df.features == feature],
        x="value",
        hue="data", hue_order=["train_data", "test_data"],
        common_norm=False,
        ax=ax
    )
    ax.set_title(feature, fontweight="bold")
    idx += 1

fig.suptitle("Distribution Numeric Columns train/test", fontsize="xx-large", fontweight="bold")
plt.tight_layout()


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


numerical_preprocessor = Pipeline(
    steps = [
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", RobustScaler()),
    ]
)


categorical_preprocessor = Pipeline(
    steps = [
        ("most_frequent" , SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("cat_encoder", OrdinalEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_preprocessor, num_columns_train),
        ("categorical", categorical_preprocessor, object_columns_train),
    ]
)


rng = np.random.RandomState(0)


pipe = Pipeline(
    [
        ("processor", preprocessor),
        ("knc", KNeighborsClassifier()),
    ]
)


params = {
    "knc__n_neighbors" : [1, 3, 5, 7, 11],
    "knc__weights" : ["uniform", "distance"],
    "knc__leaf_size" : [5, 10, 15],
    "knc__algorithm" : ["ball_tree", "kd_tree", "brute"]
}


model = GridSearchCV(
    pipe,
    param_grid=params,
    scoring='accuracy',
    cv=5,
    error_score="raise",
)


X = train.drop(columns=target_column)
y = train.loc[:, target_column].iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)


%%time
    model.fit(X_train, y_train)


model.score(X_train, y_train)


model.best_params_


y_pred = pd.Series(model.predict(X_test), index=y_test.index)


print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")


result = pd.DataFrame(model.predict(test), index=test.index, columns=["Personality"])


result.head()


result.to_csv("submission.csv")










