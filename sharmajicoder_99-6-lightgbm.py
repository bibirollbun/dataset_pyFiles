# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from lightgbm import LGBMRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df


df.drop_duplicates()


df.info()


df.describe()


print(df["Sex"].value_counts())
print("_" * 50)
print(df["Sex"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Sex'].nunique()}")


plt.figure(figsize = (8, 6), dpi = 100, facecolor = "white", edgecolor = "black")
bars = plt.bar(
    df["Sex"].value_counts().index, df["Sex"].value_counts().values,
    color = "green",
    width = 0.5,
    alpha = 0.9,
    zorder = 3
)
for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            # fontweight="bold",
            zorder=4
    )
plt.title("Bar plot of Sex", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Sex", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Age"].value_counts())
print("_" * 50)
print(df["Age"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Age'].nunique()}")


print(df["Age"].describe())
print("_" * 30)
print(df["Age"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Age"],
    bins = int(np.sqrt(df["Age"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Age", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Height"].value_counts())
print("_" * 50)
print(df["Height"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Height'].nunique()}")


print(df["Height"].describe())
print("_" * 30)
print(df["Height"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Height"],
    bins = int(np.sqrt(df["Height"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Height", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Height", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Weight"].value_counts())
print("_" * 50)
print(df["Weight"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Weight'].nunique()}")


print(df["Weight"].describe())
print("_" * 30)
print(df["Weight"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Weight"],
    bins = int(np.sqrt(df["Weight"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Weight", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Weight", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Duration"].value_counts())
print("_" * 50)
print(df["Duration"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Duration'].nunique()}")


print(df["Duration"].describe())
print("_" * 30)
print(df["Duration"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Duration"],
    bins = int(np.sqrt(df["Duration"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Duration", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Duration", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Heart_Rate"].value_counts())
print("_" * 50)
print(df["Heart_Rate"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Heart_Rate'].nunique()}")


print(df["Heart_Rate"].describe())
print("_" * 30)
print(df["Heart_Rate"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Heart_Rate"],
    bins = int(np.sqrt(df["Heart_Rate"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Heart_Rate", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Heart_Rate", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Body_Temp"].value_counts())
print("_" * 50)
print(df["Body_Temp"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Body_Temp'].nunique()}")


print(df["Body_Temp"].describe())
print("_" * 30)
print(df["Body_Temp"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Body_Temp"],
    bins = int(np.sqrt(df["Body_Temp"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Body_Temp", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Body_Temp", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


print(df["Calories"].value_counts())
print("_" * 50)
print(df["Calories"].value_counts(normalize = True) * 100)
print("_" * 50)
print(f"Number of Unique Values : {df['Calories'].nunique()}")


print(df["Calories"].describe())
print("_" * 30)
print(df["Calories"].skew())


plt.figure(figsize = (8,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.hist(
    df["Calories"],
    bins = int(np.sqrt(df["Calories"].nunique())) + 10,
    color = "orange",
)
plt.title("Histogram of Calories", fontsize = 20, fontweight = "bold", loc = "center")
plt.xlabel("Calories", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


for sex in list(df["Sex"].unique()):
    sex_age = df[df["Sex"] == sex][["Age"]]
    sex_age_range = [sex_age.min(), sex_age.max()]
    print(f"{sex} : {sex_age_range}")


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Age"], df["Height"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Age & Height", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Height", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Age"], df["Weight"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Age & Weight", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Weight", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Age"], df["Body_Temp"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Age & Body_Temp", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Body_Temp", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Age"], df["Heart_Rate"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Age & Heart_Rate", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Heart_Rate", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Weight"], df["Heart_Rate"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Weight & Heart_Rate", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Weight", fontsize = 16)
plt.ylabel("Heart_Rate", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Age"], df["Calories"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Age & Calories", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Age", fontsize = 16)
plt.ylabel("Calories", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (20,8), dpi  = 200, facecolor = "white", edgecolor = "black")
plt.scatter(
    df["Weight"], df["Calories"],
    color = plt.cm.viridis(np.arange(0,1,100)),
    marker = "o"
)
plt.title("Scatter Plot of Weight & Calories", fontsize = 20, fontweight = "bold", loc = "center", color = "black")
plt.xlabel("Weight", fontsize = 16)
plt.ylabel("Calories", fontsize = 16)
plt.xticks(fontsize = 14, rotation = 0)
plt.yticks(fontsize = 14, rotation = 0)
plt.tight_layout()
plt.show()


df.drop(columns = {"Sex"}).corr()


from sklearn.preprocessing import StandardScaler 


scalar = StandardScaler()

df[["Age", "Height", "Weight","Duration","Heart_Rate","Body_Temp"]] = scalar.fit_transform(df[["Age", "Height", "Weight","Duration","Heart_Rate","Body_Temp"]])


encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])



df


X = df.drop(columns  = {"Calories","id"})
Y = np.log1p(df["Calories"])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# lgbm_params = {
#     'objective': 'regression_l1',
#     'metric': 'rmse',
#     'n_estimators': 3000,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'num_leaves': 64,
#     'verbose': -1,
#     'n_jobs': -1,
#     'seed': 42,
#     'boosting_type': 'gbdt'
# }

model=LGBMRegressor(max_depth=6,n_estimators=4000,learning_rate=0.04,random_state=42)
# model=LGBMRegressor(**lgbm_params)


model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(f"LGBMRegressor:")
print("Mean Squared Error:", mean_squared_error(Y_test, y_pred))
print("Root Mean Squared Log Error:", mean_squared_log_error(Y_test, y_pred))
print("R² Score:", r2_score(Y_test, y_pred))


plt.figure(figsize=(6, 4))
plt.scatter(Y_test, y_pred, color="orange", edgecolor="black")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "k--", lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


df_test


df_test = df_test.drop_duplicates()


df_test.size


df_test.isnull().sum()


encoder = LabelEncoder()
df_test["Sex"] = encoder.fit_transform(df_test["Sex"])


scalar = StandardScaler()

df_test[["Age", "Height", "Weight","Duration","Heart_Rate","Body_Temp"]] = scalar.fit_transform(df_test[["Age", "Height", "Weight","Duration","Heart_Rate","Body_Temp"]])


y_pred = model.predict(df_test.drop(columns = {"id"}))


# Convert back from log scale
final_pred = np.expm1(y_pred)


final_pred = np.maximum(0, final_pred)


submission = pd.DataFrame({
    'id': df_test['id'],
    'Calories': final_pred
})
submission.to_csv('submission.csv', index=False)

