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


train = "/kaggle/input/playground-series-s5e3/train.csv"
test = "/kaggle/input/playground-series-s5e3/test.csv"
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(train)
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.drop(['id'], axis=1, inplace=True)


df.head()


df.shape


df.info()


df.describe()


df[df.duplicated(keep='first')]


sns.set_style("darkgrid")
cols = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(13, len(cols)*3))
for index, feature in enumerate(cols, 1):
    if index == 12:
        continue
    plt.subplot(len(cols), 2, index)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewedness: {round(df[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


df.corr()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True,fmt=".2f", cmap="crest", linewidths=2)
plt.title("Correlation matrix")
plt.show()


from sklearn.feature_selection import mutual_info_classif

df.columns
x = df.drop(['rainfall'], axis=1)
y = df['rainfall']

score=mutual_info_classif(x, y, discrete_features=False)
score=pd.Series(score, name= "Mutual Information Score", index=x.columns)
score=score.sort_values(ascending=False)

score


dfscore=score.reset_index()
dfscore.columns = ["Feature", "MI Score"]
sns.barplot(data=dfscore, x="Feature", y="MI Score", palette="Blues")
plt.xticks(rotation=90) 
plt.show()


x.drop(["winddirection", "windspeed", "maxtemp", "dewpoint", "mintemp", "temparature", "pressure", "day"], axis=1, inplace=True)


from sklearn import linear_model
model=linear_model.LogisticRegression()

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")
copy = x
copy["cloud"] = pt.fit_transform(copy[["cloud"]])

model.fit(copy, y)
x_test=pd.read_csv(test).drop(["winddirection", "windspeed", "maxtemp", "dewpoint", "mintemp", "temparature", "pressure", "day"], axis=1)

x_test.head()


pred = model.predict(x_test.drop(["id"], axis=1))
pd.DataFrame(pred)


from sklearn.tree import DecisionTreeClassifier

clsf = DecisionTreeClassifier(random_state=1)
clsf.fit(x, y)
pred = clsf.predict(x_test.drop(["id"], axis=1))
pd.DataFrame(pred)



wa9il = x.drop(['humidity'], axis=1)
clsf.fit(wa9il, y)
pred = clsf.predict(x_test.drop(["id", "humidity"], axis=1))
pd.DataFrame(pred)



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x, y)
pred = forest.predict(x_test.drop(["id"], axis=1))
pd.DataFrame(pred)
forest.score(x, y)

importance = forest.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
print(sorted_idx)
features = x.columns


plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), importance[sorted_idx])
plt.xticks(range(len(features)), features[sorted_idx], rotation=90)
plt.title("feature importance")
plt.show()


from sklearn.model_selection import RandomizedSearchCV

param_grid= {
    "n_estimators": [10, 50, 100, 200, 500],
    "max_features":['auto', 'sqrt', 'log2'],
    "max_depth":[5, 10, 20, None],
    "min_samples_split":[2, 3, 5],
    "bootstrap":[True, False]    
}

forest2=RandomForestClassifier(random_state=42)
random=RandomizedSearchCV(estimator=forest2, param_distributions=param_grid, verbose=2, n_jobs=-1, scoring="accuracy", return_train_score=True)

random.fit(x, y)
random.best_params_


better_forest=random.best_estimator_
pred = better_forest.predict(x_test.drop(["id"], axis=1))
pd.DataFrame(pred)


from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, verbose=2, n_jobs=-1, scoring="accuracy", return_train_score=True)
grid.fit(x,y)
grid.best_params_


again=grid.best_estimator_
pred = again.predict(x_test.drop(["id"], axis=1))
pd.DataFrame(pred)


submission = pd.DataFrame({ "id": x_test["id"], "rainfall": pred})
submission.to_csv("submission9.csv", index=False)

