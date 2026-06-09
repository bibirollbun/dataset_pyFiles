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


train_file = "/kaggle/input/playground-series-s5e3/train.csv"
test_file = "/kaggle/input/playground-series-s5e3/test.csv"

df = pd.read_csv(train_file)

df.head(10)


columns = df.columns

df.info()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

df['rainfall'].plot(kind='hist')
plt.xlim((0,1))


target = ['rainfall']
features = [
    feature
    for feature in columns
    if feature not in [target[0], 'id', 'day']
]

print("Target name:", target)
print("Features:", features)


sns.boxplot(df[features])


from sklearn.model_selection import train_test_split

X = df[features]
y = df[target]
test_size = 0.2
train_len = int(0.8 * len(df))
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, stratify=y, test_size=0.2)
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]


num_features = len(features)

sns.heatmap(X_train.corr(), annot=True, fmt=".2f")
plt.show()


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, n_jobs=-1, verbose=1). \
        fit(X_train, y_train.values.reshape(-1,))


from sklearn.metrics import classification_report

prediction = model.predict(X_test)

print(classification_report(y_test.values.reshape(-1,), prediction))


feature_importances = pd.DataFrame({
    'feature' : model.feature_names_in_,
    'importance': model.feature_importances_,
})

feature_importances.sort_values(by=['importance'], ascending=False, inplace=True)

sns.barplot(data=feature_importances, x='importance', y='feature')
plt.show()


from sklearn.feature_selection import chi2, SelectKBest

selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X_train, y_train)

print(X_new[:10])
# print(X_train.columns[selector.get_support()], selector.scores_)

feature_importances_chi2 = pd.DataFrame({
    'feature': X_train.columns,
    'importance': selector.scores_,
})

feature_importances_chi2.sort_values('importance', ascending=False, inplace=True)

sns.barplot(data=feature_importances, x='importance', y='feature')
plt.show()


from sklearn.feature_selection import mutual_info_classif

mic = mutual_info_classif(X_train, y_train.values.ravel())

print(np.vstack([X_train.columns, mic]))

feature_importances_mic = pd.DataFrame(data=np.vstack([X_train.columns, mic]).T)
# feature_importances_mic.head()
feature_importances_mic.columns = ['feature', 'importance']
feature_importances_mic.sort_values('importance', ascending=False, inplace=True)

sns.barplot(data=feature_importances_mic, x='importance', y='feature')


# new_features = ['cloud', 'sunshine', 'humidity', 'temparature']
new_features = ['cloud', 'sunshine', 'humidity']

X_train = X_train[new_features]
X_test = X_test[new_features]


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

transformer = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), new_features),
    ]
)

pipeline = Pipeline(steps=[
    ('standard', transformer),
    ('classify', CatBoostClassifier())
])

pipeline.fit(X_train, y_train)


estimates = pipeline.predict(X_test)

print(classification_report(y_test.values.ravel(), estimates))


# plot auc_roc curve
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

pred_proba = pipeline.predict_proba(X_test)
print(pred_proba[:10,1])
fpr, tpr, threshold = roc_curve(y_test, pred_proba[:,1])
auc_score = auc(fpr, tpr)

sns.lineplot(x=fpr, y=tpr, label=f"AUC score: {auc_score:.2f}")
sns.lineplot(x=[0,1], y=[0,1], linestyle='--', label = "Threshold")
plt.fill_between(x=fpr, y1=tpr, alpha=0.25, color='black')
plt.suptitle("FPR vs TPR")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


import joblib

model_file = joblib.dump(pipeline, 'model.pkl')


print(os.path.join(os.getcwd(), "model.pkl"))


if "model.pkl" in os.listdir():
    clf = joblib.load(os.path.join(os.getcwd(), "model.pkl"))

df_test = pd.read_csv(test_file)

idx = df_test.id

data= df_test[new_features]

y_pred = clf.predict(data)


output = pd.DataFrame({
    'id':idx.values,
    'rainfall':y_pred,
})

output.to_csv("submission.csv", index=False)




