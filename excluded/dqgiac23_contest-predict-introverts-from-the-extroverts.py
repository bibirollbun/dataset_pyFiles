import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_data["Stage_fear"] = train_data["Stage_fear"].map({'Yes': 1, "No": 0})
train_data["Drained_after_socializing"] = train_data["Drained_after_socializing"].map({'Yes': 1, "No": 0})
train_data["Personality"] = train_data["Personality"].map({'Extrovert': 1, "Introvert": 0})


train_data.corr()["Personality"][1:-1]


old_train_data = train_data.copy()
old_train_data

train_data = old_train_data.copy()


train_data = train_data.drop(['id'], axis=1).dropna()
train_data


rows_with_nan = old_train_data[old_train_data.isna().any(axis=1)]
introvert_nan_data = rows_with_nan[rows_with_nan['Personality'] == 0]
extrovert_nan_data = rows_with_nan[rows_with_nan['Personality'] == 1]


for i in list(introvert_nan_data.columns):
    mean_for_i = introvert_nan_data[introvert_nan_data[i].isna() == False][i].mean()
    introvert_nan_data[i] = introvert_nan_data[i].fillna(int(mean_for_i))


for i in list(extrovert_nan_data.columns):
    mean_for_i = extrovert_nan_data[extrovert_nan_data[i].isna() == False][i].mean()
    extrovert_nan_data[i] = extrovert_nan_data[i].fillna(int(mean_for_i))


from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

X = pd.concat([train_data.drop(["Personality"], axis=1), introvert_nan_data.drop(["id", "Personality"], axis=1), extrovert_nan_data.drop(["id", "Personality"], axis=1)], ignore_index=True)
y = pd.concat([train_data["Personality"], introvert_nan_data["Personality"], extrovert_nan_data["Personality"]], ignore_index=True)

skf = StratifiedKFold(shuffle=True, random_state=42)
a = (skf.split(X, y))
for train_index, test_index in a:
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    break
print(X_train, X_val)


parameters_dict = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 2,
    "verbosity": -1,
    'n_estimators': 300, 
    'learning_rate': 0.035, 
    'max_depth': 3,
    'min_data_in_leaf': 1
}
model = LGBMClassifier(**parameters_dict)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
print(mean_absolute_error(y_val_pred, y_val))


model.fit(X, y)


old_test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_data = test_data.drop(["id"], axis=1)
test_data["Stage_fear"] = test_data["Stage_fear"].map({'Yes': 1, "No": 0})
test_data["Drained_after_socializing"] = test_data["Drained_after_socializing"].map({'Yes': 1, "No": 0})
for i in list(test_data.columns):
    test_data[i] = test_data[i].fillna(int(test_data[i].mean()))
test_data


result = model.predict(test_data)
official = np.array(["Extrovert"] * len(result))
official[result == 0] = "Introvert"
official


submission = pd.DataFrame(np.stack((old_test_data["id"], official), axis=1), columns=["id", "Personality"])


submission.to_csv("submission.csv", index=False)

