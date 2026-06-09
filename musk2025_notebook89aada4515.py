import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


df_test.info()


df_train.info()


train_copie = df_train.copy()
test_copie = df_test.copy()


to_drop = ["education","contact","month","poutcome","education","job"]
train_copie.drop(columns = to_drop, inplace = True)
test_copie.drop(columns = to_drop, inplace = True)


to_transform = ["default","housing","loan"]

for transform in to_transform:
    train_copie[transform] = [True if trans != "no" else False  for trans in train_copie[transform].values]
    test_copie[transform] = [True if trans != "no" else False  for trans in test_copie[transform].values]

   
le = LabelEncoder()

train_copie["marital"] = le.fit_transform(train_copie["marital"])
test_copie["marital"] = le.fit_transform(test_copie["marital"])


plt.figure(figsize=(14,8))
sns.heatmap(train_copie.corr(), annot = True)
plt.show()


x_train = train_copie[['balance','duration', 'previous']]
x_test = test_copie[['balance','duration', 'previous']]
y_train = train_copie["y"]


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

random_for = RandomForestClassifier(
    n_estimators=200,
    max_features="log2",
    max_depth=20,
    min_samples_split=40,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced')

models = [XGBClassifier(),GaussianNB(),random_for]

for model in models :
    model.fit(x_train, y_train)
    print(f"{model} :")

    train_preds = model.predict(x_train)
    print('Training Accuracy : ', ras(y_train, train_preds))



proba_pred = models[2].predict_proba(x_test)[:, 1]


submission = pd.DataFrame({
    "id": df_test['id'],
    "y": proba_pred.round(1)
})

submission.to_csv("submission.csv", index=False)




