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


import pandas as pd
submit = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv")
train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")
train.head()


test.head()


submit.head()


print("train.shape: ", train.shape)
print("test.shape: ", test.shape)
print("submit.shape: ", submit.shape)


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


submit.isnull().sum()


for i in train:
    if train[i].dtype == 'object':
        print("\n\n\n\n\nColumn: ", i)
        print(f"Number of unique values is {len(train[i].unique())} and they are", train[i].unique())


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(train.drop(['id', 'CustomerId'], axis=1).describe().T, annot=True, cmap='winter_r', fmt='.3g')
plt.tight_layout()


sns.countplot(x = train["Gender"])


sns.countplot(x = train["Geography"])


train.Exited.value_counts()


sns.countplot(x = train.Exited)


sns.countplot(x = train.Age)


train.Age.mean()


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
onehot_df = pd.get_dummies(train[["Gender", "Geography"]], drop_first = True)
onehot_df


train.columns.values.tolist()


train = pd.concat([train, onehot_df], axis = 1)
train


train = pd.concat([train, pd.DataFrame({"Surname1": label.fit_transform(train['Surname'])})], axis=1)
train


train.drop(['Surname', 'Geography', 'Gender'], axis=1, inplace=True)


train.info()


pd.DataFrame(train.corrwith(train['Exited']).abs().sort_values(ascending=False)).style.background_gradient('winter_r')


train.drop(['CustomerId', 'id'], axis=1, inplace=True)


train['Exited'].value_counts()


from sklearn.utils import resample
majority_class = train[train['Exited'] == 0]
minority_class = train[train['Exited'] == 1]

minority_oversampled = resample(minority_class,
                                replace=True, 
                                n_samples=len(majority_class),  
                                random_state=42)

train = pd.concat([majority_class, minority_oversampled])


train['Exited'].value_counts()


X = train.drop("Exited", axis=1)
y = train["Exited"]
X


# from sklearn.decomposition import PCA
# pca = PCA(n_components=15)
# X = pca.fit_transform(X)
# X


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=12)


robust = RobustScaler()
X_train = robust.fit_transform(X_train)
X_test = robust.transform(X_test)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve, RocCurveDisplay


model = RandomForestClassifier(n_estimators=10, max_features='sqrt',
                                                            random_state=101)
model.fit(X_train,y_train)
preds = model.predict(X_test)


accuracy_score(y_test, preds)


param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 10, 15]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
preds = grid_search.predict(X_test)
accuracy_score(y_test, preds)


 grid_search.best_params_


rf_model = RandomForestClassifier(n_estimators=20, max_features='sqrt',
                                                            max_depth = 20, min_samples_split = 2, random_state=101)
rf_model.fit(X_train,y_train)
preds = rf_model.predict(X_test)
accuracy_score(y_test, preds)


fpr, tpr, _ = roc_curve(y_test, preds, pos_label=grid_search.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


auc(fpr, tpr)


classifier= XGBClassifier(n_estimators=256)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=classifier.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


auc(fpr, tpr)


svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=svc.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


auc(fpr, tpr)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


base_models = [
    ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

meta_model = LogisticRegression(max_iter = 100)

stacking_model = StackingClassifier(estimators=base_models,
                                    final_estimator=meta_model, cv=5)

stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=stacking_model.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


auc(fpr, tpr)


base_models = [
    ('rf', RandomForestClassifier(n_estimators=32, random_state=42)),
    ('svm', SVC(probability=True)),
    ('xgb', XGBClassifier(n_estimators=64))
]

meta_model1 = LogisticRegression()

stacking_model1 = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model1,
    cv=5
)
stacking_model1.fit(X_train, y_train)
y_pred = stacking_model1.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=stacking_model1.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


auc(fpr, tpr)


import tensorflow as tf


tf.random.set_seed(42)

model_14 = tf.keras.Sequential([
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])


model_14.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9),
                 metrics=["accuracy"])

history = model_14.fit(X_train,
                       y_train,
                       epochs=50,
                       validation_data=(X_test, y_test))


import torch
from torch import nn
from torch.nn.functional import relu as relu

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer1 = nn.Linear(in_features=12,
                                      out_features=16)
        self.linear_layer2 = nn.Linear(in_features=16,
                                      out_features=32)
        self.linear_layer3 = nn.Linear(in_features=32,
                                      out_features=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=relu(self.linear_layer1(x))
        x=relu(self.linear_layer2(x))
        x=relu(self.linear_layer3(x))
        return x


torch.manual_seed(42)
model_1 = Classifier()
model_1, model_1.state_dict()


from torch.nn import BCELoss
loss_fn = BCELoss()
X_train = torch.from_numpy(X_train).float() 
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(np.asarray(y_train)).float()
y_test = torch.from_numpy(np.asarray(y_test)).float()

optimizer = torch.optim.Adam(params=model_1.parameters(),
                            lr=0.01)

torch.manual_seed(42)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(X_train)
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.squeeze()

    loss = loss_fn(y_pred, torch.squeeze(y_train))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()

    with torch.inference_mode():

      test_pred = model_1(X_test)  
      y_test = y_test.squeeze().float() 
      test_loss = loss_fn(test_pred.squeeze(), y_test)

      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss} ")


plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();


rf_model = RandomForestClassifier(n_estimators=20, max_features='sqrt',
                                                            max_depth = 20, min_samples_split = 2, random_state=101)
rf_model.fit(X_train,y_train)
preds = rf_model.predict(X_test)
accuracy_score(y_test, preds)


base_models = [
    ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

meta_model = LogisticRegression(max_iter = 100)

stacking_model = StackingClassifier(estimators=base_models,
                                    final_estimator=meta_model, cv=5)

stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)
display("Aniqlik:", accuracy_score(y_test, y_pred))


test.head()


onehot_df = pd.get_dummies(test[["Gender", "Geography"]], drop_first = True) 
test = pd.concat([test, onehot_df], axis = 1)
test = pd.concat([test, pd.DataFrame({"Surname1": label.fit_transform(test['Surname'])})], axis=1)
test.drop(['id', 'CustomerId', 'Surname', 'Geography', 'Gender'], axis=1, inplace=True)

test.head()


test.shape


test = robust.transform(test)


submit.head()


preds = stacking_model.predict(test)
submit['Exited'] = preds
submit.to_csv("Loan_probs_submit.csv", index=False)

