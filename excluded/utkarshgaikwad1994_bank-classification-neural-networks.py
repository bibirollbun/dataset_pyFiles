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


%pip install --upgrade scikit-learn


import pandas as pd
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.head()


df.info()


df.nunique()


X = df.drop(columns = ["id", "y"])
Y = df["y"]


X.head()


Y.head()


from sklearn.model_selection import train_test_split


xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


xtrain.head()


ytrain.head()


xtest.head()


ytest.head()


num_cols = X.select_dtypes(include="number").columns.tolist()
num_cols


cat_unique = X.select_dtypes(include="object").nunique()
cat_unique


low_card_cols = cat_unique[cat_unique < 10].index.tolist()
low_card_cols


high_card_cols = cat_unique[cat_unique > 10].index.tolist()
high_card_cols


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, TargetEncoder, PolynomialFeatures
)
from sklearn.compose import ColumnTransformer


num_pipe = make_pipeline(
    SimpleImputer(strategy="median")
)


low_card_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknown"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
)


high_card_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknown"),
    TargetEncoder(target_type="binary", cv=5, smooth="auto", random_state=42)
)


transformer = ColumnTransformer(
    [
        ("num", num_pipe, num_cols),
        ("low", low_card_pipe, low_card_cols),
        ("high", high_card_pipe, high_card_cols)
    ]
)


pipe = make_pipeline(
    transformer,
    StandardScaler()
)


pipe.fit(xtrain, ytrain)


xtrain_pre = pipe.transform(xtrain)
xtest_pre = pipe.transform(xtest)


xtrain_pre


xtest_pre


xtrain_pre.shape


xtest_pre.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


xtrain_pre.shape[1]


model = Sequential([
    Input(shape = (xtrain_pre.shape[1],)),
    Dense(32, activation="relu", kernel_regularizer=regularizers.L2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation="relu", kernel_regularizer=regularizers.L2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(ytrain),
    y=ytrain
)
class_weights = dict(enumerate(class_weights))
class_weights


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


model.compile(loss="binary_crossentropy", optimizer=Adam(0.001), metrics=["AUC"])


history = model.fit(
    xtrain_pre, ytrain,
    validation_split=0.1,
    epochs=100,
    batch_size=256,
    class_weight=class_weights,# <- handles imbalance
    callbacks = [early_stop]
)


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.title('Learning Curve - Loss')
plt.legend()
plt.show()


model.evaluate(xtrain_pre, ytrain)


model.evaluate(xtest_pre, ytest)


ypred_test = model.predict(xtest_pre)
ypred_test[0:5]


ytest[0:5]


from sklearn.metrics import RocCurveDisplay


RocCurveDisplay.from_predictions(ytest, ypred_test)


ypred_test_actual = [1 if prob>=0.5 else 0 for prob in ypred_test]


ypred_test_actual[0:5]


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(ytest, ypred_test_actual)


from sklearn.metrics import classification_report

print(classification_report(ytest, ypred_test_actual))


xnew = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
xnew.head()


xnew_pre = pipe.transform(xnew)
xnew_pre


xnew_pre.shape


probs = model.predict(xnew_pre)
probs[0:5]


res = xnew.copy()[["id"]]
res.loc[:,"y"] = probs


res.to_csv("submission.csv", index=False)


res


len(res[res["y"] >= 0.5])


len(res[res["y"] < 0.5])




