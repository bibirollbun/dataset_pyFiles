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


# import libraries
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')


# read dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
origin = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# drop 'id' from train and test
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# concat train and origin
train = pd.concat([train, origin], axis=0, ignore_index=True)
train.info()


# numeric, categorical and target
numerics = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
category = [col for col in train.columns if train[col].dtype in ['object']]
target = 'Fertilizer Name'

# remove target from category
category.remove(target)

print(f"Numeric Features:\t{numerics}")
print(f"Categorical Features:\t{category}")
print(f"Target:\t{target}")


# encode categorical features
def encode(df, columns):
    df_copy = df.copy()
    le = LabelEncoder()
    for col in columns:
        df_copy[col] = le.fit_transform(df_copy[col])

    return df_copy

train = encode(train, category)
test = encode(test, category)


# new features
# feature engineering
train['NP_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1e-5)
train['NK_ratio'] = train['Nitrogen'] / (train['Potassium'] + 1e-5)
train['PK_ratio'] = train['Phosphorous'] / (train['Potassium'] + 1e-5)
test['NP_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1e-5)
test['NK_ratio'] = test['Nitrogen'] / (test['Potassium'] + 1e-5)
test['PK_ratio'] = test['Phosphorous'] / (test['Potassium'] + 1e-5)


# preapre for the model
X = train.drop(target, axis=1)
y = train[target]

# scale numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.fit_transform(test)

# encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
id_to_label = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))


# model architecture
def create_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(7, activation='softmax')  # 7 classes
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    return model


# Callbacks
def make_callbacks():
    lr_callback = ReduceLROnPlateau(
        monitor='accuracy',     
        factor=0.5,              
        patience=3,              
        verbose=1,               
        min_lr=1e-6              
    )
    early_stop_cb = EarlyStopping(
        monitor="accuracy", 
        patience=15,            
        restore_best_weights=True,
        mode="max", 
        verbose=1
    )
    return [lr_callback, early_stop_cb]


# MAP@3
def apk(actual, predicted, k=3):
    predicted = list(predicted)  # Convert to list for .index()
    if actual in predicted:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0

def mapk(y_true, y_pred_probs, k=3):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :k]
    return np.mean([apk(a, p) for a, p in zip(y_true, top_k_preds)])


# KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros((test.shape[0], 7))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_encoded)):
    print(f"{'-'*15} Fold: {fold+1} {'-'*15}")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = create_model(X_train.shape[1])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        callbacks=make_callbacks(),
        verbose=1
    )
    val_pred = model.predict(X_val)
    val_score = mapk(y_val, val_pred, k=3)
    print(f"Validation MAP@3:\t{val_score:.4}")
    test_preds += model.predict(test_scaled, verbose=0) / skf.n_splits


# submission
top_3_indices = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]  # Top 3 in descending order
predicted_labels = [' '.join([id_to_label[idx] for idx in row]) for row in top_3_indices]

submission['Fertilizer Name'] = predicted_labels
submission.to_csv('submission.csv', index=False)
submission.head()




