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
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


def preprocess_data(train, test):
    # Encode categorical columns
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove('Fertilizer Name')
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        le_dict[col] = le
    

    le_target = LabelEncoder()
    train['Fertilizer Name'] = le_target.fit_transform(train['Fertilizer Name'])
    class_names = le_target.classes_
    

    X = train.drop(['id', 'Fertilizer Name'], axis=1)
    y = train['Fertilizer Name']
    test_ids = test['id']
    test = test.drop('id', axis=1)
    

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])
    
    return X, y, test, test_ids, class_names, le_target

X, y, test_data, test_ids, class_names, le_target = preprocess_data(train, test)


y_cat = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model(X_train.shape[1], y_cat.shape[1])

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5)
]


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


val_probs = model.predict(X_val)
val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]


val_true = np.argmax(y_val, axis=1)
val_true_labels = le_target.inverse_transform(val_true)
val_pred_labels = [le_target.inverse_transform(pred) for pred in val_top3]

# MAP@3
def mapk(actual, predicted, k=3):
    return np.mean([1 if a in p[:k] else 0 for a, p in zip(actual, predicted)])

val_score = mapk(val_true_labels, val_pred_labels)
print(f"Validation MAP@3: {val_score:.4f}")


test_probs = model.predict(test_data)
test_top3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

test_pred_labels = [' '.join(le_target.inverse_transform(pred)) for pred in test_top3]
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': test_pred_labels
})

submission.to_csv('submission_keras.csv', index=False)

