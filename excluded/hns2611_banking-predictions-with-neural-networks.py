import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train_df = train_df.drop(columns = "id") # don't need ids in train set
train_df.head()


test_df.head()


X = train_df.copy()
y = X.pop('y')


X['month'] = \
    X['month'].map(
        {'jan':1, 'feb': 2, 'mar':3,'apr':4, 'may':5, 'jun':6, 'jul':7,
         'aug':8, 'sep':9, 'oct':10,'nov':11, 'dec':12}
    )

test_df['month'] = \
    test_df['month'].map(
        {'jan':1, 'feb': 2, 'mar':3,'apr':4, 'may':5, 'jun':6, 'jul':7,
         'aug':8, 'sep':9, 'oct':10,'nov':11, 'dec':12}
    )

features_num = ["age", "balance", "day", "month", "duration", "campaign", "pdays", "previous"]
features_cat = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]

transformer_num = StandardScaler()
transformer_cat = OneHotEncoder(handle_unknown = 'ignore')

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify = y, train_size = 0.8)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]


model = keras.Sequential([
    
    layers.Dense(256, activation = "relu", input_shape = input_shape),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(512, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(1024, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(2048, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(1, activation = "sigmoid")
])

plot_model(model, show_shapes = True, show_layer_names = True)


model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["AUC", "binary_accuracy"]
)


early_stopping = keras.callbacks.EarlyStopping(
    patience = 10,
    min_delta = 0.001,
    restore_best_weights = True,
)
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 10000,
    epochs = 500,
    callbacks = [early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['AUC', 'val_AUC']].plot(title = "AUC")


test = test_df.copy()
test_id = test.pop('id')
test = preprocessor.transform(test)
probs = model.predict(test)


submission = pd.DataFrame(test_id)
submission['y'] = probs
submission.to_csv('submission.csv', index = False)

