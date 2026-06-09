# import the basics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_train_basic = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_train_extra  = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df_train = pd.concat([df_train_basic, df_train_extra], axis=0)
df_train.head()


df_train_basic.shape, df_train.shape


# split training data in a training and validation set

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

train_set, valid_set = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)



from sklearn.base import BaseEstimator, TransformerMixin

class BooleanConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.array(X).astype(str)
        X_bool = np.array([[val.strip().lower() == "yes" for val in col] for col in X.T]).T
        return X_bool


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

num_cols = ["Compartments", "Weight Capacity (kg)"]
cat_cols = ["Brand", "Material","Style", "Color"]
bool_cols = ["Laptop Compartment", "Waterproof"]
ord_cols = ["Size"]

pipeline_num = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

pipeline_bool = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("bool_maker", BooleanConverter())
])

size_order = [['Small', 'Medium', 'Large']]
pipeline_ord = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=size_order))
])

transformer = ColumnTransformer(transformers=[
    ('t_num', pipeline_num, num_cols),
    ('t_cat', pipeline_cat, cat_cols),
    ('t_bool', pipeline_bool, bool_cols),
    ('t_ord', pipeline_ord, ord_cols)
])

preprocessor = Pipeline(steps=[
    ("transformer", transformer)
])


# 1. split X and y
X_train = train_set.drop("Price", axis=1).copy()
y_train = train_set["Price"].copy()

# 2. preprocess data
X_train_transformed = preprocessor.fit_transform(X_train)


# 1. split X and y
X_valid = valid_set.drop("Price", axis=1).copy()
y_valid = valid_set["Price"].copy()

# 2. preprocess data
X_valid_transformed = preprocessor.transform(X_valid)


# small training set

train_set_small = train_set.sample(n=300000)

# 1. split X and y
X_train_small = train_set_small.drop("Price", axis=1).copy()
y_train_small = train_set_small["Price"].copy()

# 2. preprocess data
X_train_small_transformed = preprocessor.transform(X_train_small)


# small validation set

valid_set_small = valid_set.sample(n=30000)

# 1. split X and y
X_valid_small = valid_set_small.drop("Price", axis=1).copy()
y_valid_small = valid_set_small["Price"].copy()

# 2. preprocess data
X_valid_small_transformed = preprocessor.transform(X_valid_small)


X_valid_transformed.shape[1:]


import tensorflow as tf
from tensorflow import keras


model = keras.models.Sequential([
    keras.layers.Input(shape=X_train_small_transformed.shape[1:]),
    keras.layers.Dense(25, activation="relu"),
    keras.layers.Dense(25, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(loss="mse",
             optimizer=keras.optimizers.SGD(learning_rate=1e-3))


# find optimal learning rate

K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor=1.005):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        current_lr = float(K.get_value(self.model.optimizer.learning_rate))
        new_lr = current_lr * self.factor
        model.optimizer.learning_rate.assign(new_lr) 


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


expon_lr = ExponentialLearningRate(factor=1.005)


history = model.fit(X_train_small_transformed, y_train_small,
             epochs=1,
             validation_data=(X_valid_small_transformed, y_valid_small),
             callbacks=[expon_lr])


plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
#plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expone_lr.rates))
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.ylim((1e3, 5e3))
plt.show()


def build_model(n_hidden=2, n_neurons=25, 
                learning_rate=1e-1, 
                input_shape=X_train_small_transformed.shape[1:]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


!pip install scikeras


from scikeras.wrappers import KerasRegressor

keras_reg = KerasRegressor(build_model(n_hidden=2, n_neurons=25, 
                learning_rate=1e-3))


final_model = keras_reg

df_train_small = train_set.sample(n=600000)

# prepare the full dataset for fitting
X = df_train_small.drop("Price", axis=1).copy()
y = df_train_small["Price"].copy()

# preprocess and fit the X data
X_transformed = preprocessor.fit_transform(X)
final_model.fit(X_transformed, y,
                epochs=100,
                validation_data=(X_valid_small_transformed, y_valid_small),
                callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# prepare the test set for fitting
# prepare the validation set for fitting
X_test = df_test
# preprocess the X data
X_test_transformed = preprocessor.transform(X_test)
y_pred = final_model.predict(X_test_transformed)

result = pd.DataFrame({
    "id": df_test.id,
    "Price": y_pred
})

result.to_csv('submission.csv', index=False)




