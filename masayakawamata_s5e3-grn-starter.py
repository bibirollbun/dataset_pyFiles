import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

pd.set_option("display.max_column", 500)
pd.set_option("display.max_row", 500)


import warnings
warnings.simplefilter("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
original['rainfall'] = original['rainfall'].map({'yes': 1, 'no': 0})
original['day'] = (np.arange(len(original)) % 365) + 1
original.head(3)
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Original Shape:", original.shape)
train.head(3)


original.columns = original.columns.str.strip()
desired_order = train.columns.tolist()
original = original[desired_order]
train = pd.concat([train, original], axis=0, ignore_index=True)
train.head(3)


features = train.drop(columns=['rainfall']).columns

train_imputed = train[features].interpolate(method='linear', limit_direction='both')
test_imputed = test[features].interpolate(method='linear', limit_direction='both')

train[features] = train_imputed
test = pd.DataFrame(test_imputed, columns=features)


import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


features = [col for col in train.columns if col != 'rainfall']

train[features] = train[features].fillna(train[features].median())
test = test.fillna(test.median())

X = train[features].values
y = train['rainfall'].values.astype(int)
X_test = test.values


@tf.keras.utils.register_keras_serializable()
def smish(x):
    # smish: x * tanh(log(1 + sigmoid(x)))
    return x * tf.tanh(tf.math.log(1 + tf.sigmoid(x)))


@tf.keras.utils.register_keras_serializable()
class GatedLinearUnit(L.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.linear = L.Dense(units)
        self.sigmoid = L.Dense(units, activation="sigmoid")
        self.units = units

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(L.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.relu_dense = L.Dense(units, activation=smish)
        self.linear_dense = L.Dense(units)
        self.dropout = L.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = L.LayerNormalization()
        self.project = L.Dense(units)

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, inputs):
        x = self.relu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


@tf.keras.utils.register_keras_serializable()
class VariableSelection(L.Layer):
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.grns = []
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = L.Dense(num_features, activation="softmax")
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config['num_features'] = self.num_features
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, inputs):
        v = L.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)
        x = []
        for idx, input_ in enumerate(inputs):
            x.append(self.grns[idx](input_))
        x = tf.stack(x, axis=1)
        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


@tf.keras.utils.register_keras_serializable()
class VariableSelectionFlow(L.Layer):
    def __init__(self, num_features, units, dropout_rate, dense_units=None, **kwargs):
        super().__init__(**kwargs)
        self.variableselection = VariableSelection(num_features, units, dropout_rate)
        self.split = L.Lambda(lambda t: tf.split(t, num_features, axis=-1))
        self.dense = dense_units
        if dense_units:
            self.dense_list = [L.Dense(dense_units, activation='linear') for _ in tf.range(num_features)]
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

    def get_config(self):
        config = super().get_config()
        config['num_features'] = self.num_features
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        config['dense_units'] = self.dense_units
        return config

    def call(self, inputs):
        split_input = self.split(inputs)
        if self.dense:
            l = [self.dense_list[i](split_input[i]) for i in range(len(self.dense_list))]
        else:
            l = split_input
        return self.variableselection(l)


num_features = X.shape[1]

units_1 = 32
drop_1 = 0.75
dense_units = 8

units_2 = 16
drop_2 = 0.5

units_3 = 8
drop_3 = 0.25

inputs = tf.keras.Input(shape=(num_features,))
features_1 = VariableSelectionFlow(num_features, units_1, drop_1, dense_units=dense_units)(inputs)
features_2 = VariableSelectionFlow(units_1, units_2, drop_2)(features_1)
features_3 = VariableSelectionFlow(units_2, units_3, drop_3)(features_2)
outputs = L.Dense(1, activation="sigmoid")(features_3)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
model.summary()

batch_size = 32
epochs = 100

history = model.fit(X, y, validation_split=0.1, epochs=epochs, batch_size=batch_size)

y_pred = model.predict(X_test, batch_size=batch_size)[:, 0]
print("Test predictions shape:", y_pred.shape)


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = y_pred
sub.to_csv("submission.csv", index=False)
sub.head(3)




