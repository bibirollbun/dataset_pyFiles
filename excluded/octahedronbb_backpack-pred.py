# import
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.feature_selection import mutual_info_regression

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import optuna
import xgboost as xgb
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import joblib

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import RegressorMixin


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# define the hp
train_path = '/kaggle/input/playground-series-s5e2/train.csv'
train_extra_path = '/kaggle/input/playground-series-s5e2/training_extra.csv'
test_path = '/kaggle/input/playground-series-s5e2/test.csv'

rd_state = 1234

model_path = '/kaggle/input/backpack-pred-model/stacking_model.pkl'

original_data_path = '/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv'


# load the data
train_data = pd.read_csv(train_path)
train_extra_data = pd.read_csv(train_extra_path)
test_data = pd.read_csv(test_path)


# load and see the original data
original_data = pd.read_csv(original_data_path)
original_data.head()


# fill the NaN (train data)
def fill_with_mode(data, group_col, target_col):
    mode_fill = data.groupby(group_col)[target_col].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    data.fillna({target_col: mode_fill}, inplace=True)

fill_with_mode(train_data, 'Compartments', 'Size')
fill_with_mode(train_data, 'Compartments', 'Brand')

fill_with_mode(train_data, 'Brand', 'Material')
fill_with_mode(train_data, 'Size', 'Weight Capacity (kg)')
fill_with_mode(train_data, 'Brand', 'Color')
fill_with_mode(train_data, 'Brand', 'Style')
fill_with_mode(train_data, 'Size', 'Laptop Compartment')
fill_with_mode(train_data, 'Brand', 'Waterproof')

train_nan_counts = train_data.isnull().sum()
print(train_nan_counts)


# drop and fill the NaN (original data)
original_data = original_data.dropna(subset=['Compartments', 'Price'])

fill_with_mode(original_data, 'Compartments', 'Size')
fill_with_mode(original_data, 'Compartments', 'Brand')

fill_with_mode(original_data, 'Brand', 'Material')
fill_with_mode(original_data, 'Size', 'Weight Capacity (kg)')
fill_with_mode(original_data, 'Brand', 'Color')
fill_with_mode(original_data, 'Brand', 'Style')
fill_with_mode(original_data, 'Size', 'Laptop Compartment')
fill_with_mode(original_data, 'Brand', 'Waterproof')

original_data_nan_counts = original_data.isnull().sum()
print(original_data_nan_counts)


# fill the NaN (train_extra data)
# it`s not right but i only want to have a try
fill_with_mode(train_extra_data, 'Compartments', 'Size')
fill_with_mode(train_extra_data, 'Compartments', 'Brand')

fill_with_mode(train_extra_data, 'Brand', 'Material')
fill_with_mode(train_extra_data, 'Size', 'Weight Capacity (kg)')
fill_with_mode(train_extra_data, 'Brand', 'Color')
fill_with_mode(train_extra_data, 'Brand', 'Style')
fill_with_mode(train_extra_data, 'Size', 'Laptop Compartment')
fill_with_mode(train_extra_data, 'Brand', 'Waterproof')

train_extra_data_nan_counts = train_extra_data.isnull().sum()
print(train_extra_data_nan_counts)


# trans 'Yes' 'No' to True False
train_data['Laptop Compartment'] = train_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
train_data['Waterproof'] = train_data['Waterproof'].replace({'Yes': True, 'No': False})
train_extra_data['Laptop Compartment'] = train_extra_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
train_extra_data['Waterproof'] = train_extra_data['Waterproof'].replace({'Yes': True, 'No': False})
original_data['Laptop Compartment'] = original_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
original_data['Waterproof'] = original_data['Waterproof'].replace({'Yes': True, 'No': False})


# fill the NaN (test data)
def fill_with_mode_from_train(train_data, test_data, group_col, target_col):
    mode_map = train_data.groupby(group_col)[target_col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    test_data[target_col] = test_data[target_col].fillna(test_data[group_col].map(mode_map))

fill_with_mode_from_train(train_data, test_data, 'Compartments', 'Size')
fill_with_mode_from_train(train_data, test_data, 'Compartments', 'Brand')

fill_with_mode_from_train(train_data, test_data, 'Brand', 'Material')
fill_with_mode_from_train(train_data, test_data, 'Size', 'Weight Capacity (kg)')
fill_with_mode_from_train(train_data, test_data, 'Brand', 'Color')
fill_with_mode_from_train(train_data, test_data, 'Brand', 'Style')
fill_with_mode_from_train(train_data, test_data, 'Size', 'Laptop Compartment')
fill_with_mode_from_train(train_data, test_data, 'Brand', 'Waterproof')

test_nan_counts = test_data.isnull().sum()
print(test_nan_counts)


test_data['Laptop Compartment'] = test_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
test_data['Waterproof'] = test_data['Waterproof'].replace({'Yes': True, 'No': False})


y_train = train_data['Price']
train_data = train_data.drop('Price', axis=1)

y_train_extra = train_extra_data['Price']
train_extra_data = train_extra_data.drop('Price', axis=1)

y_original = original_data['Price']
original_data = original_data.drop('Price', axis=1)


# Encoding
def onehot_encode(df, onehot_cols):
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)    
    return df_encoded
    
onehot_features = ['Brand', 'Material', 'Style', 'Color']

train_data_encoded = onehot_encode(train_data, onehot_features)
onehot_columns = train_data_encoded.columns.tolist()

test_data_encoded = onehot_encode(test_data, onehot_features)
train_extra_data_encoded = onehot_encode(train_extra_data, onehot_features)
original_data_encoded = onehot_encode(original_data, onehot_features)

for col in onehot_columns:
    if col not in test_data_encoded.columns:
        test_data_encoded[col] = 0

for col in onehot_columns:
    if col not in train_extra_data_encoded.columns:
        train_extra_data_encoded[col] = 0

for col in onehot_columns:
    if col not in original_data_encoded.columns:
        original_data_encoded[col] = 0
        
test_data_encoded = test_data_encoded.reindex(columns=onehot_columns, fill_value=0)
train_extra_data_encoded = train_extra_data_encoded.reindex(columns=onehot_columns, fill_value=0)
original_data_encoded = original_data_encoded.reindex(columns=onehot_columns, fill_value=0)

train_data_encoded.head(), test_data_encoded.head()


def ordinal_encode(df, ordinal_cols, ordinal_order):
    df_encoded = df.copy()
    for col in ordinal_cols:
        if col in df_encoded.columns:
            encoder = OrdinalEncoder(categories=[ordinal_order[col]])
            df_encoded[col] = encoder.fit_transform(df_encoded[[col]])   
    return df_encoded

ordinal_features = ['Size']
ordinal_mapping = {'Size': ['Small', 'Medium', 'Large']} 

train_data_encoded = ordinal_encode(train_data_encoded, ordinal_features, ordinal_mapping)
test_data_encoded = ordinal_encode(test_data_encoded, ordinal_features, ordinal_mapping)
train_extra_data_encoded = ordinal_encode(train_extra_data_encoded, ordinal_features, ordinal_mapping)
original_data_encoded = ordinal_encode(original_data_encoded, ordinal_features, ordinal_mapping)

train_data_encoded.head(), test_data_encoded.head()


# define X
X_train = train_data_encoded.drop('id', axis=1)
X_train_extra = train_extra_data_encoded.drop('id', axis=1)
X_train_original = original_data_encoded.drop('id', axis=1)
X_test = test_data_encoded.drop('id', axis=1)


X_train_combined = np.concatenate([X_train, X_train_extra, X_train_original], axis=0)
y_train_combined = np.concatenate([y_train, y_train_extra, y_original], axis=0)


# final estimator in the model
class KerasRegressorWrapper(RegressorMixin):
    def __init__(self, input_dim, epochs=100, batch_size=32, lr=0.01, patience=10):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(self.input_dim,)),  
            layers.Dropout(0.2), 
            layers.Dense(1, kernel_regularizer=regularizers.l2(0.01))  
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def fit(self, X, y):
        early_stopping = EarlyStopping(monitor='loss', patience=self.patience, min_delta=1e-4, restore_best_weights=True)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[early_stopping])
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = self.build_model() 
        return self

meta_learner = KerasRegressorWrapper(input_dim=3, epochs=100, batch_size=32, lr=0.01, patience=25) 


# load the model
stacking_model = joblib.load(model_path)


# predict
stacking_model.fit(X_train_combined, y_train_combined)
price_pred = stacking_model.predict(X_test)
print(price_pred)


# submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Price': price_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




