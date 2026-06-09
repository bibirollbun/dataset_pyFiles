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


# define the hp
train_path = '/kaggle/input/playground-series-s5e2/train.csv'
train_extra_path = '/kaggle/input/playground-series-s5e2/training_extra.csv'
test_path = '/kaggle/input/playground-series-s5e2/test.csv'

rd_state = 1234


# load the data
train_data = pd.read_csv(train_path)
train_extra_data = pd.read_csv(train_extra_path)
test_data = pd.read_csv(test_path)


# fill the NaN
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


train_data['Laptop Compartment'] = train_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
train_data['Waterproof'] = train_data['Waterproof'].replace({'Yes': True, 'No': False})


# Encoding
def encode_features(df, onehot_cols=None, ordinal_cols=None, ordinal_order=None):

    df = df.copy()  

    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)  
    
    if ordinal_cols and ordinal_order:
        for col in ordinal_cols:
            if col in ordinal_order:
                encoder = OrdinalEncoder(categories=[ordinal_order[col]])
                df[col] = encoder.fit_transform(df[[col]])
    
    return df


onehot_features = ['Brand', 'Material', 'Style', 'Color']

ordinal_features = ['Size']
ordinal_mapping = {'Size': ['Small', 'Medium', 'Large']} 

train_data_encoded = encode_features(train_data, onehot_cols=onehot_features, ordinal_cols=ordinal_features, ordinal_order=ordinal_mapping)

train_data_encoded.head()


# define X and y
X = train_data_encoded.drop(['id', 'Price'], axis=1)
y = train_data_encoded['Price']


# param
xgb_params = {
    'max_depth': 3, 
    'learning_rate': 0.07198962210587612, 
    'min_child_weight': 4, 
    'n_estimators': 128, 
    'subsample': 0.8225307768843365, 
    'gamma': 2.147797578369717, 
    'reg_lambda': 4.799799811166142, 
    'colsample_bytree': 0.6394262508812985,
    'random_state': rd_state,
}

catboost_params = {
    'iterations': 195, 
    'depth': 4, 
    'learning_rate': 0.1070506205837944, 
    'l2_leaf_reg': 0.0522204253306234, 
    'model_size_reg': 8.553872093460226, 
    'random_strength': 2.526088137691541, 
    'bagging_temperature': 0.7553931664716107, 
    'border_count': 165,
    'random_state': rd_state,
}

hgb_params = {
    'max_iter': 548, 
    'max_depth': 3, 
    'learning_rate': 0.13209199316680825, 
    'min_samples_leaf': 40, 
    'max_bins': 195, 
    'l2_regularization': 0.2751224598813614,
    'random_state': rd_state,
}


# create the model
xgb_model = xgb.XGBRegressor(**xgb_params)
catboost_model = cb.CatBoostRegressor(**catboost_params)
hgb_model = HistGradientBoostingRegressor(**hgb_params)

print(xgb_model, '\n', catboost_model, '\n', hgb_model)


# final estimator
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


# StackingRegressor
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('catboost', catboost_model),
        ('hgb', hgb_model)
    ],
    final_estimator=meta_learner,
    passthrough=False,
    cv=5,
    n_jobs=-1,
)

# stacking_cv_score = cross_val_score(stacking_model, X, y, cv=5, scoring='neg_mean_squared_error')
# print("Stacking CV RMSE:", np.sqrt(-stacking_cv_score.mean()))


joblib.dump(stacking_model, 'stacking_model.pkl')




