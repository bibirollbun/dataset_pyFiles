import pandas as pd
import numpy as np
import gc
from cuml.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from hyperopt import fmin, tpe, hp, Trials

class AgenticAIModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lin_reg = LinearRegression()
        self.best_TE = None
        self.dnn_model = None

    def prepare_data(self, train, test):
        train['Weight Capacity (kg)'] = pd.to_numeric(train['Weight Capacity (kg)'], errors='coerce')
        test['Weight Capacity (kg)'] = pd.to_numeric(test['Weight Capacity (kg)'], errors='coerce')
        train = train.dropna(subset=['Weight Capacity (kg)'])
        test = test.dropna(subset=['Weight Capacity (kg)'])
        X_train = train[['Weight Capacity (kg)']].astype(np.float32)
        y_train = train['Price'].astype(np.float32)
        X_test = test[['Weight Capacity (kg)']].astype(np.float32)
        return X_train, y_train, X_test, test

    def fit_transform_standardize(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_linear_regression(self, X_train_scaled, y_train):
        self.lin_reg.fit(X_train_scaled, y_train)
        gc.collect()

    def predict_linear_regression(self, X_test_scaled):
        predictions = self.lin_reg.predict(X_test_scaled)
        gc.collect()
        return predictions

    def target_encode(self, X_train, y_train):
        space = {
            'n_folds': hp.quniform('n_folds', 5, 50, 1),
            'smooth': hp.quniform('smooth', 1, 100, 1),
            'split_method': hp.choice('split_method', ['random', 'continuous', 'interleaved'])
        }
        
        def target_encode_and_evaluate(params):
            TE = TargetEncoder(
                n_folds=int(params['n_folds']),
                smooth=params['smooth'],
                split_method=params['split_method'],
                stat='mean'
            )
            X_train['pred'] = TE.fit_transform(X_train, y_train)
            s = np.sqrt(mean_squared_error(y_train, X_train['pred']))
            gc.collect()
            return {'loss': s, 'status': 'ok', 'TE': TE}
        
        trials = Trials()
        best = fmin(fn=target_encode_and_evaluate, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        best_trial = trials.best_trial['result']
        self.best_TE = best_trial['TE']
        gc.collect()

    def predict_target_encode(self, X_test):
        predictions = self.best_TE.transform(X_test)
        gc.collect()
        return predictions

    def build_dnn_model(self, input_shape):
        self.dnn_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        self.dnn_model.compile(optimizer='adam', loss='mean_squared_error')
        gc.collect()

    def train_dnn_model(self, X_train_scaled, y_train):
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        self.dnn_model.fit(
            X_train_scaled, 
            y_train, 
            epochs=10, 
            batch_size=16, 
            validation_split=0.2, 
            callbacks=[early_stop],
            verbose=1
        )
        gc.collect()

    def predict_dnn_model(self, X_test_scaled):
        predictions = self.dnn_model.predict(X_test_scaled)
        gc.collect()
        return predictions

    def handle_missing_values(self, predictions):
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions.flatten())
        return predictions.fillna(predictions.mean())




if __name__ == "__main__":
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
    print("Train shape:", train.shape)
    
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
    print("Extra Train shape:", train_extra.shape)
    
    train = pd.concat([train, train_extra], axis=0, ignore_index=True)
    print("Combined Train shape:", train.shape)
    
    test_original = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
    print("Original Test shape:", test_original.shape)
    
    model = AgenticAIModel()
    
    X_train, y_train, X_test, test_processed = model.prepare_data(train, test_original.copy())
    
    X_train_scaled, X_test_scaled = model.fit_transform_standardize(X_train, X_test)
    
    model.target_encode(X_train, y_train)
    predictions_target_encode = model.predict_target_encode(X_test)
    if len(predictions_target_encode) == len(test_original):
        test_original['Price_target_encode'] = predictions_target_encode
    else:
        predictions_target_encode = predictions_target_encode[:len(test_original)]
        test_original['Price_target_encode'] = predictions_target_encode
    
    test_original['Price'] = (
        1.00 * test_original['Price_target_encode']
    )
    
    submission = test_original[['id', 'Price']]
    submission.to_csv("submission.csv", index=False)
    
    print(submission.head())
    print("Submission shape:", submission.shape)





