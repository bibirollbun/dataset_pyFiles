import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import lightgbm as lgb

class BaseModel:
    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed

    def label_encode(self, train_data, test_data, cat_cols):
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            combined_data = pd.concat([train_data[col], test_data[col]], axis=0)
            le.fit(combined_data)
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            label_encoders[col] = le
        return train_data, test_data

class LGBMModel(BaseModel):
    def __init__(self, params, n_splits=5, early_stopping_rounds=50, seed=42):
        super().__init__(n_splits, seed)
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds

    def train(self, train_data, test_data, cat_cols):
        print("Training LightGBM...")
        train_data['num_sold'] = np.log1p(train_data['num_sold'])
        X = train_data.drop(['num_sold'], axis=1)
        y = train_data['num_sold']
        test_features = test_data.drop(columns=['id'], errors='ignore')

        # Align test_features to match training data columns
        test_features = test_features.reindex(columns=X.columns, fill_value=0)

        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        scores = []
        test_preds = []
        oof_preds = np.zeros(X.shape[0])

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**self.params)
            callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=0)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                categorical_feature=cat_cols,
                callbacks=callbacks
            )

            val_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, val_pred, squared=False)
            scores.append(rmse)

            oof_preds[val_idx] = val_pred
            test_pred = model.predict(test_features)
            test_preds.append(test_pred)

        print(f"LightGBM Mean RMSE: {np.mean(scores):.5f}")
        return np.mean(test_preds, axis=0), oof_preds


class XGBModel(BaseModel):
    def __init__(self, params, n_splits=5, seed=42):
        super().__init__(n_splits, seed)
        self.params = params

    def train(self, train_data, test_data):
        print("Training XGBoost...")
        train_data['num_sold'] = np.log1p(train_data['num_sold'])
        X = train_data.drop(['num_sold'], axis=1)
        y = train_data['num_sold']
        test_features = test_data.drop(columns=['id'], errors='ignore')

        # Align test_features to match training data columns
        test_features = test_features.reindex(columns=X.columns, fill_value=0)

        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        scores = []
        test_preds = []
        oof_preds = np.zeros(X.shape[0])

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(**self.params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', verbose=0)

            val_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, val_pred, squared=False)
            scores.append(rmse)

            oof_preds[val_idx] = val_pred
            test_pred = model.predict(test_features)
            test_preds.append(test_pred)

        print(f"XGBoost Mean RMSE: {np.mean(scores):.5f}")
        return np.mean(test_preds, axis=0), oof_preds


class NeuralNetworkModel(BaseModel):
    def __init__(self, input_dim, n_splits=5, seed=42):
        super().__init__(n_splits, seed)
        self.input_dim = input_dim

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        return model

    def train(self, train_data, test_data):
        print("Training Neural Network...")
        train_data['num_sold'] = np.log1p(train_data['num_sold'])
        X = train_data.drop(['num_sold'], axis=1)
        y = train_data['num_sold']
        test_features = test_data.drop(columns=['id'], errors='ignore')

        # Align test_features to match training data columns
        test_features = test_features.reindex(columns=X.columns, fill_value=0)

        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        scores = []
        test_preds = []
        oof_preds = np.zeros(X.shape[0])

        for train_idx, val_idx in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.build_model()
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=500,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )

            val_pred = model.predict(X_val).flatten()
            rmse = mean_squared_error(y_val, val_pred, squared=False)
            scores.append(rmse)

            oof_preds[val_idx] = val_pred
            test_pred = model.predict(test_features).flatten()
            test_preds.append(test_pred)

        print(f"Neural Network Mean RMSE: {np.mean(scores):.5f}")
        return np.mean(test_preds, axis=0), oof_preds


if __name__ == "__main__":
    # Example usage
    train_path = '/kaggle/input/andro-preprocess-sticker-forecasting-competition/train_data.csv'
    test_path = '/kaggle/input/andro-preprocess-sticker-forecasting-competition/test_data.csv'

    train_data = pd.read_csv(train_path, low_memory=False)
    test_data = pd.read_csv(test_path, low_memory=False)

    cat_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    base_model = BaseModel()
    train_data, test_data = base_model.label_encode(train_data, test_data, cat_cols)

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.08833,
        'max_depth': 13,
        'reg_alpha': 0.01,
        'lambda_l2': 0.01,
        'min_child_samples': 32,
        'colsample_bytree': 0.93,
        'subsample': 0.7,
        'seed': 42,
        'verbose': -1
    }

    xgb_params = {
        'tree_method': 'hist',
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.008,
        'random_state': 42
    }

    # LightGBM
    # lgbm_model = LGBMModel(lgbm_params)
    #lgbm_preds, lgbm_oof = lgbm_model.train(train_data, test_data, cat_cols)

    # XGBoost
    xgb_model = XGBModel(xgb_params)
    xgb_preds, xgb_oof = xgb_model.train(train_data, test_data)

    # Neural Network
   # nn_model = NeuralNetworkModel(input_dim=train_data.shape[1] - 1)
   # nn_preds, nn_oof = nn_model.train(train_data, test_data)

    # Combine predictions with weights
    # lgbm_weight = 0.4
    xgb_weight = 1.0
    #nn_weight = 0.3

    final_preds = (
                   xgb_weight * xgb_preds 
    
    ) * 1.00375

    # Save submission file
    submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1(final_preds)})
    submission.to_csv('submission.csv', index=False)
    print("submission.csv successfully saved!!")
    print(submission.head())








