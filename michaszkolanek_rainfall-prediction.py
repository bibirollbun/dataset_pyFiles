import os
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import optuna


warnings.filterwarnings('ignore', category=FutureWarning)


FS_OPTIONS = ["rf", "pca", "all"]
MODEL_OPTIONS = ["xgb", "stacked", "nn", "lstm"]

FEATURE_SELECTION = FS_OPTIONS[0]
MODEL = MODEL_OPTIONS[2]
VISUALIZATION = False
VALIDATION = True
FIND_PARAMETERS = False


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_data.info()


train_data.head()


train_data.describe()


train_data.duplicated().sum()


def data_cleaning(data):
    data.rename(columns={"temparature": "temperature"}, inplace=True)
    data['winddirection'] = data['winddirection'].fillna(data['winddirection'].mean())


def feature_engineering(data, rainfall_data=None):
    bins = [1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data["month"] = pd.cut(data["day"].astype("int64"), bins=bins, labels=labels, right=False)
    if rainfall_data is not None:
        data['mean_rainfall'] = rainfall_data.groupby('month')['rainfall'].transform('mean')
    else:
        data['mean_rainfall'] = data.groupby('month')['rainfall'].transform('mean')
    data["day_sin"] = np.sin(2 * np.pi * data["day"] / 365)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 365)
    data["temp_range"] = data["maxtemp"] - data["mintemp"] 
    data["wind_x"] = data["windspeed"] * np.cos(data["winddirection"] * math.pi / 180)
    data["wind_y"] = data["windspeed"] * np.sin(data["winddirection"] * math.pi / 180)
    data["dewpoint_diff"] = data["temperature"] - data["dewpoint"]
    data["high_humidity"] = (data["humidity"] > 90).astype(int)
    
    data['temp_humidity_interaction'] = data['temperature'] * data['humidity']
    data['cloud_sunshine_ratio'] = data['cloud'] / (data['sunshine'] + 1e-6)
    data['wind_strength'] = np.sqrt(data['wind_x']**2 + data['wind_y']**2)
    data['weekofyear'] = data['day'] // 7

    kmeans = KMeans(n_clusters=5, random_state=42)
    data['weather_cluster'] = kmeans.fit_predict(data[['temperature', 'humidity', 'windspeed', 'cloud']])


def scaling(data, method):
    float_columns = data.select_dtypes(include=['float64']).columns
    if method == "MinMax":
        scaler = MinMaxScaler()
        data[float_columns] = scaler.fit_transform(data[float_columns])


def lag_feature_add(df_train, df_test):
    lag_features = ['temperature', 'humidity', 'cloud', 'windspeed']
    df_train['rainfall_lag1'] = df_train['rainfall'].shift(1)
    df_train['rainfall_lag1'].fillna(df_train['rainfall'].mean(), inplace=True)

    lag_model = XGBRegressor()
    lag_model.fit(df_train[lag_features], df_train['rainfall_lag1'])
    df_test['rainfall_lag1'] = lag_model.predict(df_test[lag_features])


data_cleaning(train_data)
data_cleaning(test_data)
feature_engineering(train_data)
feature_engineering(test_data, train_data)
lag_feature_add(train_data, test_data)
scaling(train_data, "MinMax")
scaling(test_data, "MinMax")


train_data.columns


train_data.info()


float_columns = train_data.select_dtypes(include=['float64']).columns


def box_visualization(data, columns, target):
    rows = round(len(columns) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows*4))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x=target, y=col, data=data, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


def hist_visualization(data, columns, target=None):
    rows = round(len(columns) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows*4))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(data, x=col, hue=target, kde=True, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


def heatmap_visualization(data, columns):
    plt.figure(figsize=(16, 16))
    sns.heatmap(data[columns].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


if VISUALIZATION:
    box_visualization(train_data, float_columns, "rainfall")


if VISUALIZATION:
    hist_visualization(train_data, float_columns)


if VISUALIZATION:
    hist_visualization(train_data, float_columns, target="rainfall")


if VISUALIZATION:
    heatmap_visualization(train_data, float_columns)


if FEATURE_SELECTION == "rf":
    rfe = RFE(RandomForestRegressor(), n_features_to_select=18)
    rfe.fit(train_data[train_data.select_dtypes(include=['float64']).columns], train_data["rainfall"])
    SELECTED_FEATURES = train_data.select_dtypes(include=['float64']).columns[rfe.support_]
elif FEATURE_SELECTION == "all" or FEATURE_SELECTION == "pca":
    SELECTED_FEATURES = train_data.columns.difference(['id', 'day', 'month', 'rainfall']).tolist()
print(SELECTED_FEATURES)


if VALIDATION:
    X_train, X_val, y_train, y_val = train_test_split(train_data[SELECTED_FEATURES], train_data["rainfall"])
else:
    X_train, y_train = train_data[SELECTED_FEATURES], train_data["rainfall"]
X_test = test_data[SELECTED_FEATURES]
if FEATURE_SELECTION == "pca":
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    if VALIDATION:
        X_val = pca.fit_transform(X_val)


if MODEL == "lstm":
    def reshape_for_lstm(X, time_steps):
        X_lstm = []
        for i in range(len(X) - time_steps):
            X_lstm.append(X[i:i + time_steps])  # Create sequences
        return np.array(X_lstm)
        
    X_data = [X_train, X_val]
    y_data = [y_train, y_val]
    
    for i, X in enumerate(X_data):
        if "day" not in X.columns:
            X["day"] = train_data["day"]  # Ensure day exists
        X.drop(columns=['day'], inplace=True)
    
        X_current_values = X[SELECTED_FEATURES].values
        y_current_values = y_data[i].values
    
        time_steps = 60
        X_lstm = reshape_for_lstm(X_current_values, time_steps)
        y_lstm = y_current_values[time_steps:]  # Adjust target variable
    
        print(f"Processed X shape: {X_lstm.shape}")  # Debugging
    
        if i == 0:
            X_train = X_lstm
            y_train= y_lstm
        else:
            X_val = X_lstm
            y_val = y_lstm
    
    print(f"Final X_train shape: {X_train.shape}")  # Must be (samples, time_steps, features)
    print(f"Final y_train shape: {y_train.shape}")  # Must match samples


if FIND_PARAMETERS:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.0001, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.0001, 10.0),
            'gamma': trial.suggest_uniform('gamma', 0, 5)
        }
        
        xgb_model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
        score = cross_val_score(xgb_model, X_train, y_train, scoring='roc_auc', cv=5).mean()
        return score
    
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective, n_trials=50)
    
    best_xgb_params = study_xgb.best_params
    print("Best XGBoost Parameters:", best_xgb_params)


best_xgb_params ={'n_estimators': 700, 'learning_rate': 0.012666576542289873, 'max_depth': 4, 'min_child_weight': 6, 'subsample': 0.7079029281779418, 'colsample_bytree': 0.5276164143410431, 'reg_alpha': 0.008616356812419073, 'reg_lambda': 1.1922208237664227, 'gamma': 0.4990776423819912}


if FIND_PARAMETERS:
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        rf_model = RandomForestRegressor(**params, random_state=42)
        score = cross_val_score(rf_model, X_train, y_train, scoring='roc_auc', cv=5).mean()
        return score
    
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=50)
    
    best_rf_params = study_rf.best_params
    print("Best Random Forest Parameters:", best_rf_params)


best_rf_params = {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 9, 'min_samples_leaf': 9, 'max_features': 'log2', 'bootstrap': True}


if MODEL == "xgb":
    model = xgb.XGBRegressor(**best_xgb_params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=40)


if MODEL == "stacked":
    xgb_model = xgb.XGBRegressor(**best_xgb_params, objective='reg:squarederror', random_state=42)
    rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
    base_learners = [
    ('lr', LinearRegression()),
    ('xgb', xgb_model),
    ('rf', rf_model)
    ]
    
    meta_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        reg_alpha=0.1,
        random_state=42
    )
    model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)
    model.fit(X_train, y_train)


if MODEL == "nn":
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
    )
    
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
    
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])


if MODEL == "lstm":
    input_shape = (X_lstm.shape[1], X_lstm.shape[2])
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(256, return_sequences=True),
        Dropout(0.2),
        GRU(128),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))


if VALIDATION:
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)
    
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    roc_auc_val = auc(fpr_val, tpr_val)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for Validation Set')
    ax1.legend(loc='lower right')
    ax1.grid()
    
    ax2.plot(fpr_train, tpr_train, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc_train:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve for Training Set')
    ax2.legend(loc='lower right')
    ax2.grid()
    
    plt.tight_layout()
    plt.show()


y_pred = model.predict(X_test)


if MODEL == "xgb" or MODEL == "stacked":
    y_pred = np.clip(y_pred, 0, 1)
elif MODEL == "lstm" or MODEL == "nn":
    y_pred = y_pred.flatten().tolist()


predicted_data = pd.DataFrame(data={'rainfall': y_pred}, index=test_data["id"])
predicted_data.to_csv('prediction.csv') 

