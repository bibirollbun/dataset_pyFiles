import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import optuna
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_df = pd.concat([train_df, train_extra_df])


print(train_df.info())
print(train_df.describe())


# Xử lí giá trị thiếu
object_cols = train_df.select_dtypes(include='object').columns
for col in object_cols:
    train_df[col] = train_df[col].fillna('missing')
    test_df[col] = test_df[col].fillna('missing')

SI = SimpleImputer(strategy='mean')
train_df['Weight Capacity (kg)'] = SI.fit_transform(train_df['Weight Capacity (kg)'].values.reshape(-1, 1)).reshape(-1,)
test_df['Weight Capacity (kg)'] = SI.transform(test_df['Weight Capacity (kg)'].values.reshape(-1, 1)).reshape(-1,)

# Mã hóa các cột categorical
TE = TargetEncoder()
for col in object_cols:
    TE.fit(train_df[col], train_df['Price'])
    train_df[col] = TE.transform(train_df[col])
    test_df[col] = TE.transform(test_df[col])


X = train_df.drop('Price', axis=1)
y = train_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_test_df = scaler.transform(test_df)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

rmse_scores = {}
for name, model in models.items():
    model.fit(scaled_X_train, y_train)
    pred = model.predict(scaled_X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    rmse_scores[name] = rmse
    print(f'{name} RMSE: {rmse}')


i = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(i)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dense(1)(x)

model = Model(i, x)
model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])
model.fit(scaled_X_train, y_train, validation_data=(scaled_X_test, y_test), epochs=25, batch_size=2048, verbose=0)

dl_pred = model.predict(scaled_X_test, verbose=0)
dl_rmse = mean_squared_error(y_test, dl_pred, squared=False)
rmse_scores['Deep Learning (cơ bản)'] = dl_rmse
print(f'Deep Learning RMSE (cơ bản): {dl_rmse}')


# Tối ưu hóa hyperparameters cho mô hình Deep Learning bằng Optuna
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    units = trial.suggest_int('units', 64, 512)
    
    i = Input(shape=(X_train.shape[1],))
    x = i
    for _ in range(n_layers):
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(1)(x)
    
    model = Model(i, x)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['RootMeanSquaredError'])
    model.fit(scaled_X_train, y_train, validation_data=(scaled_X_test, y_test), epochs=10, batch_size=2048, verbose=0)
    
    val_loss = model.evaluate(scaled_X_test, y_test, verbose=0)[0]
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Hyperparameters tốt nhất:', study.best_params)
print('RMSE tốt nhất:', study.best_value)


# Huấn luyện lại mô hình Deep Learning với hyperparameters tối ưu
best_params = study.best_params
i = Input(shape=(X_train.shape[1],))
x = i
for _ in range(best_params['n_layers']):
    x = Dense(best_params['units'], activation='relu')(x)
    x = Dropout(best_params['dropout_rate'])(x)
x = Dense(1)(x)

best_model = Model(i, x)

best_model.compile(
    loss='mse', 
    optimizer=tf.keras.optimizers.Adam(best_params['learning_rate']), 
    metrics=['RootMeanSquaredError']
)

best_model.fit(scaled_X_train,
               y_train, 
               validation_data=(scaled_X_test, y_test), 
               epochs=50, 
               batch_size=64, 
               verbose=0
)


best_model_name = min(rmse_scores, key=rmse_scores.get)
best_rmse = rmse_scores[best_model_name]
print(f'Mô hình tốt nhất: {best_model_name} với RMSE: {best_rmse}')

if best_model_name == 'Deep Learning (tối ưu)':
    output = best_model.predict(scaled_test_df, verbose=0)
    preds = output[:, 0]
elif best_model_name in models:
    best_model = models[best_model_name]
    preds = best_model.predict(scaled_test_df)
else:
    print("Không tìm thấy mô hình tốt nhất.")

sub = pd.DataFrame({'id': test_df['id'], 'Price': preds})
sub.to_csv("submission.csv", index=False)
print('File submission.csv đã được lưu thành công!')

