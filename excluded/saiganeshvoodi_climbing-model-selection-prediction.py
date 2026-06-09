import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import random


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


for df in [train, test]:
    df['Height_m'] = df['Height'] / 100
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Heart_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['BMI_Heart'] = df['BMI'] * df['Heart_Rate']
    df['BMI_Duration'] = df['BMI'] * df['Duration']
    df['Temp_Weight'] = df['Body_Temp'] * df['Weight']
    df.drop(columns=['Height_m'], inplace=True)


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])



X = train.drop(columns=['id', 'Calories'])
y = train['Calories']
X_test = test.drop(columns=['id'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Hill Climbing for XGBoost
def run_xgb_hill():
    best_score = float('inf')
    best_model = None
    for _ in range(5):
        params = {
            'n_estimators': random.choice([100, 200, 300]),
            'max_depth': random.choice([4, 6, 8]),
            'learning_rate': random.choice([0.01, 0.05, 0.1]),
        }
        model = XGBRegressor(**params, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = rmsle(y_val, preds)
        if score < best_score:
            best_score = score
            best_model = model
    return best_score, best_model

# Hill Climbing for LGBM
def run_lgbm_hill():
    best_score = float('inf')
    best_model = None
    for _ in range(5):
        params = {
            'n_estimators': random.choice([100, 200, 300]),
            'max_depth': random.choice([4, 6, 8]),
            'learning_rate': random.choice([0.01, 0.05, 0.1]),
        }
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = rmsle(y_val, preds)
        if score < best_score:
            best_score = score
            best_model = model
    return best_score, best_model

# Hill Climbing for RandomForest
def run_rf_hill():
    best_score = float('inf')
    best_model = None
    for _ in range(5):
        params = {
            'n_estimators': random.choice([100, 200, 300]),
            'max_depth': random.choice([6, 10, None]),
        }
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = rmsle(y_val, preds)
        if score < best_score:
            best_score = score
            best_model = model
    return best_score, best_model

# Hill Climbing for Keras MLP
def run_mlp_hill():
    best_score = float('inf')
    best_model = None
    for _ in range(5):
        tf.keras.backend.clear_session()
        layers = random.choice([(128, 64), (64, 32), (256, 128)])
        dropout = random.uniform(0.1, 0.4)
        lr = random.choice([0.001, 0.0005])

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
        for u in layers:
            model.add(tf.keras.layers.Dense(u, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )
        model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
        preds = model.predict(X_val).flatten()
        score = rmsle(y_val, preds)
        if score < best_score:
            best_score = score
            best_model = model
    return best_score, best_model


# Runs all models
results = {}

xgb_score, xgb_model = run_xgb_hill()
results['XGBoost'] = (xgb_score, xgb_model)

lgb_score, lgb_model = run_lgbm_hill()
results['LGBM'] = (lgb_score, lgb_model)

rf_score, rf_model = run_rf_hill()
results['RandomForest'] = (rf_score, rf_model)

mlp_score, mlp_model = run_mlp_hill()
results['MLP'] = (mlp_score, mlp_model)


best_model_name = min(results, key=lambda k: results[k][0])
best_score, best_model = results[best_model_name]

print(f"\n Best Model: {best_model_name} with RMSLE = {best_score:.4f}")


if best_model_name == 'MLP':
    best_model.fit(X_scaled, y, epochs=30, batch_size=64, verbose=0)
    preds = best_model.predict(X_test_scaled).flatten()
else:
    best_model.fit(X_scaled, y)
    preds = best_model.predict(X_test_scaled)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(0, preds)
})
submission.to_csv('submission.csv', index=False)
print(" submission.csv saved")





