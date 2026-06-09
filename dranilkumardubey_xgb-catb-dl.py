import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print(train.info())
print(train.describe())

# Drop 'User_ID' as it's not predictive
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Check nulls
print(train.isnull().sum())


from scipy.stats import zscore

z_scores = np.abs(zscore(train.select_dtypes(include=np.number)))
train = train[(z_scores < 3).all(axis=1)]


# Encode Gender
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.show()


X = train.drop('Calories', axis=1)
y = train['Calories']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)



def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

grid_xgb = GridSearchCV(xgb, param_grid=params, scoring=rmsle_scorer, cv=3)
grid_xgb.fit(X_train, y_train)
xgb_best = grid_xgb.best_estimator_

pred_xgb = xgb_best.predict(test)


cat = CatBoostRegressor(verbose=0, random_state=42)
cat.fit(X_train, y_train)
pred_cat = cat.predict(test)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
          epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

pred_dl = model.predict(test_scaled).flatten()


final_pred = (0.4 * pred_xgb) + (0.3 * pred_cat) + (0.3 * pred_dl)


submission['Calories'] = final_pred
submission.to_csv('submission.csv', index=False)




