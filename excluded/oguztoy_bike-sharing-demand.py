import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV



df1=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
df1.head()


df1.isnull().sum()


df1.head(3)


df1.tail()


df2=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df2.head()


df2.isnull().sum()


df1.shape, df2.shape


df=pd.concat([df1,df2])


df.sample(5)


df.info()


df.describe().T


df["temp_atemp_diff"] = df["temp"] - df["atemp"]


df["temp_humidity_product"] = df["temp"] * df["humidity"]


df["is_windy"] = (df["windspeed"] > 0).astype(int)


plt.figure(figsize=(18,12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")


df.isnull().sum()


df['datetime'] = pd.to_datetime(df['datetime'])


df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['weekday'] = df['datetime'].dt.day_name()


df["rush_hour"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)


df.sample(3)


df.season.unique(),df.holiday.unique(),df.workingday.unique(),df.weather.unique()  


df["season"] = df["season"].astype("category")
df["weather"] = df["weather"].astype("category")
df = pd.get_dummies(df, columns=["season","weather","weekday"], drop_first=True)


abs(df.corr(numeric_only=True)["count"].sort_values(ascending=False))


abs(df.corr(numeric_only=True)["casual"].sort_values(ascending=False))


df.info()


train=df[:10886]
test=df[10886:]


test.isnull().sum()


test.info()


train.isnull().sum()


numerical_cols = train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

plt.figure(figsize=(15, 8))

sns.boxplot(data=train[numerical_cols], orient='h')

plt.title("Boxplot of Numeric Columns")
plt.tight_layout()
plt.show()


sns.boxplot(x=train["count"])


train = train[(train["count"] <680)]


sns.boxplot(x=train["registered"])


train = train[(train["registered"] < 550)]


sns.boxplot(x=train["casual"])


train = train[(train["casual"] < 145)]


features = train.drop(columns=['datetime', 'casual', 'registered', 'count'])
X_train = features
X_test = test[features.columns]

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}


target = 'casual'
y_train = train[target]

model_gpu = XGBRegressor(tree_method="hist", device="cuda",random_state=42)

random_search_casual = RandomizedSearchCV(
    estimator=model_gpu,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

random_search_casual.fit(X_train, y_train)
test['casual'] = np.clip(random_search_casual.predict(X_test), 0, None)


target = 'registered'
y_train = train[target]

random_search_registered = RandomizedSearchCV(
    estimator=model_gpu,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

random_search_registered.fit(X_train, y_train)
print("Best registered params:", random_search_registered.best_params_)

test['registered'] = np.clip(random_search_registered.predict(X_test), 0, None)


test.isnull().sum()


x = train.drop(["datetime","count"], axis=1)
y = np.log1p(train["count"])
test = test.drop(["datetime","count"], axis=1)


scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)


model = Sequential([
    Dense(256, activation='relu', input_shape=(x.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)

early_stop = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.3,patience=3, min_lr=1e-11)
model.fit(x, y, epochs=600, batch_size=64, callbacks=[early_stop,reduce_lr], verbose=1)


prediction_log = model.predict(test)
prediction = np.expm1(prediction_log)


prediction


prediction = prediction.round()
prediction = np.clip(prediction, 0, None)


prediction


submission = pd.DataFrame({
    "datetime": df2["datetime"].values,
    "count": prediction.flatten()
})


submission.head()


submission.to_csv("submission.csv", index=False)


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error


def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)



xgb = XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    verbosity=0
)

xgb_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.05],
    'max_depth': [4, 6],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring=rmsle_scorer, n_jobs=-1, verbose=1)
xgb_grid.fit(x_train, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(x_test)
xgb_rmsle = rmsle(y_test, xgb_preds)

print("----- XGBoost Results -----")
print(f"Best Params: {xgb_grid.best_params_}")
print(f"Train RMSLE (CV mean): {-xgb_grid.best_score_:.4f}")
print(f"Test RMSLE: {xgb_rmsle:.4f}\n")


cat = CatBoostRegressor(
    task_type="GPU",
    devices="0",
    verbose=0
)

cat_params = {
    'iterations': [300, 500],
    'learning_rate': [0.01, 0.05],
    'depth': [6, 8]
}

cat_grid = GridSearchCV(cat, cat_params, cv=3, scoring=rmsle_scorer, n_jobs=1, verbose=1)
cat_grid.fit(x_train, y_train)

cat_best = cat_grid.best_estimator_
cat_preds = cat_best.predict(x_test)
cat_rmsle = rmsle(y_test, cat_preds)

print("----- CatBoost Results -----")
print(f"Best Params: {cat_grid.best_params_}")
print(f"Train RMSLE (CV mean): {-cat_grid.best_score_:.4f}")
print(f"Test RMSLE: {cat_rmsle:.4f}")


rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=3,
    scoring=rmsle_scorer,
    verbose=1,
    n_jobs=-1
)

rf_grid.fit(x_train, y_train)

best_rf = rf_grid.best_estimator_
rf_preds = best_rf.predict(x_test)

rf_rmsle = rmsle(y_test, rf_preds)

print("----- Random Forest Results -----")
print(f"Best Params: {rf_grid.best_params_}")
print(f"Train RMSLE (CV mean): {-rf_grid.best_score_:.4f}")
print(f"Test RMSLE: {rf_rmsle:.4f}")


prediction_log = best_rf.predict(test)
prediction = np.expm1(prediction_log)


prediction


prediction = prediction.round()
prediction = np.clip(prediction, 0, None)
prediction


submission = pd.DataFrame({
    "datetime": df2["datetime"].values,
    "count": prediction.flatten()
})


submission.to_csv("submission2.csv", index=False)
submission.head()


prediction_log = xgb_best.predict(test)
prediction = np.expm1(prediction_log)
prediction


prediction = prediction.round()
prediction = np.clip(prediction, 0, None)
prediction


submission = pd.DataFrame({
    "datetime": df2["datetime"].values,
    "count": prediction.flatten()
})


submission.to_csv("submission3.csv", index=False)
submission.head()




