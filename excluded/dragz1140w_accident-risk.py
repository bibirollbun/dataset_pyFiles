import pandas as pd


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
# train_df.head()


# train_df.isnull().sum()


# train_df.info()


# train_df['road_type'].value_counts()


# train_df['lighting'].value_counts()


# train_df['weather'].value_counts()


# train_df[ 'time_of_day'].value_counts()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


transformer = ColumnTransformer(transformers=[
    ('tnfi' , OrdinalEncoder(categories = [['night', 'daylight', 'dim']]), ['lighting']),
    ('tnf2' , OneHotEncoder(sparse = False, drop = 'first'), ['time_of_day', 'weather', 'road_type'])
], remainder = 'passthrough')


from sklearn.model_selection import train_test_split

x_full = train_df.drop(['id', 'accident_risk'], axis = 1)
y_full = train_df['accident_risk']
x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, test_size = 0.2, random_state = 42)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
x_test = test_df.drop(['id'], axis = 1)


x_train = transformer.fit_transform(x_train)
x_val = transformer.fit_transform(x_val)
x_test = transformer.fit_transform(x_test)


import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=10,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


lgb_model.fit(x_train, y_train)


y_pred = lgb_model.predict(x_val)


from sklearn.metrics import mean_squared_error, r2_score

print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))
print("R²:", r2_score(y_val, y_pred))


import pickle

pickle.dump(lgb_model, open('/kaggle/working/lgb_model.pkl', 'wb'))


y_submission = lgb_model.predict(x_test)


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'num_reported_accidents': y_submission
})


submission_df.head()


submission_df.to_csv('submission2.csv', index=False)




