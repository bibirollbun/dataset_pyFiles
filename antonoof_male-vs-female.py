import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train.head()


train['Sex'] = (train[['Sex']] == 'female').astype(int)
test['Sex'] = (test[['Sex']] == 'female').astype(int)


train_male = train[train['Sex'] == 0]
train_female = train[train['Sex'] == 1]

test_male = test[test['Sex'] == 0]
test_female = test[test['Sex'] == 1]


X_male = train_male.drop(columns=['Calories', 'id'])
y_male = train_male['Calories']

X_female = train_female.drop(columns=['Calories', 'id'])
y_female = train_female['Calories']


model_male = CatBoostRegressor(
    iterations=7000,
    learning_rate=0.015,
    depth=9,
    l2_leaf_reg=8,
    loss_function='RMSE',
    verbose=100,
    random_seed=0
)

model_male.fit(X_male, y_male)

y_preds = model_male.predict(X_male)
y_preds = np.clip(y_preds, 1, 314)

rmsle = np.sqrt(mean_squared_log_error(y_male, y_preds))
print("RMSLE:", rmsle)


model_female = CatBoostRegressor(
    iterations=7000,
    learning_rate=0.015,
    depth=9,
    l2_leaf_reg=8,
    loss_function='RMSE',
    verbose=100,
    random_seed=0
)

model_female.fit(X_female, y_female)

y_preds = model_female.predict(X_female)
y_preds = np.clip(y_preds, 1, 314)

rmsle = np.sqrt(mean_squared_log_error(y_female, y_preds))
print("RMSLE:", rmsle)


X_test_male = test_male.drop(columns=['id'])

male_preds = model_male.predict(X_test_male)
male_preds = np.clip(male_preds, 1, 315)


X_test_female = test_female.drop(columns=['id'])

female_preds = model_female.predict(X_test_female)
female_preds = np.clip(female_preds, 1, 315)


np.concatenate([male_preds, female_preds])


res = pd.DataFrame({
    "id": np.concatenate([test_male['id'], test_female['id']]),
    "Calories": np.concatenate([male_preds, female_preds])
})

res.to_csv("submission.csv", index=False)


res.head()




