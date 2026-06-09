import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train.head()


train['Sex'] = (train[['Sex']] == 'female').astype(int)
test['Sex'] = (test[['Sex']] == 'female').astype(int)


def predict_calories(sex, age, height, weight, duration, heart_rate, body_temp):
    predict = ((age * 0.11) - (weight * 0.059) - (sex * 1.31) + (heart_rate * 0.451) - 20.82) * duration / 4.183

    if predict > 1:
        return round(predict, 2)

    return 1.


for (i, row) in train.iterrows():
    _, Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories = row

    pred = predict_calories(Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)

    print(f'Real: {Calories} | Predict: {pred} | Error: {abs(Calories - pred):.2f}')
    
    if i == 6:
        break


train['New_feature'] = ((train['Age'] * 0.11) - (train['Weight'] * 0.059) + (train['Heart_Rate'] * 0.451) - 20.82) * train['Duration'] / 4.183
test['New_feature'] = ((test['Age'] * 0.11) - (test['Weight'] * 0.059) + (test['Heart_Rate'] * 0.451) - 20.82) * test['Duration'] / 4.183


train.head()


X = train.drop(columns=['Calories', 'id'])

y = train['Calories']


model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.005,
    depth=16,
    loss_function='RMSE',
    verbose=100,
    random_seed=0
)

model.fit(X, y)

y_preds = model.predict(X)


y_preds = np.clip(y_preds, 1, 315)

rmsle = np.sqrt(mean_squared_log_error(y, y_preds))
print("RMSLE:", rmsle)


# y_preds = [predict_calories(*X.loc[i].values) if value < 1 else value for i, value in enumerate(y_preds)]

# rmsle = np.sqrt(mean_squared_log_error(y, y_preds))
# print("RMSLE:", rmsle)


X_test= test.drop(columns=['id'])

test_preds = model.predict(X_test)
# test_preds = [predict_calories(*X_test.loc[i].values) if value < 1 else value for i, value in enumerate(test_preds)]
test_preds = np.clip(test_preds, 1, 315)


res = pd.DataFrame({
    "id":test['id'],
    "Calories":test_preds
})

res.to_csv("submission.csv", index=False)


res.head()




