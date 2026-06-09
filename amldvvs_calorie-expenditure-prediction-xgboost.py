import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.info()


print(train.shape)
print(test.shape)


train.drop('id',axis=1,inplace=True)
test_id=test['id']
test.drop('id',axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder,StandardScaler


le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])
test['Sex']=le.transform(test['Sex'])


x=train.drop('Calories',axis=1)
y=train['Calories']


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
test_scaled=scaler.transform(test)


x_train,x_val,y_train,y_val=train_test_split(x_scaled,y,test_size=0.2,random_state=42)


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)
model.fit(x_train, y_train)


y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


model.fit(x_scaled, y)


predictions = model.predict(test_scaled)


submission = pd.DataFrame({
    'id': test_id,
    'Calories': predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")

