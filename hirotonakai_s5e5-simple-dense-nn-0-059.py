import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


test.tail()


sample_submission.sample()


train.info()


test.info()


train.describe().T


test.describe().T


train.isnull().sum()


train['Duration'].value_counts()


train.duplicated().sum()


train.head()


train.info()


import seaborn as sns

sns.pairplot(data=train.head(100))
plt.show()


from sklearn.preprocessing import MinMaxScaler,StandardScaler


def preprocess(df):
    df = pd.get_dummies(df,columns=['Sex'])
    #df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['MAX_HR'] = 211 - 0.64 * df['Age']
    df['HR_Percentage'] = df['Heart_Rate'] / df['MAX_HR']
    df['duration_HR'] = df['Duration'] * df['Heart_Rate']
    #Normalize every figures - filter out during parameter tuning
    scaler = StandardScaler()
    df['duration_HR'] = scaler.fit_transform(df[['duration_HR']])
    df['Weight'] = scaler.fit_transform(df[['Weight']])
    df['Height'] = scaler.fit_transform(df[['Height']])
    df['BMI'] = scaler.fit_transform(df[['BMI']])
    df['Heart_Rate'] = scaler.fit_transform(df[['Heart_Rate']])
    df['Duration'] = scaler.fit_transform(df[['Duration']])
    df['Age'] = scaler.fit_transform(df[['Age']])

    return df[[ 'Duration','Heart_Rate','Sex_male','Sex_female'
               ,'BMI','HR_Percentage','Age','Weight']]

y = train[['Calories']]
train = preprocess(train)
train.info()





test_id = test[['id']]
test = preprocess(test)

X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=42)



pd.concat([train, y], axis=1).corr()


train.head(10)


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


model = models.Sequential([
    layers.Dense(32, activation='swish', input_shape=(X_train.shape[1],)),
    layers.Dense(96, activation='swish'),
    layers.Dense(32, activation='swish'),
    layers.Dense(1,  activation='linear') 
])



model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mae',
    metrics=['mae',RootMeanSquaredError()]
)

early_stop = EarlyStopping(
    monitor='mae', 
    patience=3,
    restore_best_weights=True
)


y_train_log = np.log(y_train + 1)  

history = model.fit(
    X_train, y_train_log,
    epochs=100,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("End of Learning ")



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_squared_log_error

#y_pred = model.predict(X_test).flatten()

# revert log
y_pred_log = model.predict(X_test).flatten()
y_pred = np.clip(np.expm1(y_pred_log), 0, None)
mae = mean_absolute_error(y_test, y_pred)
y_true = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")
print(f"RMSLE:{rmsle:.3f} ")




test_pred = model.predict(test).flatten()
test_pred = np.clip(np.expm1(test_pred), 0, None)

ids = test_id['id'].to_numpy()

submission = pd.DataFrame({
    'id': ids,  
    'Calories': test_pred
})


print(submission.describe())

submission.to_csv('submission.csv', index=False)
print("csv exported")

