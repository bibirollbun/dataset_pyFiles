import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df_train_1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


df_train_1.head()


df_train_2.head()


df_train_1.info()


print(df_train_1.shape)
print(df_train_2.shape)


df_train_1.isna().sum()


df_train_2.isna().sum()


categorical_cols = df_train_1.select_dtypes('object').columns


categorical_cols


for col in categorical_cols:

    print(df_train_1[col].value_counts())
    print('#'* 30)



for col in categorical_cols:

    sns.countplot(x=df_train_1[col])
    plt.title(f'Distribution of [{col}]')
    plt.show()
    print('*' *30)


df_train_1.dropna(inplace=True)
df_train_2.dropna(inplace=True)


df_train_1.isna().sum()


print(df_train_1.head())
print(df_train_2.head())


df_train_1.drop(columns='id', inplace=True)
df_train_2.drop(columns='id', inplace=True)





label_encoder = LabelEncoder()


for col in categorical_cols:

    df_train_1[col] = label_encoder.fit_transform(df_train_1[col])
    df_train_2[col] = label_encoder.fit_transform(df_train_2[col])


df_train_1.head()


corr = df_train_1.corr()


corr


sns.pairplot(df_train_1)
plt.show()


df_train_1.drop(columns=['Style', 'Compartments'], inplace=True)
df_train_2.drop(columns=['Style', 'Compartments'], inplace=True)


df_train_1.head()


df = pd.concat([df_train_1, df_train_2])


x = df.drop(columns='Price')
y = df['Price']


standardizer = StandardScaler()


x = standardizer.fit_transform(x)

x = pd.DataFrame(x)


x.head()


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42, shuffle=True)


x_train.head()





import tensorflow as tf
import keras


kerasModel = keras.models.Sequential([
    keras.layers.Input(shape=(x_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

kerasModel.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])


history = kerasModel.fit(x_train, y_train,
               validation_data=(x_test, y_test),
               epochs=100, # in simple way this mean how many time it should read data
               batch_size=32 # it use to divide data into groups so we can say that number 32 refer to how much every group contain
               , verbose=1, # display detailed result
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   patience=10, #  a technique used to prevent overfitting and improve the model's performance on new
                   monitor= 'val_loss', #if Regression monitor='val_loss'
                   restore_best_weights=True
               )])


y_predict = kerasModel.predict(x_test)


ModelLoss, ModelMSE = kerasModel.evaluate(x_test, y_test)

print(f'ModelLoss ==> {ModelLoss},\n ModelMSE ==> {ModelMSE}')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'], loc='upper right')


df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df_test.head()


df_test.isna().sum()


df_test.drop(columns=['id', 'Style', 'Compartments'], inplace=True)


categorical_cols_test = df_test.select_dtypes('object').columns


for col in categorical_cols_test:

  df_test[col].fillna(df_test[col].mode()[0], inplace=True)

df_test['Weight Capacity (kg)'].fillna(df_test['Weight Capacity (kg)'].mean(), inplace=True)


df_test.isna().sum()


df_test


for col in categorical_cols_test:

  df_test[col] = label_encoder.fit_transform(df_test[col])


scaled_test_df = standardizer.fit_transform(df_test)

test_df = pd.DataFrame(scaled_test_df)


test_df.head()


y_test_predict = kerasModel.predict(test_df)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


len(y_test_predict)


# pd.DataFrame({"id": sample_submission["id"], "Price": y_test_predict.flatten()}).to_csv("sample_submission.csv", index=False)

