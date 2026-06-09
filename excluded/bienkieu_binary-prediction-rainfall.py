import pandas as pd
import tensorflow as tf


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


data.head()


data = data.drop(columns = ['id', 'day'])


data.columns


data.isna().sum()


data.describe(include = "all")


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


X_predict = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


X_predict = X_predict.drop(columns = ['id', 'day'])


X_predict.isna().sum()


X_predict['winddirection'] = X_predict['winddirection'].fillna(X_predict['winddirection'].mode()[0])


X = data.drop(columns = ['rainfall'])
y = data['rainfall']


minmax = MinMaxScaler()

X = minmax.fit_transform(X)
X_predict = minmax.transform(X_predict)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


input_dim = X_train.shape[1]
input_dim


from tensorflow.keras import models, layers, optimizers

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(7, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),  # optimizer
    loss='binary_crossentropy',                      
    metrics=['accuracy']                             
)


model.fit(X_train, y_train, epochs = 500, batch_size = 32)


from sklearn.metrics import f1_score

# Dự đoán xác suất
y_predict = model.predict(X_test)

# Chuyển thành nhãn 0/1 với threshold 0.5
y_predict_label = (y_predict > 0.5).astype(int)

# Tính F1-score
print(f1_score(y_test, y_predict_label))


dt = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


id_test = dt['id']


y_predict = model.predict(X_predict)


df_result = pd.DataFrame({
    'id': id_test,         
    'rainfall': y_predict.flatten()  # Chuyển dạng (n,1) thành (n,)
})


df_result.to_csv('prediction_result.csv', index=False)

