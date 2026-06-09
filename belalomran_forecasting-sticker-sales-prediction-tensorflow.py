import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid, linear


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test_df.head()


df.info()


df.isna().sum()


df["country"].unique()


df["store"].unique()


df["product"].unique()


test_df.info()


test_df.isna().sum()


df.dropna(inplace=True)
df.isna().sum()


df["country"].replace(['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'], [0,1,2,3,4,5], inplace = True)
df["store"].replace(['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'], [0,1,2], inplace = True)
df["product"].replace(['Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode','Holographic Goose'], [0,1,2,3,4], inplace = True)

test_df["country"].replace(['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'], [0,1,2,3,4,5], inplace = True)
test_df["store"].replace(['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'], [0,1,2], inplace = True)
test_df["product"].replace(['Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode','Holographic Goose'], [0,1,2,3,4], inplace = True)


test_df_copy = test_df.copy()



df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df.drop(['date', 'id'], axis=1, inplace=True)


test_df['date'] = pd.to_datetime(test_df['date'])
test_df['day'] = test_df['date'].dt.day
test_df['month'] = test_df['date'].dt.month
test_df['year'] = test_df['date'].dt.year
test_df.drop(['date', 'id'], axis=1, inplace=True)


df.head()


test_df.head()


x = df.drop('num_sold', axis=1)
y = df['num_sold']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = Sequential([
  Dense(7, activation='relu'),
  Dense(64, activation='relu'),
  Dense(32, activation='relu'),
  Dense(1, activation = "linear")])


model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


history = model.fit(x_train, y_train, epochs=50, batch_size=32)


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



test_predictions = model.predict(test_df)
test_predictions = np.round(test_predictions).astype(int)



y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"R2: {r2}")



submission = pd.DataFrame({
    'id': test_df_copy["id"],
    'num_sold': test_predictions.flatten()
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")




