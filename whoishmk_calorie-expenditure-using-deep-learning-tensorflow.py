import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col=0)


df.head()


df.info()


df.isnull().sum()


df.describe()


grouped_sum = df.groupby('Sex')['Calories'].sum()


grouped_sum


# Create a pie chart
grouped_sum.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Calories by')
plt.ylabel('')  # Hide the y-label
plt.show()
          





le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])



df.head()


df.info()


df.head()


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ct = make_column_transformer(
 (MinMaxScaler(),['Sex' , 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']),)


X = df.drop('Calories', axis=1)
y = df['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ct.fit_transform(X_train)


X_train = ct.transform(X_train)
X_test = ct.transform(X_test)


X_train[2]


tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25 , activation = 'relu'),
    tf.keras.layers.Dense(12, activation = 'relu'),
    tf.keras.layers.Dense(6 , activation = 'relu'),
    tf.keras.layers.Dense(3 , activation = 'relu'),
    tf.keras.layers.Dense(1 , activation = 'linear')
])


model.compile(loss = 'mae',
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ['mae'])


history = model.fit(X_train,y_train,validation_data = (X_test , y_test), epochs = 5,batch_size = 32)



import pandas as pd
pd.DataFrame(history.history).plot(figsize=(10, 7));


test_df =pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df.head()


le = LabelEncoder()
test_df['Sex'] = le.fit_transform(test_df['Sex'])


ct.fit_transform(test_df)
X_test = ct.transform(test_df)


X_test[2]


predictions = model.predict(X_test)
predictions[:5]


test_df = test_df.reset_index(drop=True)       # ensure clean index
test_df['prediction'] = predictions
results = (test_df[['id','prediction']])



results.head()


results.to_csv('results.csv',index = False)




