import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv',index_col=0)


df.shape


df.head()


df.describe()


df.info()


df.isnull().sum()


x=df.drop('BeatsPerMinute',axis=1)


y=df['BeatsPerMinute']


from sklearn.model_selection import train_test_split 


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.head()


y_train.head()


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()


x_train_scaled=ss.fit_transform(x_train)


x_test_scaled=ss.transform(x_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np


# ✅ Step 2: Define the model
model = Sequential()
model.add(Dense(128, input_dim=x_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.1))  # Slight regularization
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear output for regression

# ✅ Step 3: Compile with a stable optimizer
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])

# ✅ Step 4: Train with validation monitoring
history = model.fit(
    x_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ✅ Step 5: Predict and inverse scale
y_pred_scaled = model.predict(x_test_scaled)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv',index_col=0)



df_test_scaled=ss.transform(df_test)


test_pred=model.predict(df_test)


test_pred


pd.DataFrame(test_pred, columns=['BeatsPerMinute'], index=df_test.index).to_csv('submission.csv')

