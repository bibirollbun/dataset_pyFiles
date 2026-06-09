import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder,OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
df.head()


df.info()


df.duplicated().sum()


df.drop("id", axis=1, inplace=True)


X=df.drop("Target", axis=1)
y=df["Target"]


#Using OneHotEncoder because model output has shape (None, 3) (3 classes, one-hot encoded output), 
#but your y_train and y_valid are shaped like (None,) — meaning they are 
#integer class labels, not one-hot vectors.
encoder=OneHotEncoder(sparse_output=False) 
y=encoder.fit_transform(y.values.reshape(-1,1))


scaler=MinMaxScaler()
X=scaler.fit_transform(X)


df["Target"].value_counts()


#Split using (Stratify) to ensures same class distribution (30% from "Graduate", "Dropout", and "Enrolled"     )
X_train, X_dummy, y_train, y_dummy = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
X_valid, X_test,y_valid,y_test=train_test_split(X_dummy, y_dummy, test_size=0.5, random_state=42,stratify=y_dummy)


model=Sequential([
    Dense(256,activation="relu",input_dim=X_train.shape[1]),
    Dense(128,activation="relu"),
    Dense(64,activation="relu"),
    Dense(32,activation="relu"),
    Dense(3,activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.01),loss="categorical_crossentropy",metrics=["accuracy"])

model.summary()


#EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
early_stop=EarlyStopping(
    monitor='val_loss',# what to watch
    patience=5, # wait 5 epochs before stopping
    restore_best_weights=True# roll back to the best model
)

csv_logger=CSVLogger("Training_log.csv",append=True)

reduce_lr=ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5, #reduce lr by 0.5
    patience=3, #waits 3 epochs before reduce lr
    min_lr=1e-6 # don’t go lower than this
)

checkpoint = ModelCheckpoint(
    "best_model.h5",   # file to save
    monitor="val_loss", # what to watch
    save_best_only=True, # only keep the best one
    mode="min"
)

callbacks=[early_stop,csv_logger,reduce_lr,checkpoint ]


history=model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=20,batch_size=36, callbacks=callbacks)



model.evaluate(X_train,y_train)


model.evaluate(X_test,y_test)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

