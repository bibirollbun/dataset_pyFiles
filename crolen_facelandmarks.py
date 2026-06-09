import numpy as np 
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization,LeakyReLU,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError

plt.rcParams["axes.titlesize"] = 10
plt.rcParams["figure.titlesize"] = 20
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

np.random.seed(42)


df0 = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip', compression='zip')
columns = df0.columns.values
RCimg = 96 # Image 96x96 rows and cols


def PlotImages(x,y,y_pred=None):
  sz    = x.shape[1]
  nrows = 5
  ncols = 5
  _, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=[ncols*2,nrows*2])
  for row in range(nrows):
      for col in range(ncols):
          ix = np.random.randint(y.shape[0])
          ax[row,col].imshow(x[ix,:,:], cmap='gray')
          ax[row,col].scatter(y[ix,0::2]*sz,y[ix,1::2]*sz,marker='X',c='r')
          if y_pred is not None:
              ax[row,col].scatter(y_pred[ix,0::2]*sz,y_pred[ix,1::2]*sz,marker='+',c='b')
          ax[row,col].set_title('ix = %d' %(ix))


print(df0.isna().sum())


# OK, a lot of missing values lets drop them
df_cleaned = df0[["left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y", "nose_tip_x","nose_tip_y","mouth_center_bottom_lip_x","mouth_center_bottom_lip_y","Image"]].dropna()
print(df_cleaned.isna().sum())
print(f"\nShape: {df_cleaned.shape}")


y0 = np.array(df_cleaned.drop(columns="Image"))
X0 = []
for i in range(df_cleaned.shape[0]):
   X0.append(np.fromstring(df_cleaned.iloc[i]["Image"], dtype=int, sep=' ').reshape(RCimg,RCimg).astype(np.float64))


# Simple normalization [0..1)
y = y0/RCimg
X = np.reshape(X0,(-1,RCimg,RCimg))/256


PlotImages(X,y)


# Plot distribution of dataset landmarks
alpha = 0.2
mksz = 2
plt.figure(figsize=(5,5))
plt.imshow( np.mean(X,axis=0),cmap='gray')
plt.scatter(y[:,[0,2]]*RCimg-1, y[:,[1,3]]*RCimg-1,c='r',label="Eye center",alpha=alpha,s=mksz)
plt.scatter(y[:,4]*RCimg-1, y[:,5]*RCimg-1,c='b',label="Nose tip",alpha=alpha,s=mksz)
plt.scatter(y[:,6]*RCimg-1, y[:,7]*RCimg-1,c='y',label="Lower lip",alpha=alpha,s=mksz)
plt.gca().legend()
plt.title("Landmarks in dataset")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


model = Sequential()
model.add(Input(shape=(RCimg, RCimg, 1)))
model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='sigmoid'))

model.summary()

model.compile(optimizer='Adam',  metrics=[RootMeanSquaredError(), 'accuracy'], loss='mse')


## Train


model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data = (X_val, y_val), verbose = 1)


score = model.evaluate(X_val, y_val, verbose=2)
print('Val loss:', score[0])
print('Val accuracy:', score[2])


y_pred=model.predict(X_val)

PlotImages(X_val,y_val,y_pred)

