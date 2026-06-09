import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from tensorflow import keras
from keras import layers


zip_train = '../input/facial-keypoints-detection/training.zip'
zip_test = '../input/facial-keypoints-detection/test.zip'
extract_to_path = '../working/'

with zipfile.ZipFile(zip_train, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
with zipfile.ZipFile(zip_test, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)


data = pd.read_csv('../working/training.csv')
data.info()


all_image = []
for all_pixel in data['Image'].values:
    image = list(map(int,all_pixel.split(' ')))
    all_image.append(image)


#image = [list(map(int, all_pixel.split(' '))) for all_pixel in data['Image'].values]


all_image = np.array(all_image).reshape(-1,96,96)


def visual_key_point(row=5,col=5):
    fig,ax = plt.subplots(row,col,figsize=(8,6))
    ax=ax.flatten()
    for i in range(len(ax)):
        sample = np.random.randint(0,all_image.shape[0])
        ax[i].imshow(all_image[sample],cmap='gray')
        x = data.iloc[sample,[i for i in range(0,30,2)]]
        y = data.iloc[sample,[i+1 for i in range(0,30,2)]]
        ax[i].scatter(x,y)
        ax[i].axis('off')

    plt.tight_layout()

visual_key_point(2,5)


data_new = data.dropna()
data_new.info()


value = list(range(0, 4)) + list(range(20, 22)) + list(range(28,30))
target = list(range(4,20)) + list(range(22, 28))
x_train,x_test,y_train,y_test = train_test_split(data_new.iloc[:,value],data_new.iloc[:,target],
                                                test_size = 0.1,random_state = 42)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


def evaluation(model,x_test,y_test):
    print(model.__class__.__name__)  
    print('--------------------')
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    print('MAE: {}'.format(mae))
    print('MSE: {}'.format(mse))
    print('R2: {}'.format(r2))

    return [mae,mse,r2]


xgb = XGBRegressor()
xgb.fit(x_train,y_train)
score_xgb = evaluation(xgb,x_test,y_test)


y_pred = xgb.predict(data.iloc[:,value])


def visual_key_point(row=5,col=5):
    fig,ax = plt.subplots(row,col,figsize=(16,10))
    ax=ax.flatten()
    for i in range(len(ax)):
        sample = np.random.randint(0,all_image.shape[0])
        ax[i].imshow(all_image[sample],cmap='gray')
        x = data.iloc[sample,[i for i in range(0,30,2)]]
        y = data.iloc[sample,[i+1 for i in range(0,30,2)]]
        ax[i].scatter(x,y)
        
        x = [y_pred[sample][i] for i in range(0,len(target),2)]
        y = [y_pred[sample][i+1] for i in range(0,len(target),2)]
        ax[i].scatter(x,y)
        ax[i].axis('off')
    
    plt.tight_layout()

visual_key_point(5,5)


data_values = data.iloc[:, target].values
y_pred_values = np.array(y_pred) 
data.iloc[:, target] = np.where(np.isnan(data_values), y_pred_values, data_values)


data_new = data.dropna()
data_new.shape


data_new.info()


all_image = []
for all_pixel in data_new['Image'].values:
    image = list(map(int,all_pixel.split(' ')))
    all_image.append(image)


all_image = np.array(all_image).reshape(-1,96,96)


target = data_new.drop('Image',axis=1).values
x_train,x_test,y_train,y_test = train_test_split(all_image,target,
                                                test_size = 0.1,
                                                random_state = 42)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


layers.Conv2D


class CNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = self.block_conv2d(16,16)
        self.conv2 = self.block_conv2d(32,32)
        self.conv3 = self.block_conv2d(64,64)
        self.conv4 = self.block_conv2d(128,128)

        self.fc = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation = 'leaky_relu'),
            layers.Dense(256, activation = 'leaky_relu'),
            layers.Dense(30)
        ])

    def block_conv2d(self,in_channel,out_channel):
        return keras.Sequential([
            layers.Conv2D(in_channel,3),
            layers.BatchNormalization(),
            layers.Activation('leaky_relu'),
            layers.Conv2D(out_channel,3),
            layers.BatchNormalization(),
            layers.Activation('leaky_relu'),
            layers.MaxPooling2D((2,2))
        ])

    def call(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


model = CNN()
model(np.random.rand(32,96,96,1))
model.summary()


learning_rate = 0.001
batch_size = 32
epochs = 50
factor = 0.2

model.compile(
    loss=keras.losses.LogCosh(),
    metrics=[keras.metrics.LogCoshError()], 
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= factor, patience=3, min_lr=1e-7)

H = model.fit(
    x_train, y_train, batch_size=batch_size,
    validation_split = 0.1,
    steps_per_epoch = x_train.shape[0] // batch_size,
    epochs = epochs,
    callbacks = [early_stopping , reduce_lr]
)


score_train = H.history['logcosh']
score_val = H.history['val_logcosh']

num_epoch = len(score_train)
plt.plot(range(1,num_epoch+1) , score_train , label = 'Train')
plt.plot(range(1,num_epoch+1) , score_val, label = 'Val')
plt.xlabel('Epochs')
plt.ylabel('LogCosh')
plt.legend()


evaluation(model,x_test,y_test)


data_test = pd.read_csv('../working/test.csv')
data_test.info()


all_image_test = [list(map(int,image.split(' '))) for image in data_test['Image'].values]
all_image_test = np.array(all_image_test).reshape(-1,96,96)
all_image_test.shape


y_pred = model.predict(all_image_test)
y_pred.shape


pattern = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')
pattern.head()


pattern.info()


y_pred_df = pd.DataFrame(y_pred)
y_pred_df.columns = data.drop('Image',axis=1).columns


y_pred_df.index = [i for i in range(1,len(y_pred_df)+1)]


for i, j in zip(pattern['ImageId'].values, pattern['FeatureName']):
    pattern.loc[(pattern['ImageId'] == i) & (pattern['FeatureName'] == j), 'Location'] = y_pred_df.loc[i, j]


submit = pattern.drop(['ImageId','FeatureName'],axis=1)


submit.to_csv('submission.csv',index=False)




