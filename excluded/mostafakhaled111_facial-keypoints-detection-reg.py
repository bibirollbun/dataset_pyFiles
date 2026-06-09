# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/facial-keypoints-detection/training.zip")
test =pd.read_csv("/kaggle/input/facial-keypoints-detection/test.zip")
lookid_data=pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")



train.info()


train.head().T


test.head()


train.isnull().sum()


train.fillna(method = 'ffill' ,inplace = True)


test.isnull().sum()


imag = []
for i in range(0, 7049):
    img_str_list = train['Image'][i].split(' ')
    
   
    processed_img_list = [] 
    for x in img_str_list: 
        if x == '':
            processed_img_list.append('0')
        else:
            processed_img_list.append(x)
    
    imag.append(processed_img_list) 


image_to_list = np.array(imag,dtype = 'float')
X_train = image_to_list.reshape(-1,96,96,1)

plt.imshow(X_train[10].reshape(96,96),cmap='gray')
plt.show()


training= train.drop("Image", axis= 1)
y_train = []
for i in range(0,7049):
    y = training.iloc[i,:].values

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')


y_train


from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,BatchNormalization ,MaxPool2D
from keras.layers import LeakyReLU

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                  input_shape = (96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same' 
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same' 
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same' 
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same' 
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same' 
                 ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(30))
model.summary()



model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])


model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split = 0.15)



timag = []
for i in range(0, 1783):
    img_str_list = test['Image'][i].split(' ')
    
   
    processed_img_list = [] 
    for x in img_str_list: 
        if x == '':
            processed_img_list.append('0')
        else:
            processed_img_list.append(x)
    
    timag.append(processed_img_list) 


timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96,1) 


plt.imshow(X_test[10].reshape(96,96),cmap = 'gray')
plt.show()


predict = model.predict(X_test)


lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(predict)


rowid = lookid_data['RowId']
rowid=list(rowid)


feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))


preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])


rowid = pd.Series(rowid,name = 'RowId')


loc = pd.Series(preded,name = 'Location')



submission = pd.concat([rowid,loc],axis = 1)



submission.to_csv('facial_detection_submission.csv',index = False)




