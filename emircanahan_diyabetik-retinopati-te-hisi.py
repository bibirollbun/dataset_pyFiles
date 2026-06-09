import pandas as pd
import numpy as np


path = '/kaggle/input/aptos2019-blindness-detection/'


df = pd.read_csv(path + 'train.csv', sep = ',')


df.head()


df['diagnosis'].hist()
df['diagnosis'].value_counts()


import os
files = os.listdir(path + 'train_images')
# files


len(files)


import cv2


img_list = []

for i in files[0:20]:
    image = cv2.imread(path + 'train_images/'+i)
    image = cv2.resize(image,(400,400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_list.append(image)


import matplotlib.pyplot as plt
plt.imshow(img_list[2])


plt.imshow(img_list[4])


kopya = img_list[4].copy()


kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)


plt.imshow(kopya, cmap='gray')


kopya.shape


blur = cv2.GaussianBlur(kopya,(5,5),0)


plt.imshow(blur,cmap='gray')


thresh = cv2.threshold(blur,10,255, cv2.THRESH_BINARY)[1]


plt.imshow(thresh, cmap='gray')


kontur = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


 kontur = kontur[0][0]


kontur.shape


kontur = kontur[:,0,:]


kontur.shape


kontur[:,0].argmax()


kontur[335]


kontur[:,0].argmin()


kontur[111]


sol = tuple(kontur[kontur[:,0].argmin()])
sağ = tuple(kontur[kontur[:,0].argmax()])
üst = tuple(kontur[kontur[:,1].argmin()])
alt = tuple(kontur[kontur[:,1].argmax()])


sol, sağ, üst, alt


x1 = sol[0]
y1 = üst[1]
x2 = sağ[0]
y2 = alt[1]


x1, y1, x2, y2


orijinal = img_list[4].copy()


plt.imshow(orijinal)


crop_ilk = orijinal[y1:y2 , x1:x2]


plt.imshow(crop_ilk)


crop_ilk.shape


crop_ilk = cv2.resize(crop_ilk,(400,400))


plt.imshow(crop_ilk)


x = int(x2-x1)*4//100
y = int(y2-y1)*5//100


x,y


crop_son = orijinal[y1+y : y2-y , x1+x : x2-x]


plt.imshow(crop_son)


crop_son = cv2.resize(crop_son,(400,400))


plt.imshow(crop_son)


lab = cv2.cvtColor(crop_son, cv2.COLOR_RGB2LAB)


lab.shape


l,a,b = cv2.split(lab)


plt.imshow(l, cmap='gray')


l.shape


düz = l.flatten()


düz.shape


plt.hist(düz,25,[0,256], color = 'r')
plt.show()


clahe = cv2.createCLAHE(clipLimit=7.0,tileGridSize=((8,8)))
cl = clahe.apply(l)


plt.hist(cl.flatten(),25,[0,256], color = 'r')
plt.show()


plt.imshow(cl)


plt.imshow(l)


limg = cv2.merge((cl,a,b))


son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


plt.imshow(son)


plt.imshow(crop_son)


med_son = cv2.medianBlur(son, 3)


plt.imshow(med_son)


arka_plan = cv2.medianBlur(son, 37)


plt.imshow(arka_plan)


maske = cv2.addWeighted(med_son,1,arka_plan,-1,255)
plt.imshow(maske)


son_img = cv2.bitwise_and(maske,med_son)


plt.imshow(son_img)


plt.imshow(med_son)


img_list = []

from tqdm import tqdm_notebook as tqdm

for i in tqdm(files):
    image = cv2.imread(path + 'train_images/'+i)
    image = cv2.resize(image,(400,400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kopya = image.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(kopya,(5,5),0)
    thresh = cv2.threshold(blur,10,255, cv2.THRESH_BINARY)[1]
    kontur = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontur = kontur[0][0]
    kontur = kontur[:,0,:]
    x1 = tuple(kontur[kontur[:,0].argmin()])[0]
    y1 = tuple(kontur[kontur[:,1].argmin()])[1]
    x2 = tuple(kontur[kontur[:,0].argmax()])[0]
    y2 = tuple(kontur[kontur[:,1].argmax()])[1]
    x = int(x2-x1)*4//50
    y = int(y2-y1)*5//50
    kopya2 = image.copy()
    if x2-x1 >100 and y2-y1> 100:
        kopya2 = kopya2[y1+y : y2-y , x1+x : x2-x]
        kopya2 = cv2.resize(kopya2,(400,400))
    lab = cv2.cvtColor(kopya2, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0,tileGridSize=((8,8)))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    med_son = cv2.medianBlur(son, 3)
    arka_plan = cv2.medianBlur(son, 37)
    maske = cv2.addWeighted(med_son,1,arka_plan,-1,255)
    son_img = cv2.bitwise_and(maske,med_son)
    img_list.append(son_img)


plt.imshow(img_list[2])


fig = plt.figure(figsize=(20,12))

for i in range(12):
    img = img_list[i]
    fig.add_subplot(3,4,i+1)
    plt.imshow(img)

plt.tight_layout()


# df['diagnosis']


y_train = pd.get_dummies(df['diagnosis']).values


# y_train


df['diagnosis'][1]


y_train[1]


y_train_son = np.ones(y_train.shape, dtype='uint8')


# y_train_son


y_train_son[:,4] = y_train[:,4]


# y_train_son


import numpy as np


np.logical_or(0,0)


np.logical_or(1,0)


np.logical_or(0,1)


np.logical_or(1,1)


np.logical_and(0,1)


np.logical_and(1,1)


for i in range(3,-1,-1):
    y_train_son[:,i] = np.logical_or(y_train[:,i], y_train_son[:,i+1])


# y_train_son


x_train = np.array(img_list)


x_train.shape


y_train_son.shape


from sklearn.model_selection import train_test_split

x_train, x_val , y_train, y_val = train_test_split(x_train,
                                                   y_train_son,
                                                   test_size=0.15,
                                                   random_state=2019,
                                                   shuffle=True)


x_train.shape, x_val.shape , y_train.shape, y_val.shape


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
data_generator = datagen.flow(x_train,y_train,batch_size=2,seed=2020)




!pip install efficientnet


import tensorflow
from tensorflow.keras.applications import EfficientNetB5



örnek_model2 = EfficientNetB5(include_top=False)
# örnek_model2.summary()


img_shape = (400,400,3)
EfficientNetB5 = EfficientNetB5(weights = 'imagenet',
            include_top = False,
           input_shape = img_shape)


for layer in EfficientNetB5.layers[:-3]:   #  EfficientNetB5 modeline ait olan son 3 nörondan öncesini öğrenemez
    layer.trainable = False


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout,Dense


model = Sequential()
model.add(EfficientNetB5)
model.add(GlobalAveragePooling2D())   # diğer modellerde 'flatten' bu modelimizde GlobalAveragePooling2D
model.add(Dropout(0.5))
model.add(Dense(5,activation = 'sigmoid'))



from tensorflow.keras.optimizers import Adam

model.compile(loss='binary_crossentropy',
             optimizer=Adam(learning_rate=5e-5),
             metrics=['accuracy'])


from keras.callbacks import ReduceLROnPlateau

lr = ReduceLROnPlateau(monitor = 'val_loss',
                      patience = 3,
                      verbose = 1,
                      mode='auto',
                      factor=0.25,
                      min_lr=0.000001)


history = model.fit(data_generator,
                    steps_per_epoch = 1000, #aslında 3100 / 2 (train / batch) 
                    epochs = 3,
                    validation_data = (x_val,y_val),
                    callbacks = [lr])

