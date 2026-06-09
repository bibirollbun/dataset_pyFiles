%%capture
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


training_data = pd.read_csv('/kaggle/input/dlp-image-classification-for-handwriting/downlaod1/train.csv')


training_data.head()


%%capture
!pip install protobuf==3.20.3


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import collections
import tensorflow as tf
import random
import cv2
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage.feature import hog


seed=50
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


def preprocess(img,size):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    _,thresh = cv2.threshold(
        gray,0,255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((size,size),dtype=np.float32)
    x,y,w,h = cv2.boundingRect(coords)
    digit = thresh[y:y+h,x:x+w]

    h,w = digit.shape
    pad = abs(h-w)//2

    if h>w:
        digit = cv2.copyMakeBorder(
            digit,0,0,pad,pad,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        digit = cv2.copyMakeBorder(
            digit,pad,pad,0,0,
            cv2.BORDER_CONSTANT, value=0
        )

    digit = cv2.resize(digit,(size,size))
    digit = digit.astype("float32")/255.0
    return digit


img_size=32

x=[]
y=[]
img_dir = "/kaggle/input/dlp-image-classification-for-handwriting/downlaod1/train"

for _,row in training_data.iterrows():
    img_path = os.path.join(img_dir,row['image_id'])
    img = load_img(img_path, target_size=(img_size,img_size))
    img = img_to_array(img).astype("uint8")
    img = preprocess(img,img_size)
    '''
    img = img/255.0'''
    x.append(img)
    y.append(row['label'])
    
x = np.array(x)
y = np.array(y)
le = LabelEncoder()
y = le.fit_transform(y)



print(x.shape)
print(y.shape)


print(np.min(x), np.max(x))


x_train,x_val,y_train,y_val = train_test_split(x,y, test_size = 0.25, random_state=42, stratify=y)


print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)


#x_train = x_train[..., np.newaxis]
#x_val   = x_val[..., np.newaxis]


print(collections.Counter(y_train))
print(collections.Counter(y_val))


i=500
plt.imshow(x_train[i])
plt.title(f"Label : {y_train[i]}")
plt.show()


img_size=32

x_test = []

test_img_dir = "/kaggle/input/dlp-image-classification-for-handwriting/downlaod1/test"
test_images = sorted(os.listdir(test_img_dir))

for image in test_images:
    img_path = os.path.join(test_img_dir,image)
    img = load_img(img_path, target_size=(img_size,img_size))
    img = img_to_array(img).astype("uint8")
    img = preprocess(img,img_size)
    #img = img/255.0
    x_test.append(img)
    
x_test = np.array(x_test)



#x_test  = x_test[..., np.newaxis]


param_grid={
    "lr":[1e-4,3e-4,5e-4],
    "batch_size":[8,16],
    "filters":[32,48]
}


epoch= 50


NUM_CLASSES = len(le.classes_)


def build_model(lr,filt):
    model = tf.keras.Sequential(
    [   
        tf.keras.layers.Input(
            shape=(img_size,img_size,1)
        ),
        
        tf.keras.layers.Conv2D(filt,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(filt*2,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(filt*3,3,activation='relu',padding='same'),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(64,activation='relu'),
                              #,kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')
        
    ]
    ) 
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )
    return model


result=[]
for lr in param_grid["lr"]:
    for bs in param_grid["batch_size"]:
        for filters in param_grid["filters"]:
            print("LR=",lr,",bs=",bs,",filter=",filters)
            model=build_model(lr,filters)
            hist = model.fit(x_train,y_train,
                             validation_data = (x_val,y_val),
                             epochs=epoch,
                             batch_size=bs,
                             shuffle=True,
                             verbose=0)
            best_val=max(hist.history["val_accuracy"])
            result.append((lr,bs,filters,best_val))
            print("Best acc:",best_val)


sorted(result,key=lambda x: x[3],reverse=True)[0]


x = np.concatenate([x_train,x_val],axis=0)
y = np.concatenate([y_train,y_val],axis=0)


print(x.shape,y.shape)


#NUM_CLASSES = len(le.classes_)

model = tf.keras.Sequential(
    [   
        tf.keras.layers.Input(
            shape=(img_size,img_size,1)
        ),
        
        tf.keras.layers.Conv2D(48,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(96,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(144,3,activation='relu',padding='same'),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(64,activation='relu'),
                              #,kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')
        
    ]
)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=sorted(result,key=lambda x: x[3],reverse=True)[0][0]),#0.0003
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )


model.fit(x,y,
          epochs=epoch,
          batch_size=sorted(result,key=lambda x: x[3],reverse=True)[0][1], #8
         shuffle=True,
         verbose=0)


print(model.evaluate(x_val,y_val))


def extract_hog(images):
    features=[]
    for i in range(images.shape[0]):
        img = images[i]
        feat = hog(img,
                  pixels_per_cell=(8,8),
                  cells_per_block=(2,2),
                  orientations=9,
                  block_norm='L2-Hys')
        features.append(feat)
    return np.array(features)


x_hog_train = extract_hog(x_train)
x_hog_val = extract_hog(x_val)
x_hog = extract_hog(x)


x_hog_train.shape


svm = SVC(kernel = 'rbf',
         C=10,
         gamma='scale',
         probability=True)



svm.fit(x_hog_train,y_train)


svm_proba = svm.predict_proba(x_hog_val)


cnn_proba = model.predict(x_val)


cnn_weight = 0.5
svm_weight = 0.5


ensemble_proba = cnn_weight*cnn_proba + svm_weight*svm_proba


ensemble_preds = ensemble_proba.argmax(axis=1)


for i in range(10):
    print(y_val[i], ensemble_preds[i])



print(accuracy_score(y_val,ensemble_preds))


x_test.shape


svm.fit(x_hog,y)


#pred_labels = le.inverse_transform(cnn_proba_final.argmax(axis=1))
#print(pred_labels)


#np.unique(pred_labels, return_counts=True)


x_hog_test = extract_hog(x_test)

cnn_test_proba = model.predict(x_test)
svm_test_proba = svm.predict_proba(x_hog_test)

final_test_proba = cnn_weight * cnn_test_proba + cnn_weight * svm_test_proba
final_test_labels = final_test_proba.argmax(axis=1)



submission = pd.DataFrame({
    'image_id' : test_images,
    'label' : final_test_labels
})


submission.head(20)


submission.to_csv('submission.csv',index=False)

