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


# VGG=Visual Geometry Group
# VGG16 & VGG19
# The input to this can network is a fixed 224x224 RGB image and the only preprocessing it does is substracting the mean RGB values, which are computed on the training dataset. from each pixel
# Then the image is running through stack of convolution layers, where there are filters with a very small recptive field that is 3x3 
# We have 13 convolution layer, 5 max_pooling layers, 3 fully connected layers
# Total parameters are 138 million

import numpy as np
import pandas as pd 
import os
from PIL import Image
from numpy import array
from numpy import asarray
import seaborn as sns
training_img_list=list()
path_to_train='/kaggle/input/plant-seedlings-classification/train'
shape_sum=0
class_name=dict()
train_avg_shape=80

for dirname, _, filenames in os.walk(path_to_train):
    for filename in filenames:
        img_data=Image.open(os.path.join(dirname, filename))
        resizedImage= img_data.resize((train_avg_shape, train_avg_shape))
        resizedImage= resizedImage.convert('RGB')
        resizedImage= asarray(resizedImage)/255

        class_label=dirname.split('/')[-1]
        training_img_list.append([resizedImage, class_label])
        shape_sum += np.max(img_data.size)
        class_name[class_label]=len(class_name)-1


from keras import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D



from sklearn.model_selection import KFold
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(17)
kf = KFold(n_splits=5)
epochs = 20
batch_size = 32


# Load VGG19 Model
from keras.applications.vgg19 import VGG19
vgg19_model = VGG19(weights='imagenet',include_top=False)
from keras.utils import plot_model
plot_model(vgg19_model, to_file='VGG19.png', show_shapes=True, show_layer_names=True)


vgg19_model.summary()


x=vgg19_model.output


from keras.layers import Dense,GlobalAveragePooling2D

x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(len(class_name), activation='softmax')(x) #final layer with softmax activation


from keras.models import Model
newModel=Model(inputs=vgg19_model.input,outputs=preds)
print("changed model layer count %d" %len(newModel.layers))
newModel.summary()


for layer in newModel.layers[:-5]:
    layer.trainable=False

newModel.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(path_to_train, 
                                                  target_size=(80,80),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)


newModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
history = newModel.fit(train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=2)


print("History for cross validation fold 1")
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model loss and accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy','loss'], loc='upper right')
plt.show()


pathToTestData ='/kaggle/input/plant-seedlings-classification/test'
test_img_list = list()

for dirname, _, filenames in os.walk(pathToTestData):
    for filename in filenames:
        img_data = Image.open(os.path.join(dirname, filename))
        
        resizedImage = img_data.resize((train_avg_shape, train_avg_shape))
        resizedImage = resizedImage.convert('RGB')
        resizedImage = asarray(resizedImage)/255

        test_img_list.append([resizedImage,filename])


X_test = np.zeros((len(test_img_list), train_avg_shape, train_avg_shape, 3), dtype='float32')

for i,img in enumerate(test_img_list):
    X_test[i] = test_img_list[i][0]


predictions = newModel.predict(X_test, batch_size=None, verbose=0)
predictions=pd.DataFrame(predictions)


inverse_label_map = dict()
for k,v in train_generator.class_indices.items():
    inverse_label_map[v] = k


pred_label_num = predictions.idxmax(axis=1)
pred_label_num_new = list()

for x in pred_label_num:
    y = inverse_label_map[x]
    pred_label_num_new.append(y)

pred_label_num_new = pd.DataFrame(pred_label_num_new)
print(pred_label_num_new[0])



pred=pd.DataFrame()


testImages = pd.DataFrame(test_img_list) 
pred.insert(0,'file',testImages[1])
pred.insert(1,'species',pred_label_num_new[0])
pred.head()


pred.to_csv('predictionsVgg19.csv',index = None, header=True)







