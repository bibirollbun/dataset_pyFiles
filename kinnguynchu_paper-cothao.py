!pip install np_utils



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K



# Đường dẫn
train_data = '/kaggle/input/fruits262/Fruit-262'
val_test_split = 0.2  # 20% cho test, 20% còn lại cho validation

# Tạo generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=val_test_split  # Chia tự động
)

# Generator cho training
train_generator = datagen.flow_from_directory(
    train_data,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Phần dành cho training
)

# Generator cho validation
val_generator = datagen.flow_from_directory(
    train_data,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Phần dành cho validation
)



# #converting images into array to start computation

# def convert_image_to_array(files):
#     images_as_array=[]
#     for file in files:
#         # Thêm target_size để đảm bảo tất cả ảnh cùng kích thước
#         img = load_img(file, target_size=(128, 128))  # Thêm dòng này
#         images_as_array.append(img_to_array(img))
#     return images_as_array

# X_train = np.array(convert_image_to_array(X_train))
# X_val = np.array(convert_image_to_array(X_val))
# X_test = np.array(convert_image_to_array(X_test))


# #nomalizing the pixel values before feeding into a neural network

# X_train = X_train.astype('float32')/255
# X_val = X_val.astype('float32')/255
# # X_test = X_test.astype('float32')/255


#Building model 1 using customized convolutional and pooling layers

model = Sequential()

#input_shape is 100*100 since thats the dimension of each of the fruit images
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(128,128,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))
# specifying parameters for fully connected layer
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes,activation = 'softmax'))
model.summary()


#importing ootimizers

from keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision
optimizer = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', Precision()])


# 3. Tạo model checkpoint
checkpointer = ModelCheckpoint(
    filepath='cnn_from_scratch_fruits.keras',
    verbose=1,
    save_best_only=True
)

# 4. Huấn luyện model với generator
CNN_model = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size, # Số batch/epoch
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpointer],
    verbose=2,
    shuffle=True
)


#checking testset accuracy

score = model.evaluate(X_test, Y_test)
print('Test accuracy:', score[1])


# using model to predict on test data
Y_pred = model.predict(X_test)

# Lets plot the predictions of different fruits and check their original labels

fig = plt.figure(figsize=(20, 15))
for i, idx in enumerate(np.random.choice(X_test.shape[0], size=25, replace=False)):
    ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_test[idx]))
    pred_idx = np.argmax(Y_pred[idx])
    true_idx = np.argmax(Y_test[idx])
    ax.set_title("{} ({})".format(labels[pred_idx], labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))


#plotting the loss function and accuracy for different epochs

plt.figure(1, figsize = (10, 10))  
plt.subplot(211)  
plt.plot(CNN_model.history['acc'])  
plt.plot(CNN_model.history['val_acc'])  
plt.title('Model Accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')  
plt.legend(['train', 'validation'], loc='upper left')   

# plotting model loss 
plt.subplot(212)  
plt.plot(CNN_model.history['loss'])  
plt.plot(CNN_model.history['val_loss'])  
plt.title('Model Loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['train', 'validation'], loc='upper left')  
plt.show()







