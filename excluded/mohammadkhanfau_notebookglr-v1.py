import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import cv2
import tensorflow as tf


path = '/kaggle/input/landmark-recognition-2021'
os.listdir(path)

train_images = f'{path}/train'
df_train = pd.read_csv(f'{path}/train.csv')
df_train['path'] = df_train['id'].apply(lambda f: os.path.join('/kaggle/input/landmark-recognition-2021/train', f[0], f[1], f[2], f + '.jpg'))
df_train.head()


test_images = f'{path}/test'
df_test = pd.read_csv(f'{path}/sample_submission.csv')
df_test['path'] = df_test['id'].apply(lambda f: os.path.join('/kaggle/input/landmark-recognition-2021/test', f[0], f[1], f[2], f + '.jpg'))
df_test.head()


# Defining the amount of classes and images in the training dataset.
nr_classes = len(df_train["landmark_id"].unique())
nr_images = len(df_train)

print("Number of classes in training dataset: ", nr_classes)
print("Number of images in training dataset: ", nr_images)


# Histogram of data distribution, to show the amount of images in each class.

hist = plt.figure(figsize = (10, 10))
ax = plt.hist(df_train["landmark_id"], bins = df_train["landmark_id"].unique())
plt.ylim([0, 400])
plt.show()

classes = ax[0]
from0To5 = len(classes[classes <= 5])
from5To10 = len(classes[classes <= 10] - from0To5)
print("Number of classes with 0 to 5 images: ", from0To5)
print("Number of classes with 5 to 10 images: ", from5To10)


# Overall representation of the data distribution

print('Distribution of images in classes:')
value_counts = df_train['landmark_id'].value_counts()
print(value_counts.describe()) 


#Visualization of some samples.

displayImages = []
for i in range(0,4):
    randomClass = df_train[df_train['landmark_id'] == value_counts.iloc[[np.random.randint(0, nr_classes)]].index[0]]
    for j in range(0,4):
        randomImages = randomClass.iloc[np.random.randint(0, len(randomClass))]
        displayImages.append(randomImages)
        
plt.subplots(4, 4, figsize = (15, 10))
for i in range(len(displayImages)):
    plt.subplot(4, 4, i + 1)
    plt.axis('Off')
    img = cv2.imread(displayImages[i][2])
    plt.imshow(img)
    plt.title(f'landmark id: {displayImages[i][1]} ', fontsize=8)


# Setting up hyperparameters and splitting training data into train and val
def imagePath(imgPath):
    images = []
    for imgFile in imgPath:
        imgPic = cv2.imread(imgFile, 1)
        images.append(cv2.resize(imgPic, (img_size, img_size)))
    
    return images

# Hyperparameters
epochs = 20
batch_size = 32
img_size = 128
train_split = 0.7
val_split = 0.2
nrClasses = 100

# Setting up dataset for training
imgList = []
labels = []
temp_labels = []

# i = 0
# for lbl in df_train['landmark_id'].unique():
#     if i == nrClasses:
#         break
#     if(len(df_train['path'][df_train['landmark_id'] == lbl].value_counts()) > 200 and
#        len(df_train['path'][df_train['landmark_id'] == lbl].value_counts()) < 400): 
#         for path in df_train['path'][df_train['landmark_id'] == lbl]: 
#             imgList.append(path) 
#             labels.append(lbl)
#             temp_labels.append(i)
#         i = i + 1


i = 0
counts_df_train = df_train['landmark_id'].value_counts()
sorted_label_counts = list(counts_df_train.index)
for lbl in sorted_label_counts[:nrClasses]:
    for path in df_train['path'][df_train['landmark_id'] == lbl]: 
        imgList.append(path) 
        labels.append(lbl)
        temp_labels.append(i)
    i = i + 1


# Random shuffle dataset, so it is no longer set up in classes
shuff = list(zip(imgList, temp_labels))
random.shuffle(shuff)

imgList, lbls = zip(*shuff)

# Preparing data to be split into train and val
imgNr = round(len(imgList) * train_split)

trainImages = imgList[:imgNr]
trainData = imagePath(trainImages)
trainLabels = lbls[:imgNr]

print("Number of total images for training: ", len(trainData))


# Setting images and labels to be split into train and val data
xData = np.array(trainData)
yData = tf.keras.utils.to_categorical(trainLabels, num_classes = nrClasses)

x_train, x_val, y_train, y_val = train_test_split(xData, yData, test_size = val_split, random_state = 101)

print("Number of train images: ", len(x_train))
print("Number of val images: ", len(x_val))

# Random Crop for Data Augmentation
def random_crop(image):  
    height, width = image.shape[:2]
    random_array = np.random.random(size=4);
    w = int((width*0.5) * (1+random_array[0]*0.5))
    h = int((height*0.5) * (1+random_array[1]*0.5))
    x = int(random_array[2] * (width-w))
    y = int(random_array[3] * (height-h))

    image_crop = image[y:h+y, x:w+x, 0:3]
    image_crop = cv2.resize(image_crop, image.shape)
    return image_crop

# Setting up data generator for data augmentation
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = False, # False for no flip, True for flip
                                                                vertical_flip = False, # False for no flip, True for flip
                                                                rotation_range = 0, # 0 for no rotation, 90 for rotation
                                                                zoom_range = 0, # 0 for normal, [0.5, 1.5] for augmentation
                                                                width_shift_range = 0, # 0 for normal, 0.3 for augmentation
                                                                height_shift_range = 0, # 0 for normal, 0.3 for augmentation
                                                                shear_range = 0, # 0 for normal, 45 for augmentation
                                                                #brightness_range=(0.1, 0.9), # comment for no brightness change
                                                                #preprocessing_function=random_crop, 
                                                                fill_mode = "nearest") # wrap, reflect and constant are some other for experimentation
                                                 

opt = tf.optimizers.SGD(learning_rate = 0.001) # 0.01
opt2 = tf.optimizers.Adam(learning_rate = 0.001) #0.01
opt3 = tf.optimizers.Adagrad(learning_rate = 0.001) #0.01


# CNN model 
# Resnet101 also be used and some other models as well
ResNet152 = tf.keras.applications.resnet.ResNet152(input_shape = (img_size, img_size, 3),
                                                      include_top = False, #True
                                                      weights = 'imagenet', 
                                                      pooling = 'avg') # max

for layer in ResNet152.layers:
    layer.trainable = False


inputs = ResNet152.input
flatten = tf.keras.layers.Flatten()(ResNet152.output)
dropout1 = tf.keras.layers.Dropout(0.2)(flatten)
dense1 = tf.keras.layers.Dense(units = 4096, activation = "relu")(dropout1)
dropout2 = tf.keras.layers.Dropout(0.2)(dense1)
dense2 = tf.keras.layers.Dense(units = 4096, activation = "relu")(dropout2)
dropout3 = tf.keras.layers.Dropout(0.2)(dense2)
output = tf.keras.layers.Dense(units = nrClasses, activation = "softmax")(dropout3) # activation = relu, linear, softmax
model = tf.keras.Model(inputs = inputs, outputs = output)

print(model.summary())


# Compilation
#model = ResNet50(input_shape = (128, 128, 3), classes = nrClasses) # should be used when using batch norm editable resnet
model.compile(optimizer = opt2 , loss = "categorical_crossentropy", metrics = ['accuracy'])
#print(model.summary()) # should be used when using batch norm editable resnet
epochs = 20
history = model.fit(dataGenerator.flow(x_train, y_train, batch_size = batch_size), validation_data = (x_val, y_val), epochs = epochs)



model = tf.keras.models.load_model('/kaggle/working/GLR_model_ResNet152_ep20.h5')


# Plotting the performance of the model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet152 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

# Plot of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNet152 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()


# Predictions of the model -> running model on test data
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics

testImages = imgList[len(trainImages):]
testImages = imagePath(testImages)
testLabels = lbls [len(trainLabels):]

testData = np.array(testImages)
testPrediction = model.predict(dataGenerator.flow(testData, batch_size = batch_size))

goodAcc = []
badAcc = []
confidence = []
for i in testPrediction:
    confidence.append(max(i))
    goodAcc.append(np.argmax(i))
    badAcc.append(np.argmax(i))

precision, recall, fscore, support = score(testLabels, goodAcc, labels = np.unique(goodAcc))

print("Confusion Matrix")
print(metrics.confusion_matrix(testLabels, goodAcc))
print("Classification Report")
print(metrics.classification_report(testLabels, goodAcc, digits = 3))


# Visulaise the best output of the model on the test data
plm_count = 0
for i in range(len(goodAcc)):
    plt.axis('Off')
    if (testLabels[i] == goodAcc[i]):
        print("Perfect Label Match")
        title = ('True label: ' + str(testLabels[i]) + '_' + 'Predicted label: ' + str(goodAcc[i]) + '_' + 'confidence: ' + str(confidence[i]))
        plt.title(title, fontsize = 10)
        plt.imshow(testImages[i])
        plt.show()

        plm_count += 1
        
    elif(confidence[i] < 0.1):
        print("Poor Confidence")
        title = ('True label: ' + str(testLabels[i]) + '_' + 'Predicted label: ' + str(goodAcc[i]) + '_' + 'confidence: ' + str(confidence[i]))
        plt.title(title, fontsize = 10)
        plt.imshow(testImages[i])
        plt.show()
    if plm_count == 100:
        break


model.save("GLR_model_ResNet152_ep20.h5")







