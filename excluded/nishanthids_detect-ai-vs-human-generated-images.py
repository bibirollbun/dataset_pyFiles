# import libraries

import random
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
import cv2
import numpy as np


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
    



class DataLoader:
    def __init__(self, HEIGHT=512, WIDTH=512, RESIZE=True, noof_img=3000):
        self.height = HEIGHT
        self.width = WIDTH
        self.size = (self.height, self.width)
        self.resize_flag = RESIZE
        self.num_images = noof_img

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.resize_flag:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return img

    def load_targets(self, label_path):
        file = pd.read_csv(label_path)

        # only get filename
        csv_filenames = file['file_name'].to_list()[0:self.num_images]
        self.csv_filenames = [path.split("/")[-1] for path in csv_filenames]

        self.labels = file['label'].to_list()[0:self.num_images]

    def load_test_id(self, label_path):
        
        file = pd.read_csv(label_path)

        self.id = file['id'].to_list()[0:self.num_images]
        self.csv_filenames = [path.split("/")[-1] for path in self.id]

    def load(self, folder_path, label_path, train=True):
        self.folder_path = folder_path

        self.images, self.labels = [], []
        if train:
            self.load_targets(label_path)
        else:
            self.load_test_id(label_path)

        for path in self.csv_filenames:

            path = self.folder_path + path
            img = self.load_image(path)
            self.images.append(img)
            # plt.imshow(img)
            # plt.show()
        return self.images, self.labels


import os

root = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
if os.path.exists(root):
    print("path present")
else:
    print("path absent")


HEIGHT=224
WIDTH=224
img_path =   '/kaggle/input/ai-vs-human-generated-dataset/train_data/'
label_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_img_path = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'
test_path = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

# Load Train datas
loader = DataLoader(HEIGHT, WIDTH,noof_img=1000)
images, targets = loader.load(img_path, label_path)

# Load test datas
loader_test = DataLoader(HEIGHT,WIDTH, noof_img=100)
test_images, _ =  loader_test.load(test_img_path, test_path, train=False)
id = loader_test.id

x,y = np.array(images), np.array(targets)



# Augmentation pipeline
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    #A.GaussNoise(var_limit=(1, 5), p=0.5),
    A.MotionBlur(blur_limit=3, p=0.5),
    A.MedianBlur(blur_limit=3, p=0.5)
])

# Function to apply augmentation
def augment_image(image):
    image = np.array(image, dtype=np.uint8)
    image = augmentor(image=image)['image']
    return image/255


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)
#print("Samples Shape",np.shape(X_train), np.shape(X_val))

plt.figure(figsize=(12, 9))  # Increase figure size for better visibility

# Loop through 3 rows
for i in range(3):

    # AI-Generated Image (Left)
    plt.subplot(3, 2, 2 * i + 1)  # (Rows=3, Columns=2, Position=1,3,5)
    plt.imshow(x[2 * i], cmap='gray')
    plt.title(f'AI Image {i+1}')
    plt.axis('off')

    # Human-Generated Image (Right)
    plt.subplot(3, 2, 2 * i + 2)  # (Rows=3, Columns=2, Position=2,4,6)
    plt.imshow(x[2 * i + 1], cmap='gray')
    plt.title(f'Human Image {i+1}')
    plt.axis('off')

plt.show()



input_shape = (HEIGHT,WIDTH,np.shape(images)[3])

# Create data generators
train_datagen = ImageDataGenerator(preprocessing_function=lambda x: augment_image(x))
val_datagen = ImageDataGenerator(preprocessing_function=lambda x: augment_image(x))# rescale=1./255) # 

# Fit generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)



from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def resnet_model(input_shape=(224, 224, 3), num_classes=1):

    # Load Pretrained ResNet50 Model
    base_model = ResNet50(include_top=False, input_shape=(HEIGHT, WIDTH, 3),
                         weights='/kaggle/input/resnet50/tensorflow2/default/1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
)

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)  # Change 10 to your number of classes

    # Define the new model
    model = Model(inputs=base_model.input, outputs=x)

    return model

# layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')

model = resnet_model(input_shape, num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=5, batch_size=32)

# Save and load the entire model
model.save("cnn_model2.h5")  # Saves model architecture + weights + optimizer


# plot performance
plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='-', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--', marker='x')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/kaggle/working/renetpt_gauno_ep5')
plt.show()


#model = load_model("cnn_model2.h5")
pred = model.predict(X_val)

pred = np.array(pred).flatten()
y_pred = []

for i, out in enumerate(pred):
    if out > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)




##################################### Submission ####################################################

test_images = np.array(test_images)
N = len(test_images)  # Total number of test images
split_index = N // 2  # Split at the midpoint

# Split the test set into two halves
test_images_1 = test_images[:split_index]  # First half
test_images_2 = test_images[split_index:]  # Second half

# Make predictions in two batches
pred_1 = model.predict(test_images_1)
pred_2 = model.predict(test_images_2)

# Concatenate the results to get the full prediction
full_pred = np.concatenate([pred_1, pred_2], axis=0)
#[print(i) for i in full_pred]
y_pred = [1 if i > 0.1 else 0 for i in full_pred]


y_pred = np.expand_dims(y_pred, axis=1)
id = np.expand_dims(id, axis=1)

output = np.concatenate((id,y_pred), axis=1)
tested = pd.DataFrame(output)
tested.to_csv('/kaggle/working/Submission.csv', header=['id','label'], index=False)

