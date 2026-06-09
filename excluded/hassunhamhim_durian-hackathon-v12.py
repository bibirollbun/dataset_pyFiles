# 1. นำเข้าไลบรารีที่จำเป็น
import os
from PIL import Image

# 2. กำหนดแหล่งที่อยู่ของไฟล์ภาพ (Input) และโฟลเดอร์สำหรับส่งออก (Output)
input_folder = "/kaggle/input/durian-hackathon-durian-disease-classification/Images"
# กำหนด output_folder ให้เป็นโฟลเดอร์ใน Kaggle working directory (คุณสามารถเปลี่ยนแปลงได้ตามต้องการ)
output_folder = "/kaggle/working/output_images"
os.makedirs(output_folder, exist_ok=True)  # สร้างโฟลเดอร์ส่งออกถ้ายังไม่มี

# 3. กำหนดขนาดที่ต้องการให้กับภาพทั้งหมด (เช่น 256x256 พิกเซล)
target_size = (512, 512)

# 4. นำไฟล์ภาพทุกไฟล์ในโฟลเดอร์ Input ที่มีนามสกุลที่ต้องการและปรับขนาดให้เท่ากัน
for filename in os.listdir(input_folder):
    # ตรวจสอบไฟล์ที่มีนามสกุลเป็นภาพ
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)
        try:
            with Image.open(img_path) as img:
                # ปรับขนาดภาพให้เท่ากับ target_size
                img_resized = img.resize(target_size)
                
                # 5. ส่งออกภาพที่ได้ไปยังโฟลเดอร์ Output
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


#;;;;;;;;;;;;;;;;;;;;;;;;


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.io import imread

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


df = pd.read_csv('/kaggle/input/durian-hackathon-durian-disease-classification/train.csv')

# Get the counts for each class
cases_count = df['label'].value_counts()
print(cases_count)

# Plot the results 
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of images', fontsize=14)
plt.xlabel('class', fontsize=12)
plt.ylabel('Count', fontsize=12)
# plt.xticks(range(len(cases_count.index)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
plt.show()
del cases_count


df.head()


from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(df, test_size=0.50, stratify=df['label'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


train_df['label'] = train_df['label'].astype(str)
validate_df['label'] = validate_df['label'].astype(str)


train_df.head()


train_df['label'].value_counts().sort_index().plot.bar()


validate_df['label'].value_counts().sort_index().plot.bar()


image_size = 512
target_size = (512, 512)
input_shape = (512, 512, 3)

batch_size = 16
num_class = 4
epochs = 30


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/working/output_images", 
    x_col='id',
    y_col='label',
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size
)


batch=next(train_generator)  # returns the next batch of images and labels 
print(batch[0].shape) # batch[0] is the images, batch[1] are the labels

for i in range(0,4):
    img=batch[0][i] 
    plt.imshow(img)
    plt.show()


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/kaggle/working/output_images", 
    x_col='id',
    y_col='label',
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size
)


import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

import os

base_model = tf.keras.applications.VGG16(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(num_class, activation="softmax")(x)

model = keras.Model(inputs = base_model.input,
                    outputs = output)


model.summary()


from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Define the optimizer
adam = Adam(learning_rate=0.001)

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
mcp_save = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)


# Train the model
history = model.fit(train_generator, 
                validation_data = validation_generator,
                epochs = 30,
                batch_size = batch_size,
                callbacks=[earlyStopping, mcp_save])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


#test data
test_df = pd.read_csv('/kaggle/input/durian-hackathon-durian-disease-classification/submit.csv')


test_df.head()


test_df.tail()


# create test generator

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/kaggle/working/output_images", 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False
)


# Load best model from saved file
model = load_model('/kaggle/working/best_model.keras')


# Predict
predict = model.predict(test_generator)


test_df['predict'] = np.argmax(predict, axis=-1)


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['predict'] = test_df['predict'].replace(label_map)


test_df['predict'].value_counts().plot.bar()


print(test_df.head()) 


test_df.to_csv('../working/submit.csv', index=False)
from IPython.display import FileLink
FileLink(r'submit.csv')

