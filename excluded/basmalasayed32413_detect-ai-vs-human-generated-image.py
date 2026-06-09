import os
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import collections
from tensorflow.keras.layers import *
import cv2
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

train_df.head()


train_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(train_data_dir, x))
train_df['label'] = train_df['label'].astype(str)



print(train_df['label'].value_counts())



# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.8,1.2],
#     fill_mode='nearest',
#     validation_split=0.2
    
# )


# train_generator = train_datagen.flow_from_dataframe(
#     train_df,
#     x_col='file_name',
#     y_col='label',
#     class_mode='binary',
#     target_size=(224, 224),
#     batch_size=32,
#     subset='training',
#     shuffle=True
# )


# val_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation',
#     shuffle=False  )



import matplotlib.image as mpimg
images = train_df['file_name'].values
labels = train_df['label'].values
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = mpimg.imread(images[i])
    plt.imshow(img)
    plt.title(f"Label: {int(labels[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()


import collections
import matplotlib.pyplot as plt

# معالجة الليبلات
labels_list = [int(float(l)) if str(l).replace('.', '', 1).isdigit() else l for l in labels]

# عدّ الليبلات
label_counts = collections.Counter(labels_list)

# طباعة للتأكد
print("Label counts:", label_counts)

# رسم بياني
plt.bar(label_counts.keys(), label_counts.values())
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
plt.xlabel('Labels')
plt.show()




from tensorflow.keras.preprocessing import image

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))  # لازم الحجم يناسب VGG
    img_array = image.img_to_array(img)
    img_array /= 255.0  # تطبيع الصور
    return img_array


from keras.layers import GlobalAveragePooling2D
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

VGG_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


VGG_model.compile(optimizer = Adam(learning_rate=1e-5)
, loss='binary_crossentropy'
, metrics=['accuracy'])

X = np.array([load_image(path) for path in train_df['file_name']])
y = train_df['label'].values

VGG_model.fit(X, y, epochs=4, validation_split=0.2, callbacks=[early_stop])



VGG_model.summary()


plt.plot(history_vgg.history['accuracy'], label='Train Acc')
plt.plot(history_vgg.history['val_accuracy'], label='Val Acc')
plt.title('VGG Accuracy')
plt.legend()
plt.show()

plt.plot(history_vgg.history['loss'], label='Train Loss')
plt.plot(history_vgg.history['val_loss'], label='Val Loss')
plt.title('VGG Loss')
plt.legend()
plt.show()


from tensorflow.keras.models import load_model
VGG_model.save_weights('my_weights.weights.h5')


val_generator.reset()
val_predictions = VGG_model.predict(val_generator)
val_predictions = (val_predictions > 0.5).astype(int).ravel()
val_labels = val_generator.classes


cm = confusion_matrix(val_labels, val_predictions)
classes_names = ['Human', 'AI']
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_names,
            yticklabels=classes_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



print("Classification Report:\n")
print(classification_report(val_labels, val_predictions,target_names=val_generator.class_indices.keys()))



from tensorflow.keras.preprocessing import image
import numpy as np
import os

test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join("/kaggle/input/ai-vs-human-generated-dataset", x))

def preprocess_image(path):
    img = image.load_img(path, target_size=(128, 128))  
    img_array = image.img_to_array(img) / 255.0
    return img_array

X_test = np.array([preprocess_image(p) for p in test_df['file_path']])




predictions = VGG_model.predict(X_test)
test_df['label'] = (predictions > 0.5).astype(int)


submission = test_df[['id', 'label']]
submission.to_csv('submission.csv', index=False)

