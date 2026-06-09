import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

from tensorflow.keras import backend as K
import tensorflow as tf

# تفعيل الـ deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'


df_train = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
df_train.head()


train_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"
df_train['filepath'] = df_train['filename'].apply(lambda x: os.path.join(train_dir, x))
df_train


train_data = df_train[['filepath', 'label']]
train_data


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_data,
    test_size=0.2,
    stratify=train_data['label'],
    random_state=42
)

print(f"Train set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")

print("\nSample from train set:")
print(train_df.head())

print("\nSample from validation set:")
print(val_df.head())


import matplotlib.pyplot as plt

class_counts = train_df['label'].value_counts()

print("Class distribution in train data:\n")
print(class_counts)

plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Images per Class in Train Data')
plt.xlabel('Sheep Breed')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(train_df['label'])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])
class_weights = dict(zip(classes, weights))


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=8,
    class_mode='sparse',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=8,
    class_mode='sparse',
    shuffle=False
)



label_map = train_generator.class_indices
inv_label_map = {v: k for k, v in label_map.items()}


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


from tensorflow.keras.applications import EfficientNetV2M ,MobileNetV2 , Xception , DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , GlobalAveragePooling2D, Dropout ,BatchNormalization ,Flatten
from tensorflow.keras.optimizers import Adam


base_model = DenseNet121(include_top=False,
                             weights='imagenet',
                             input_shape=(224, 224, 3),
                             pooling = 'avg'
                          )

base_model.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# Dense layer 1
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Dense layer 2
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=30, validation_data=val_generator, 
                    callbacks=[early_stopping, reduce_lr])



loss, acc = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {acc:.4f}")
print(f"Validation Loss: {loss:.4f}")


model.save("my_final_model.h5")


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

acc_score = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy Score: {acc_score:.4f}")



import matplotlib.pyplot as plt

# Step 1: دمج كل ال epochs
full_history = {
    'loss': history.history['loss'] ,
    'val_loss': history.history['val_loss'] ,
    'accuracy': history.history['accuracy'] ,
    'val_accuracy': history.history['val_accuracy'] 
}

# Step 2: عدد الـ epochs الإجمالية
epochs_range = range(1, len(full_history['loss']) + 1)

# Step 3: الرسم
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, full_history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, full_history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, full_history['loss'], label='Training Loss')
plt.plot(epochs_range, full_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()



test = glob.glob('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/*.jpg')


X_test = []
for path in test:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    X_test.append(img)

X_test = np.array(X_test)


y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_pred_labels = [inv_label_map[i] for i in y_pred]
df_test = pd.DataFrame({
    'filename': [os.path.splitext(os.path.basename(p))[0] + '.jpg' for p in test],
    'label': y_pred_labels
})
df_test.head()




df_test.to_csv('submission.csv', index=False)
kaggle competitions submit -c sheep-classification-challenge-2025 -f submission.csv -m "Message"

from IPython.display import FileLink
FileLink('submission.csv')

