pip install -q tensorflow


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score,classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


labels_df=pd.read_csv("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv")
labels_df


labels_df['label'].value_counts()


labels_df['label'].unique()


label_counts=labels_df['label'].value_counts()
ax=label_counts.plot(kind='bar',color='yellow',edgecolor='black')
plt.title("Sheep Breed Distribution")
plt.ylabel("Breed")
plt.xlabel("Count")
plt.xticks(rotation=45)
plt.show()



# Path to train images
image_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"

# Read the labels CSV
labels_df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')

# Get one sample image path for each breed
sample_images = labels_df.groupby('label').first().reset_index()

# Set figure size for visualization
plt.figure(figsize=(15, 8))

# Plot 1 sample image per breed
for i, row in enumerate(sample_images.itertuples()):
    img_path = os.path.join(image_dir, row.filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(row.label)
    plt.axis('off')

plt.tight_layout()
plt.show()



# Encode breed labels
le=LabelEncoder()
labels_df['breed_label']=le.fit_transform(labels_df['label'])
labels_df['breed_label'] = labels_df['breed_label'].astype(str)

labels_df


image_size=(224,224)
batch_size=32
train_dir="/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"
labels_df['filepath']=labels_df['filename'].apply(lambda x:os.path.join(train_dir,x))
labels_df


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_gen = datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='filepath',
    y_col='breed_label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=42
)
val_gen = datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='filepath',
    y_col='breed_label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)



datagen


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_df['breed_label']),
    y=labels_df['breed_label']
)
class_weights = dict(enumerate(class_weights))



base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])




early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)



# Unfreeze base model
base_model.trainable = True

# Compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weights
)



model.load_weights("best_model.h5")

# Predict
val_preds = model.predict(val_gen, verbose=1)
val_pred_classes = np.argmax(val_preds, axis=1)

# Get true labels
true_classes = []
for i in range(len(val_gen)):
    _, y = val_gen[i]
    true_classes.extend(y)

true_classes = np.array(true_classes).astype(int)

# F1 Score
f1 = f1_score(true_classes, val_pred_classes, average='macro')
print(f"Validation F1 Score: {f1:.4f}")

# Classification report
print(classification_report(true_classes, val_pred_classes, target_names=le.classes_))



# 1. Create DataFrame for test images
test_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
test_images = sorted(os.listdir(test_dir))

test_df = pd.DataFrame({
    'filename': test_images,
    'filepath': [os.path.join(test_dir, img) for img in test_images]
})

# 2. Create test data generator (only rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col=None,                    # No labels
    target_size=image_size,
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# 3. Load best weights
model.load_weights("best_model.h5")

# 4. Predict on test data
test_preds = model.predict(test_gen)
predicted_classes = np.argmax(test_preds, axis=1)

# 5. Map predictions back to breed names
test_df['predicted_label_index'] = predicted_classes
test_df['predicted_breed'] = le.inverse_transform(predicted_classes)

# 6. Display results
test_df[['filename', 'predicted_breed']].head()



submission = test_df[['filename', 'predicted_breed']]
submission.to_csv('submission.csv', index=False)
submission.head()


submission = pd.DataFrame({
    "filename": test_filenames,
    "label": final_preds
})
submission.to_csv("submission.csv", index=False)

submission.head()


labels_df['label'].value_counts()




