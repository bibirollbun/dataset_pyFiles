import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import collections
import cv2



train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

train_df.head()


train_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(train_data_dir, x))
train_df['label'] = train_df['label'].astype(str)



print(train_df['label'].value_counts())



train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest',
    validation_split=0.1
    
)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_name',
    y_col='label',
    class_mode='binary',
    target_size=(128, 128),
    batch_size=32,
    subset='training',
    shuffle=True
)


val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False  )



images, labels = next(train_generator)
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Label: {int(labels[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()


labels_list = []

for i in range(5):  
    _, labels = next(train_generator)
    labels_list.extend(labels)

label_counts = collections.Counter(labels_list)

plt.bar(['Label 0', 'Label 1'], [label_counts[0], label_counts[1]])
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
plt.show()


augmented_images, _ = next(train_generator)

plt.figure(figsize=(12, 4))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
plt.suptitle('Augmented Samples')
plt.show()



unique, counts = np.unique(train_generator.classes, return_counts=True)
print(dict(zip(unique, counts)))



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()



from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)



history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)


for layer in base_model.layers[-20:]:
    layer.trainable = True

# re compile
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)



model.save_weights("resnet50.weights.h5")



val_generator.reset()
y_probs = model.predict(val_generator, verbose=1)
y_pred = (y_probs > 0.5).astype(int).ravel()
y_true = val_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("ðŸ“‹ Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))

print("\nðŸ§± Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



test_data_dir = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2"
test_df['file_name'] = test_df['id'].apply(lambda x: os.path.join("/kaggle/input/ai-vs-human-generated-dataset", x))

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='file_name',
    y_col=None,
    class_mode=None,
    target_size=(128, 128),
    batch_size=32,
    shuffle=False
)


predictions = model.predict(test_generator, verbose=1)

# Assign predicted labels (0 or 1) based on threshold
test_df['label'] = (predictions > 0.5).astype(int)

# Extract image ID from file_name (remove path and extension)
test_df['id'] = test_df['file_name'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

# Create the submission DataFrame with required columns
submission = test_df[['id', 'label']]

# Export to CSV file for submission
submission.to_csv('submission.csv', index=False)



import matplotlib.pyplot as plt

# Get accuracy and val_accuracy separately
acc_before = history.history['accuracy']
val_acc_before = history.history['val_accuracy']
acc_after = fine_tune_history.history['accuracy']
val_acc_after = fine_tune_history.history['val_accuracy']

# Create epochs index
epochs_before = range(1, len(acc_before) + 1)
epochs_after = range(len(acc_before) + 1, len(acc_before) + len(acc_after) + 1)

plt.figure(figsize=(10, 6))

# Plot before fine-tuning
plt.plot(epochs_before, acc_before, 'b--', label='Train Acc (Before Tuning)')
plt.plot(epochs_before, val_acc_before, 'r--', label='Val Acc (Before Tuning)')

# Plot after fine-tuning
plt.plot(epochs_after, acc_after, 'b-', label='Train Acc (After Tuning)')
plt.plot(epochs_after, val_acc_after, 'r-', label='Val Acc (After Tuning)')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Before and After Fine-Tuning')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Get training and validation loss
loss_before = history.history['loss']
val_loss_before = history.history['val_loss']
loss_after = fine_tune_history.history['loss']
val_loss_after = fine_tune_history.history['val_loss']

# Create epochs index
epochs_before = range(1, len(loss_before) + 1)
epochs_after = range(len(loss_before) + 1, len(loss_before) + len(loss_after) + 1)

plt.figure(figsize=(10, 6))

# Plot before fine-tuning
plt.plot(epochs_before, loss_before, 'g--', label='Train Loss (Before Tuning)')
plt.plot(epochs_before, val_loss_before, 'orange', linestyle='--', label='Val Loss (Before Tuning)')

# Plot after fine-tuning
plt.plot(epochs_after, loss_after, 'g-', label='Train Loss (After Tuning)')
plt.plot(epochs_after, val_loss_after, 'orange', label='Val Loss (After Tuning)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Before and After Fine-Tuning')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





