import numpy as np
import pandas as pd
import seaborn as sns
import keras, os
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D, LeakyReLU, SpatialDropout2D
from keras import regularizers, optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle




#Reading in the training data, we'll merge this with the given labels
pdir = '/kaggle/input/histopathologic-cancer-detection/'
train_dir = pdir + 'train/'

df_train = pd.DataFrame({'path': glob.glob(os.path.join(train_dir, '*.tif'))}) #Take all .tif files in training folder
df_train['id'] = df_train['path'].str.extract(r'([^//]+).tif$') #Use filenames as ID
labels = pd.read_csv(pdir + 'train_labels.csv')
df_train = df_train.merge(labels, on='id') #merge with labels
print(df_train.head(5))
print(df_train.info())
print(df_train['path'][0])


#Creating dataframe with the test data in the same way
test_dir = pdir + 'test/'
df_test = pd.DataFrame({'path': glob.glob(os.path.join(test_dir, '*.tif'))})
df_test['id'] = df_test['path'].str.extract(r'([^//]+).tif$')
print(df_test.head(5))
print(df_test.info())
print(df_test['path'][0])


#Checking the distribution of classes
counts = df_train['label'].value_counts()
print(counts)
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel('Label')
plt.ylabel('Number of Observations')
plt.title('Distribution of Labels in the Training Data')
plt.xticks(ticks=[0, 1], labels=['0: No Cancer', '1: Cancer'])
plt.show()


#Balancing the data by undersampling class 0
c0 = df_train[df_train['label'] == 0]
c1 = df_train[df_train['label'] == 1]

c0_bal = c0.sample(n=len(c1), random_state=1)

df_trainbal = pd.concat([c0_bal, c1], axis=0)

df_trainbal = df_trainbal.sample(frac=1, random_state=1).reset_index(drop=True)


#Checking the distribution of classes
counts = df_trainbal['label'].value_counts()
print(counts)
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel('Label')
plt.ylabel('Number of Observations')
plt.title('Distribution of Labels in the Balanced Training Data')
plt.xticks(ticks=[0, 1], labels=['0: No Cancer', '1: Cancer'])
plt.show()


random_samples = np.random.randint(1, len(df_trainbal) + 1, size=15)
for i in random_samples:

    image = mpimg.imread(df_trainbal['path'][i])
    imageplot = plt.imshow(image)
    plt.title('Label: ' + df_trainbal['label'][i].astype(str))
    plt.show()


#Let's add a column with the full filenames for use with imagedatagenerator and labels must be strings
df_trainbal['filename'] = df_trainbal['id'] + '.tif'
df_test['filename'] = df_test['id'] + '.tif'
df_trainbal['label'] = df_trainbal['label'].astype(str)





#Create training and validation sets with image datagenerator
train_df, val_df = train_test_split(df_trainbal, test_size=0.2, stratify=df_trainbal['label'], random_state=1)
train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                                      vertical_flip=True, zoom_range=0.2, shear_range=0.2, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory='/kaggle/input/histopathologic-cancer-detection/train',
                                                    x_col='filename',
                                                    y_col='label',
                                                    target_size=(96, 96),
                                                    batch_size=32,
                                                    class_mode='binary')
val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,
                                                    directory='/kaggle/input/histopathologic-cancer-detection/train',
                                                    x_col='filename',
                                                    y_col='label',
                                                    target_size=(96, 96),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    shuffle=False)


#Create the test image datagenerator
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                    directory='/kaggle/input/histopathologic-cancer-detection/test',
                                                    x_col='filename',
                                                    y_col=None,
                                                    target_size=(96, 96),
                                                    batch_size=32,
                                                    class_mode=None,
                                                    shuffle=False)


from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

initial_lr = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

# Compile with updated optimizer
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)
print(model.summary())


# Enhanced callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# Train with more epochs and callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=callbacks,
    verbose=1
)

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Baseline Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Baseline Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


model.save('/kaggle/working/my_model.keras')
with open('/kaggle/working/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Get true labels and predicted labels
val_generator.reset()
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices.keys())
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
plt.savefig('EfficientNetB0transferlearningtrainingCM.png')


from sklearn.metrics import roc_curve, auc

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('mobilenetAUCROC.png')


predictions = model.predict(test_generator, verbose=1)
submission = df_test.copy()
submission = submission.drop(columns=['filename', 'path'])
submission['label'] = predictions
print(submission.head())
submission.to_csv('MobileNetv2.csv', index=False)


submission.to_csv('submission.csv', index=False)

