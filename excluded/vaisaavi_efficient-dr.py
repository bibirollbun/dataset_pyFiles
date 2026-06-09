# To have reproducible results and compare them
nr_seed = 2019
import numpy as np 
np.random.seed(nr_seed)
import tensorflow as tf
tf.random.set_seed(nr_seed)


# import libraries
import json
import math
from tqdm import tqdm, tqdm_notebook
import gc
import warnings
import os

import cv2
from PIL import Image

import pandas as pd
import scipy
import matplotlib.pyplot as plt

from keras import backend as K
from keras import layers
from keras.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

warnings.filterwarnings("ignore")

%matplotlib inline


# Image size
im_size = 224
# Batch size
BATCH_SIZE = 4


new_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
old_train = pd.read_csv('../input/diabetic-retinopathy-resized/trainLabels.csv')
print(new_train.shape)
print(old_train.shape)


old_train = old_train[['image','level']]
old_train.columns = new_train.columns
old_train.diagnosis.value_counts()

# path columns
new_train['id_code'] = '../input/aptos2019-blindness-detection/train_images/' + new_train['id_code'].astype(str) + '.png'
old_train['id_code'] = '../input/diabetic-retinopathy-resized/resized_train/resized_train/' + old_train['id_code'].astype(str) + '.jpeg'

train_df = old_train.copy()
val_df = new_train.copy()
train_df.head()


# Not used in version 5
#train_df, val_df = train_test_split(train_df, shuffle=True, stratify=train_df.diagnosis, test_size=0.1, random_state=2019)


# Let's shuffle the datasets
train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
print(train_df.shape)
print(val_df.shape)


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image_path, desired_size=im_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)
    
    return img

def preprocess_image_old(image_path, desired_size=im_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)
    
    return img


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'{image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = crop_image_from_gray(img)
        img = cv2.resize(img, (im_size,im_size))
        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), im_size/40) ,-4 ,128)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# Validation set
N = val_df.shape[0]
x_val = np.empty((N, im_size, im_size, 3), dtype=np.float32)

for i, image_id in enumerate(tqdm_notebook(val_df['id_code'])):
    img = preprocess_image(f'{image_id}', desired_size=im_size)
    x_val[i, :, :, :] = preprocess_input(img)



y_train = pd.get_dummies(train_df['diagnosis']).values
y_val = pd.get_dummies(val_df['diagnosis']).values

print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

y_val_multi = np.empty(y_val.shape, dtype=y_val.dtype)
y_val_multi[:, 4] = y_val[:, 4]

for i in range(3, -1, -1):
    y_val_multi[:, i] = np.logical_or(y_val[:, i], y_val_multi[:, i+1])

print("Y_train multi: {}".format(y_train_multi.shape))
print("Y_val multi: {}".format(y_val_multi.shape))


y_train = y_train_multi
y_val = y_val_multi


# delete the uneeded df
del new_train
del old_train
del val_df
gc.collect()


class Metrics(Callback):

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


from keras.applications.efficientnet import preprocess_input

def create_datagen():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ✅ Use EfficientNetB7-specific preprocessing
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360
    )



# efficientnet = EfficientNetB7(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(im_size, im_size, 3)
# )


# def build_model():
#     model = Sequential()
#     model.add(efficientnet)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(5, activation='sigmoid'))

#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(learning_rate=1e-4, decay=1e-6),  
#         metrics=['accuracy']
#     )

#     return model


def build_model(input_shape=(im_size, im_size, 3)):
    base_model = ResNet34(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid', kernel_regularizer=l2(0.0001)))

    model.compile(
        optimizer=Adam(learning_rate=1e-4, decay=1e-6),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model



# def build_model(input_shape=(im_size, im_size, 3)):
#     efficientnet = EfficientNetB7(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
#     efficientnet.trainable = True # Or False initially, then True for fine-tuning later

#     model = models.Sequential()
#     model.add(efficientnet)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.BatchNormalization()) # Add Batch Normalization
#     model.add(layers.Dropout(0.5))        # Keep Dropout (or adjust)
#     model.add(layers.Dense(5, activation='sigmoid'))

#     model.compile(
#         optimizer=Adam(learning_rate=1e-4, decay=1e-6), # Or try 1e-5
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


model = build_model()
model.summary()


#train_df = train_df.reset_index(drop=True)
bucket_num = 8
div = round(train_df.shape[0]/bucket_num)


df_init = {
    'val_loss': [0.0],
    'val_acc': [0.0],
    'loss': [0.0], 
    'acc': [0.0],
    'bucket': [0.0]
}
results = pd.DataFrame(df_init)


from sklearn.metrics import cohen_kappa_score
from keras.callbacks import Callback

# Custom callback class to calculate Quadratic Kappa
class KappaMetrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_val, y_val = self.validation_data

        y_val_true = y_val.argmax(axis=1)
        y_val_pred = self.model.predict(x_val)
        y_val_pred = y_val_pred.argmax(axis=1)

        kappa = cohen_kappa_score(y_val_true, y_val_pred, weights='quadratic')
        self.val_kappas.append(kappa)

        print(f"\nval_kappa: {kappa:.4f}")
        logs['val_kappa'] = kappa

# Epochs for each bucket
epochs = [4, 4, 8, 10, 10, 12, 12, 15]  # Reduced epochs  # Reduced epochs from the 3rd bucket onwards

# Pass validation data to the callback
kappa_metrics = KappaMetrics(validation_data=(x_val, y_val))



# for i in range(0, bucket_num):
#     if i != (bucket_num-1):
#         print("Bucket Nr: {}".format(i))

#         N = train_df.iloc[i*div:(1+i)*div].shape[0]
#         x_train = np.empty((N, im_size, im_size, 3), dtype=np.float32)  # ✅ float32 for preprocess_input

#         for j, image_id in enumerate(tqdm_notebook(train_df.iloc[i*div:(1+i)*div, 0])):
#             img = preprocess_image(f'{image_id}', desired_size=im_size)  # ✅ Use updated function
#             x_train[j, :, :, :] = img  # ✅ Apply EfficientNetB7 preprocessing

#         data_generator = create_datagen().flow(x_train, y_train[i*div:(1+i)*div, :], batch_size=BATCH_SIZE)

#         history = model.fit(
#             data_generator,
#             steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
#             epochs=epochs[i],
#             validation_data=(x_val, y_val),
#             callbacks=[kappa_metrics]
#         )

#         dic = history.history
#         df_model = pd.DataFrame(dic)
#         df_model['bucket'] = i

#     else:
#         print("Bucket Nr: {}".format(i))

#         N = train_df.iloc[i*div:].shape[0]
#         x_train = np.empty((N, im_size, im_size, 3), dtype=np.float32)

#         for j, image_id in enumerate(tqdm_notebook(train_df.iloc[i*div:, 0])):
#             img = preprocess_image(f'{image_id}', desired_size=im_size)
#             x_train[j, :, :, :] = img

#         data_generator = create_datagen().flow(x_train, y_train[i*div:, :], batch_size=BATCH_SIZE)

#         history = model.fit(
#             data_generator,
#             steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
#             epochs=epochs[i],
#             validation_data=(x_val, y_val),
#             callbacks=[kappa_metrics]
#         )

#         dic = history.history
#         df_model = pd.DataFrame(dic)
#         df_model['bucket'] = i

#     results = results.append(df_model)

#     del data_generator
#     del x_train
#     gc.collect()
#     K.clear_session()
#     tf.compat.v1.reset_default_graph()

#     print('-'*40)



for i in range(bucket_num):
    print(f"Bucket Nr: {i}")

    # Rebuild model for each bucket – this ensures a fresh start and helps free memory
    model = build_model()

    # Reinitialize the KappaMetrics callback with current validation data
    kappa_metrics = KappaMetrics(validation_data=(x_val, y_val))
    
    # Select the current bucket's training data and labels
    if i != (bucket_num - 1):
        current_df = train_df.iloc[i*div:(i+1)*div]
        current_labels = y_train[i*div:(i+1)*div, :]
    else:
        current_df = train_df.iloc[i*div:]
        current_labels = y_train[i*div:, :]

    # Load images for the current bucket into x_train
    N = current_df.shape[0]
    x_train = np.empty((N, im_size, im_size, 3), dtype=np.float32)
    for j, image_id in enumerate(tqdm_notebook(current_df['id_code'])):
        # Load image using your custom function. It should return the resized image.
        img = preprocess_image(f'{image_id}', desired_size=im_size)
        # DO NOT call preprocess_input here because create_datagen() already applies it.
        x_train[j, :, :, :] = img

    # Create the data generator for this bucket; it applies preprocessing_function internally.
    data_generator = create_datagen().flow(x_train, current_labels, batch_size=BATCH_SIZE)

    # Train the model on the current bucket
    history = model.fit(
        data_generator,
        steps_per_epoch = x_train.shape[0] // BATCH_SIZE,
        epochs = epochs[i],
        validation_data = (x_val, y_val),
        callbacks = [kappa_metrics]
    )

    # Save training history for analysis
    df_model = pd.DataFrame(history.history)
    df_model['bucket'] = i
    results = pd.concat([results, df_model], ignore_index=True)

    # Cleanup to free memory
    del model, data_generator, x_train
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

    print('-' * 40)



results = results.iloc[1:]
results['kappa'] = kappa_metrics.val_kappas
results = results.reset_index()
results = results.rename(columns={"index": "epoch"})  # ✅ removed deprecated `index=str`
results



results[['loss', 'val_loss']].plot()
results[['acc', 'val_acc']].plot()
results[['kappa']].plot()
results.to_csv('model_results.csv',index=False)


model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

def compute_score_inv(threshold):
    y1 = y_val_pred > threshold
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)

best_threshold = simplex['x'][0]

y1 = y_val_pred > best_threshold
y1 = y1.astype(int).sum(axis=1) - 1
y2 = y_val.sum(axis=1) - 1
score = cohen_kappa_score(y1, y2, weights='quadratic')
print('Threshold: {}'.format(best_threshold))
print('Validation QWK score with best_threshold: {}'.format(score))

y1 = y_val_pred > .5
y1 = y1.astype(int).sum(axis=1) - 1
score = cohen_kappa_score(y1, y2, weights='quadratic')
print('Validation QWK score with .5 threshold: {}'.format(score))


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predicted and actual labels
y_pred_final = (y_val_pred > best_threshold).astype(int).sum(axis=1) - 1
y_true_final = y_val.sum(axis=1).astype(int) - 1

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_final, y_pred_final))

# Confusion Matrix
cm = confusion_matrix(y_true_final, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import classification_report

# Convert predictions to final labels using the best threshold
y_pred_best = (y_val_pred > best_threshold).astype(int)
# Sum across the binary predictions (as done previously) to get class indices
y_pred_labels = y_pred_best.sum(axis=1) - 1
y_true = y_val.sum(axis=1) - 1

print("Classification Report with Best Threshold:")
print(classification_report(y_true, y_pred_labels))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Best Threshold)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()



from sklearn.metrics import precision_score, recall_score, f1_score

# Compute weighted metrics
precision = precision_score(y_true, y_pred_labels, average='weighted')
recall = recall_score(y_true, y_pred_labels, average='weighted')
f1 = f1_score(y_true, y_pred_labels, average='weighted')

print("Performance Metrics (Weighted):")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")



results[['loss', 'val_loss']].plot(title='Loss over Epochs')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

results[['acc', 'val_acc']].plot(title='Accuracy over Epochs')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


from sklearn.metrics import roc_curve, auc
import numpy as np

n_classes = y_val.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC curve and AUC for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10,8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()



import seaborn as sns

# Plot distribution of predicted class labels
plt.figure(figsize=(8,6))
sns.countplot(x=y_pred_labels)
plt.title("Distribution of Predicted Class Labels (Best Threshold)")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Plot distribution of true class labels
plt.figure(figsize=(8,6))
sns.countplot(x=y_true)
plt.title("Distribution of True Class Labels")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

















