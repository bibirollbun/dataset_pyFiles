import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import InceptionV3






df = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
df.head()


real = df[df['label'] == 'real']
fake = df[df['label'] != 'real']


fake.iloc[:5, 0]


train_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train'
test_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test'


import matplotlib.pyplot as plt
import os
import random
from PIL import Image


fake_sample = fake.iloc[:5, 0]
real_sample = real.iloc[:5, 0]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, fake_image in enumerate(fake_sample):
    img = Image.open(os.path.join(train_dir, fake_image))
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title('Fake')

for i, real_image in enumerate(real_sample):
    img = Image.open(os.path.join(train_dir, real_image))
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title('Real')

plt.tight_layout()
plt.show()



df.shape


df["image_path"] = df["image"].apply(lambda x: os.path.join(train_dir, x))
df.head()
label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])


df['label'].value_counts().plot.bar()


label_mappings = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_mappings


train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"])


print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")


batch_size = 32
image_size = (224, 224, 3)


def train_and_eval(preprocess_input, model):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",      
        y_col="label_encoded",       
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode="raw",        
        shuffle=True,            
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",      
        y_col="label_encoded",       
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode="raw",    
        shuffle=False           
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    steps_per_epoch = train_generator.n // 32
    validation_steps = val_generator.n // 32
    
    history = model.fit(
        train_generator,
        batch_size=32,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=30,
        callbacks=[early_stopping]
    )
    plot_auc(val_generator, model)


def plot_auc(generator, model):
    y_true = generator.labels  # True labels
    
    # Predict probabilities
    y_pred = model.predict(generator, batch_size=32)
    
    # For binary classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc}')
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



def preprocess_to_hsv(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_hsv[..., 0] = image_hsv[..., 0] / 180.0  # Hue range: 0-179 (OpenCV) to 0-1
    image_hsv[..., 1] = image_hsv[..., 1] / 255.0  # Saturation range: 0-255 to 0-1
    image_hsv[..., 2] = image_hsv[..., 2] / 255.0  # Value range: 0-255 to 0-1
    
    return image_hsv


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

efficientnet_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])


train_and_eval(preprocess_to_hsv, efficientnet_model)


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(1024, activation='relu')(x)  
predictions = Dense(1, activation='sigmoid')(x)  

inception_model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


train_and_eval(preprocess_to_hsv, inception_model)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = Flatten()(x)  
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  

resnet_model = Model(inputs=base_model.input, outputs=x)



train_and_eval(preprocess_to_hsv, resnet_model)


base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = Flatten()(x)  
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  

densenet_model = Model(inputs=base_model.input, outputs=x)


train_and_eval(preprocess_to_hsv, densenet_model)


def predict_and_save_csv(preprocess_to_hsv, feature_extractor, svm_model, csv_path):
    image_paths = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # or other image formats
            image_paths.append(os.path.join(test_dir, filename))
    
    test_df = pd.DataFrame(image_paths, columns=["image_path"])
    
    test_df.head()
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_to_hsv 
    )
    
    # Assuming you have a dataframe `test_df` with the image paths and labels
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,                # DataFrame with image paths
        x_col="image_path",                # Column containing image paths
        target_size=(224, 224),            # Resize to match VGG input size
        batch_size=batch_size,             # Batch size for predictions
        class_mode=None,                   # No labels for prediction
        shuffle=False                      # Do not shuffle since we need the original order
    )
    X_test_features = feature_extractor_densenet.predict(test_generator)
    print(X_test_features.shape)
    
    y_pred = svm_model.predict(X_test_features)
    test_df['image'] = test_df['image_path'].str.split('/').str[-1]
    test_df['label'] = y_pred
    
    test_df = test_df[['image', 'label']]
    test_df.to_csv(csv_path, index=False)
    
    print("Predictions saved to predictions.csv")



train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_to_hsv
    )
    
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",      
    y_col="label_encoded",       
    target_size=(224, 224),  
    batch_size=batch_size,
    class_mode="raw",        
    shuffle=True,            
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_to_hsv
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",      
    y_col="label_encoded",       
    target_size=(224, 224),  
    batch_size=batch_size,
    class_mode="raw",    
    shuffle=False           
)


base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor_densenet = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

def extract_features(generator, model):
    features = []
    labels = []
    
    for batch in generator:
        batch_features = model.predict(batch[0], batch_size=32)
        features.append(batch_features)
        labels.append(batch[1])
        
        if len(features) * batch_size >= generator.samples:
            break
    
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)






X_train_features, y_train = extract_features(train_generator, feature_extractor_densenet)
X_val_features, y_val = extract_features(val_generator, feature_extractor_densenet)

print(X_train_features.shape)



svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_model.fit(X_train_features, y_train)


from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = svm_model.predict_proba(X_val_features)[:, 1]  # Get probability of positive class

auc_score = roc_auc_score(y_val, y_pred_prob)
print(f"AUC Score: {auc_score:.2f}")

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


predict_and_save_csv(preprocess_to_hsv, feature_extractor_densenet, svm_model, csv_path='dense_svm_hsv_wiht_preprocess_input_predictions.csv')


train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_to_hsv
    )
    
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col="image_path",      
    y_col="label_encoded",       
    target_size=(224, 224),  
    batch_size=batch_size,
    class_mode="raw",        
    shuffle=True,            
)


base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor_densenet = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

def extract_features(generator, model):
    features = []
    labels = []
    
    for batch in generator:
        batch_features = model.predict(batch[0], batch_size=32)
        features.append(batch_features)
        labels.append(batch[1])
        
        if len(features) * batch_size >= generator.samples:
            break
    
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)




X_train_features, y_train = extract_features(train_generator, feature_extractor_densenet)

print(X_train_features.shape)


svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_model.fit(X_train_features, y_train)


y_pred_prob = svm_model.predict_proba(X_val_features)[:, 1]  # Get probability of positive class

auc_score = roc_auc_score(y_val, y_pred_prob)
print(f"AUC Score: {auc_score:.2f}")

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


predict_and_save_csv(preprocess_to_hsv, feature_extractor_densenet, svm_model, csv_path='dense_svm_hsv_wiht_preprocess_input_with_all_df_predictions.csv')




