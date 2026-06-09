"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""


#imports 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf



#Read
df_train = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv') 

print (df_train.head())
print (df_train.shape)

#As we are going to need URL to images we are going to create new colummn with this path
#path to actual images in new colummn
df_train['location'] = '/kaggle/input/histopathologic-cancer-detection/train/' + df_train['id'] + '.tif'

#Most networks will need label as string so lets convert it
df_train["label"] = df_train["label"].astype(str)  # Convert to string

#We are first going to test various networks so lest get a random sample, this is for reducing the time of trining
df_sampled = df_train.sample(n=1024, random_state=42)  # Set seed for reproducibility



#NN

print ('Overview of dataset')
print(df_train.info())  
print()
duplicates = df_train.duplicated(subset=['id']).sum()
print(f'Duplicates\n{duplicates}')
print()
print('Check missing values')
print(df_train.isnull().sum())
print()
print('Labels')
df_train['label'].value_counts()

# Count values
total = len(df_train)  # Total number of rows
plt.figure(figsize=(6,6))

ax = sns.countplot(data=df_train, x='label', palette='viridis')

# Add percentages on bars
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.2f}%'  # Calculate %
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, color='black')

plt.title('Class Distribution with Percentages')
plt.show()

#preview of first 10 images ant its labels
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns

for i, ax in enumerate(axes.flat):
    img = mpimg.imread(df_train['location'].iloc[i])  # Load image
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    ax.set_title(f"Label: {df_train.loc[i, 'label']}")  # Show label

plt.tight_layout()
plt.show()




#imports for CNNs

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge
from tensorflow.keras.metrics import AUC



# Cheking similar distribution between sample and complete df.

total = len(df_sampled)  # Total number of rows
plt.figure(figsize=(6,6))

ax = sns.countplot(data=df_sampled, x='label', palette='viridis')

# Add percentages on bars
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.2f}%'  # Calculate %
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, color='black')

plt.title('Class Distribution with Percentages')
plt.show()



# Load EfficientNetB0  (Feature Extractor)
EfficientNet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  
EfficientNet_model.trainable = True  # Allow fine-tuning

# Add Custom Classifier for Binary Classification
x = GlobalAveragePooling2D()(EfficientNet_model.output)  # Convert feature maps to vector
x = Dense(128, activation="relu")(x)  # Dense Layer
x = Dense(1, activation="sigmoid")(x)  # Binary Classification (Sigmoid Output)

# Create Model
model_Eff = Model(inputs=EfficientNet_model.input, outputs=x)

# Compile Model (Use Binary Loss & AUC Metric)
model_Eff.compile(optimizer=Adam(learning_rate=1e-5), 
              loss="binary_crossentropy", 
              metrics=["binary_accuracy", AUC(name="auc")])

datagen_Eff = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  # 80-20 train-val split
)
train_generator_Eff = datagen_Eff.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(224, 224), batch_size=64, subset="training",
    class_mode="binary"  # Important for Binary Classification
)

val_generator_Eff = datagen_Eff.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(224, 224), batch_size=64, subset="validation",
    class_mode="binary"
)

# Train the Model
model_Eff.fit(train_generator_Eff, validation_data=val_generator_Eff, epochs=10)

# Evaluate on Validation Set
val_loss, val_acc, val_auc = model_Eff.evaluate(val_generator_Eff)
print(f"âœ… EfficientNetB0 Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")




#ResNet50 
Res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
Res_model.trainable = True  # Allow fine-tuning

# Add Custom Classifier for Binary Classification
x = GlobalAveragePooling2D()(Res_model.output)  # Convert feature maps to vector
x = Dense(128, activation="relu")(x)  # Dense Layer
x = Dense(1, activation="sigmoid")(x)  # Binary Classification (Sigmoid Output)

# Create Model
model_Res = Model(inputs=Res_model.input, outputs=x)

# Compile Model (Use Binary Loss & AUC Metric)
model_Res.compile(optimizer=Adam(learning_rate=1e-5), 
              loss="binary_crossentropy", 
              metrics=["binary_accuracy", AUC(name="auc")])

datagen_Res = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  # 80-20 train-val split
)
train_generator_Res = datagen_Res.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="training",
    class_mode="binary"
)

val_generator_Res = datagen_Res.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="validation",
    class_mode="binary"
)

# Train the Model
model_Res.fit(train_generator_Res, validation_data=val_generator_Res, epochs=10)

# Evaluate on Validation Set
val_loss, val_acc, val_auc = model_Res.evaluate(val_generator_Res)
print(f"âœ… ResNet50 Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")# Load ResNet50 WITHOUT Top Layers (Feature Extractor)


#Load DenseNet-169 WITHOUT Top Layers (Feature Extractor)
DenseN_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
DenseN_model.trainable = True  # Allow fine-tuning

# Add Custom Classifier for Binary Classification
x = GlobalAveragePooling2D()(DenseN_model.output)  # Convert feature maps to vector
x = Dense(128, activation="relu")(x)  # Dense Layer
x = Dense(1, activation="sigmoid")(x)  # Binary Classification (Sigmoid Output)

# Create Model
model_DenseN = Model(inputs=DenseN_model.input, outputs=x)

# Compile Model (Use Binary Loss & AUC Metric)
model_DenseN.compile(optimizer=Adam(learning_rate=1e-5), 
              loss="binary_crossentropy", 
              metrics=["binary_accuracy", AUC(name="auc")])

datagen_DenseN = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  # 80-20 train-val split
)
train_generator_DenseN = datagen_DenseN.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="training",
    class_mode="binary"
)

val_generator_DenseN = datagen_DenseN.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="validation",
    class_mode="binary"
)

# Train the Model
model_DenseN.fit(train_generator_DenseN, validation_data=val_generator_DenseN, epochs=10)

# Evaluate on Validation Set
val_loss, val_acc, val_auc = model_DenseN.evaluate(val_generator_DenseN)
print(f"âœ… DenseNet-169 Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")


import time

# Train EfficientNetB0
start_time = time.time()
history_efficient = model_Eff.fit(train_generator_Eff, validation_data=val_generator_Eff, epochs=10)
efficient_time = time.time() - start_time

# Train ResNet50
start_time = time.time()
history_resnet = model_Res.fit(train_generator_Res, validation_data=val_generator_Res, epochs=10)
resnet_time = time.time() - start_time

# Train DenseNet-169
start_time = time.time()
history_densenet = model_DenseN.fit(train_generator_DenseN, validation_data=val_generator_DenseN, epochs=10)
densenet_time = time.time() - start_time

# Extract Performance Metrics
epochs = range(1, 11)

efficient_acc = history_efficient.history["val_binary_accuracy"]
efficient_auc = history_efficient.history["val_auc"]

resnet_acc = history_resnet.history["val_binary_accuracy"]
resnet_auc = history_resnet.history["val_auc"]

densenet_acc = history_densenet.history["val_binary_accuracy"]
densenet_auc = history_densenet.history["val_auc"]

# ğŸ“Œ Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, efficient_acc, label="EfficientNetB0", marker="o")
plt.plot(epochs, resnet_acc, label="ResNet50", marker="o")
plt.plot(epochs, densenet_acc, label="DenseNet-169", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Model Comparison: Validation Accuracy")
plt.legend()

# ğŸ“Œ Plot AUC
plt.subplot(1, 2, 2)
plt.plot(epochs, efficient_auc, label="EfficientNetB0", marker="o")
plt.plot(epochs, resnet_auc, label="ResNet50", marker="o")
plt.plot(epochs, densenet_auc, label="DenseNet-169", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Validation AUC")
plt.title("Model Comparison: AUC Score")
plt.legend()

plt.show()

# ğŸ“Œ Compare Training Speed
print(f"â�³ Training Time (Seconds):")
print(f"âœ… EfficientNetB0: {efficient_time:.2f} sec")
print(f"âœ… ResNet50: {resnet_time:.2f} sec")
print(f"âœ… DenseNet-169: {densenet_time:.2f} sec")



import keras_tuner as kt

def build_model(hp):
    base_model = DenseNet169(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = True  # Allow fine-tuning
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # ğŸ“Œ Hyperparameters for tuning
    learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
    optimizer = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])
    loss = hp.Choice("loss", ["binary_crossentropy", "hinge"])
    
    # Select optimizer
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=opt, loss=loss, metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")])
    
    return model

tuner = kt.GridSearch(
    hypermodel=build_model,
    objective="val_binary_accuracy",
    max_trials=10,  # Number of different combinations to try
    executions_per_trial=1  # Run each configuration once
)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="training",
    class_mode="binary"
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df_sampled, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="validation",
    class_mode="binary"
)

# Perform hyperparameter search
tuner.search(train_generator, validation_data=val_generator, epochs=5)

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print(f"Best Learning Rate: {best_hps.get('learning_rate')}")
print(f"Best Optimizer: {best_hps.get('optimizer')}")
print(f"Best Loss Function: {best_hps.get('loss')}")



#Data for final model

datagen_final = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  # 80-20 train-val split
)
train_generator_final = datagen_final.flow_from_dataframe(
    dataframe=df_train, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="training",
    class_mode="binary"
)

val_generator_final = datagen_final.flow_from_dataframe(
    dataframe=df_train, directory=None, x_col="location", y_col="label",
    target_size=(96, 96), batch_size=64, subset="validation",
    class_mode="binary"
)


#Final model with best features
final_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
final_model.trainable = True  # Allow fine-tuning

# Add Custom Classifier for Binary Classification
x = GlobalAveragePooling2D()(final_model.output)  # Convert feature maps to vector
x = Dense(128, activation="relu")(x)  # Dense Layer
x = Dense(1, activation="sigmoid")(x)  # Binary Classification (Sigmoid Output)

# Create Model
model_final = Model(inputs=final_model.input, outputs=x)

# Compile Model (Use Binary Loss & AUC Metric)
model_final.compile(optimizer=Adam(learning_rate=0.0001), 
              loss="hinge", 
              metrics=["binary_accuracy", AUC(name="auc")])



final_history = model_final.fit(train_generator_final, validation_data=val_generator_final, epochs=5)


val_loss, val_acc, val_auc = model_final.evaluate(val_generator_final)
print(f"âœ… FINAL MODEL on Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")



# Extract history from your existing object
epochs = range(1, 6)

final_acc = final_history.history["val_binary_accuracy"]
final_auc = final_history.history["val_auc"]

# ğŸ“Œ Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, final_acc, label="Final Model", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy over Time")
plt.legend()

# ğŸ“Œ Plot AUC
plt.subplot(1, 2, 2)
plt.plot(epochs, final_auc, label="Final Model", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Validation AUC")
plt.title("Model AUC Score over Time")
plt.legend()

plt.show()



# working on test for submition
test_dir = "/kaggle/input/histopathologic-cancer-detection/test"

# Create DataFrame with columns 'id' and 'path'
df_test = pd.DataFrame({
    'id': [f for f in os.listdir(test_dir)],
    'path': [os.path.join(test_dir, f) for f in os.listdir(test_dir)],
    'predicted label': None  # Placeholder for predictions
})

print(df_test.head())


# Define ImageDataGenerator
datagen_test = ImageDataGenerator(rescale=1./255)

# Create a test generator
test_generator = datagen_test.flow_from_dataframe(
    dataframe=df_test,
    x_col="path",   # The column with image file paths
    target_size=(96, 96),  # Match model input size
    batch_size=64,  # Adjust batch size based on GPU memory
    class_mode=None,  # No labels for test data
    shuffle=False  # Keep order for consistent results
)

# Predict using the generator
predictions = model_final.predict(test_generator)

# Convert sigmoid outputs to binary labels
df_test["predicted label"] = (predictions >= 0.5).astype(int)

print(df_test.head())



# Save results to CSV
df_test.to_csv("submission.csv", index=False)

