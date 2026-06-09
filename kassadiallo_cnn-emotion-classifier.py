import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from keras import regularizers
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/emotion-detector-zone01/test_with_emotions.csv")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



df_train.shape


df_test.shape


df_train.head()





df_test.head()


def preprocess_pixels(pixels_str):
    pixel_values = np.array(pixels_str.split(), dtype=int)  
    return pixel_values.reshape(48, 48)
    
df_train['image'] = df_train['pixels'].apply(preprocess_pixels)
df_test['image'] = df_test['pixels'].apply(preprocess_pixels)
df_train["emotion"] = df_train["emotion"].apply(int)


df_train.head()


df_train["emotion"].unique()


emotion_percentage = (df_train["emotion"].value_counts() / len(df_train)) * 100

for emotion_idx, percentage in emotion_percentage.items():
    emotion_name = emotions[emotion_idx]  
    print(f"{emotion_name}: {percentage:.2f}%")



import seaborn as sns
emotion_counts = df_train['emotion'].value_counts().sort_index()

# Configuration de la figure
plt.figure(figsize=(10, 5))

# Créer un diagramme à barres
sns.barplot(x=emotion_counts.index, y=emotion_counts.values, color='steelblue')

# Ajouter les étiquettes des émotions
plt.xticks(range(len(emotions)), emotions, rotation=45)

# Ajouter les titres et libellés
plt.title('Distribution des émotions dans le jeu de donnée', fontsize=16)
plt.xlabel('Émotion', fontsize=14)
plt.ylabel('Nombre d\'image', fontsize=14)

# Ajouter les valeurs sur chaque barre
for i, count in enumerate(emotion_counts.values):
    plt.text(i, count + 100, f"{count}", ha='center', fontsize=12)

# Ajuster la mise en page
plt.tight_layout()

# Afficher la figure
plt.show()


def plot_emotion_samples(df, emotions, num_samples=5):
    """Affiche un échantillon d'images pour chaque émotion."""
    plt.figure(figsize=(16, 16))
    
    for row, (emotion, label) in enumerate(zip(np.unique(df["emotion"]), emotions)):
        emotion_imgs = df[df.emotion == emotion]
        sample_count = min(len(emotion_imgs), num_samples)  
        
        for i in range(sample_count):
            img = np.array(emotion_imgs.iloc[i]["image"], dtype=int)
            
            plt.subplot(len(emotions), num_samples, row * num_samples + i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            plt.text(24, 52, s=label, fontsize=12, color='blue', ha='center')

    plt.show()


plot_emotion_samples(df_train, emotions)


def preprocess_data_for_cnn(df, num_classes=7):
    """Prétraitement des images et des labels pour l'entraînement du cnn."""
    # reshape dimensions
    X_list = df['pixels'].apply(lambda pixels_str: preprocess_pixels(pixels_str)).tolist()
    X = np.array(X_list)    
    # Normalisation
    X = X / 255.0  
    X = X.reshape(-1, 48, 48, 1) 
    # Reformater pour CNN    
    y = to_categorical(df["emotion"].astype(int), num_classes=num_classes)
    return X, y


X_train, y_train = preprocess_data_for_cnn(df_train)
X_test, y_test = preprocess_data_for_cnn(df_test)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=df_train["emotion"])


datagen = ImageDataGenerator(
    rotation_range=20,         # Rotation aléatoire jusqu'à 20 degrés
    width_shift_range=0.2,     # Décalage horizontal jusqu'à 20% de la largeur de l'image
    height_shift_range=0.2,    # Décalage vertical jusqu'à 20% de la hauteur de l'image
    shear_range=0.2,           # Transformation affine (cisaillement) de 20%
    zoom_range=0.2,            # Zoom aléatoire jusqu'à 20%
    horizontal_flip=True,      # Flip horizontal aléatoire (effet miroir)
    fill_mode='nearest'        # Remplit les pixels vides avec les valeurs les plus proches
)


img_width = 48
img_height = 48
batch_size  = 128
epochs      =  50
fit_verbosity = 1
num_classes = 7
learning_rate = 0.0001


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer="glorot_uniform", padding='same', input_shape=(img_width, img_height, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.02))) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.02))) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.02)))  
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.02))) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model


model = cnn_model()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


import datetime
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    verbose=fit_verbosity,
    callbacks=[early_stop, reduce_lr]
)


model.save('final_emotion_model.keras')
print("Le modèle a été enregistré avec succès")


losses=pd.DataFrame(history.history)


losses[['loss','val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='--', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Accuracy')
plt.legend()
plt.grid(True)
plt.show()



# Évaluer le modèle
print("shape X_test:", X_test.shape)
print("shape y_test:", y_test.shape)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Accuracy sur le test set : {test_acc:.4f}")


def visualize_misclassified_images(X_test, y_test, model, emotions, num_samples=15, max_columns=5):
    """Affiche des images mal prédites par le modèle sous forme de grille."""
    
    y_pred = model.predict(X_test)    
    misclassified_indices = np.where(np.argmax(y_pred, axis=1) != np.argmax(y_test, axis=1))[0]    
    num_samples = min(num_samples, len(misclassified_indices))
    
    num_rows = (num_samples + max_columns - 1) // max_columns
    
    plt.figure(figsize=(max_columns * 4, num_rows * 4))
    
    for i in range(num_samples):
        index = misclassified_indices[i]
        
        # Image mal prédite
        img = X_test[index].reshape(48, 48)  
        true_label = emotions[np.argmax(y_test[index])]  
        predicted_label = emotions[np.argmax(y_pred[index])]  
        
        plt.subplot(num_rows, max_columns, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        plt.title(f"True: {true_label}", fontsize=18, color='blue', pad=20)  
        plt.text(24, -8, f"Pred: {predicted_label}", fontsize=16, color='red', ha='center')  
    
    plt.tight_layout(pad=3.0)  

visualize_misclassified_images(X_test, y_test, model, emotions, num_samples=10, max_columns=5)



y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
df = pd.DataFrame(y_pred_classes, columns=['3'])  
df.to_csv("submission.csv", index=False, header=True)
print("✅ Fichier submission.csv enregistré avec succès !")
df.head()


