# copy the weights and configurations for the pre-trained models 
!mkdir ~/.keras
!mkdir ~/.keras/models7
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


thetrainpath= glob('/kaggle/input/diabetic-retinopathy-train-unzipped/train/*.jpeg')


file_lbl="/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
df_train=pd.read_csv(file_lbl,sep=',')
df_train


# Dossier contenant les images d'entraînement
train_images_path = '/kaggle/input/diabetic-retinopathy-train-unzipped/train/'

# Obtenir la liste des fichiers image avec extension .jpeg
image_files = glob(os.path.join(train_images_path, '*.jpeg'))

# Chargement du fichier CSV contenant les étiquettes
labels_path = os.path.join('/kaggle/input/diabetic-retinopathy-train-unzipped', file_lbl)
labels_df = pd.read_csv(labels_path)

# Ajout de l'identifiant du patient et du chemin complet de l'image
labels_df['PatientId'] = labels_df['image'].apply(lambda name: name.split('_')[0])
labels_df['image_path'] = labels_df['image'].apply(lambda name: os.path.join(train_images_path, f'{name}.jpeg'))

# Vérification de l'existence du fichier image
labels_df['is_present'] = labels_df['image_path'].apply(os.path.exists)
print(f"{labels_df['is_present'].sum()} images disponibles sur {labels_df.shape[0]} attendues.")

# Encodage de l'œil (gauche = 1, droite = 0)
labels_df['eye_side'] = labels_df['image'].apply(lambda name: 1 if name.endswith('left') else 0)



from tensorflow.keras.utils import to_categorical

# Conversion de la colonne 'level' en vecteurs one-hot
labels_df['level_encoded'] = labels_df['level'].apply(
    lambda label: to_categorical(label, num_classes=labels_df['level'].max() + 1)
)

# Suppression des lignes contenant des valeurs manquantes
labels_df.dropna(inplace=True)

# Filtrage pour ne garder que les images disponibles
labels_df = labels_df[labels_df['is_present']]

# Affichage de quelques exemples
labels_df.sample(3)



import seaborn as sns
import matplotlib.pyplot as plt

# Visualiser la distribution des variables 'level' et 'eye' avec un pairplot
sns.pairplot(labels_df[['level', 'eye_side']], hue='eye_side', palette='coolwarm')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot pour la distribution des niveaux en fonction de l'œil
sns.boxplot(x='eye_side', y='level', data=labels_df, palette='coolwarm')
plt.title('Distribution des niveaux en fonction de l\'œil')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Violinplot pour la distribution des niveaux selon l'œil
sns.violinplot(x='eye_side', y='level', data=labels_df, palette='muted')
plt.title('Répartition des niveaux en fonction de l\'œil')
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Préparation des données à partir du jeu de données 'labels_df'
# Extraction des identifiants patients uniques avec leurs niveaux de gravité
unique_patients_df = labels_df[['PatientId', 'level']].drop_duplicates()

# Séparation stratifiée des patients pour garantir une répartition équilibrée des niveaux
train_patients, val_patients = train_test_split(
    unique_patients_df['PatientId'],
    test_size=0.25,
    stratify=unique_patients_df['level'],
    random_state=2018
)

# Filtrage des données complètes selon les identifiants sélectionnés
train_df = labels_df[labels_df['PatientId'].isin(train_patients)]
val_df = labels_df[labels_df['PatientId'].isin(val_patients)]

print(f"Nombre d'observations - Entraînement: {train_df.shape[0]}, Validation: {val_df.shape[0]}")

# Équilibrage de l'ensemble d'entraînement par sur-échantillonnage (oversampling)
# Calcul du nombre cible d'échantillons pour chaque groupe (level, eye)
target_count = train_df.groupby(['level', 'eye_side']).size().max()

# Application de la stratégie de sur-échantillonnage
balanced_train_df = (
    train_df.groupby(['level', 'eye_side'], group_keys=False)
    .apply(lambda group: resample(group, replace=True, n_samples=target_count, random_state=2018))
    .reset_index(drop=True)
)

print(f"Taille après équilibrage de l'ensemble d'entraînement : {balanced_train_df.shape[0]}")

# Visualisation de la distribution des classes dans l'ensemble équilibré
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
balanced_train_df['level'].value_counts().sort_index().plot(kind='bar', ax=axes[0], title="Répartition par niveau")
balanced_train_df['eye_side'].value_counts().sort_index().plot(kind='bar', ax=axes[1], title="Répartition par œil")
plt.tight_layout()
plt.show()






import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical


# 1 Initialisation des ImageDataGenerators (avec et sans augmentation)
train_datagen = ImageDataGenerator(
    rescale           = 1. / 255,
    rotation_range    = 40,
    width_shift_range = 0.20,
    height_shift_range= 0.20,
    shear_range       = 0.20,
    zoom_range        = 0.20,
    horizontal_flip   = True,
    fill_mode         = "nearest"
)

valid_datagen = ImageDataGenerator(rescale = 1. / 255)

# 2 Fonction utilitaire pour générer les batches à partir d'un DataFrame
def generate_from_dataframe(df,
                             datagen,
                             batch_size  : int = 32,
                             target_size : tuple = (224, 224),
                             num_classes : int = None,
                             shuffle     : bool = True):
    """
    Génère des batches (images, labels) en appliquant des transformations 
    via ImageDataGenerator.

    Paramètres
    ----------
    df          : DataFrame avec au minimum les colonnes 'image_path' et 'level'
    datagen     : un ImageDataGenerator configuré
    batch_size  : nombre d'images par batch
    target_size : taille (H, W) des images après redimensionnement
    num_classes : si précisé, les labels seront encodés en one-hot
    shuffle     : réorganiser le DataFrame au début de chaque époque
    """
    while True:
        # Réorganiser les lignes du DataFrame si nécessaire
        if shuffle:
            df = df.sample(frac=1, random_state=None).reset_index(drop=True)

        # Parcourir les données par batchs
        for start in range(0, len(df), batch_size):
            end   = start + batch_size
            batch = df.iloc[start:end]

            # ----- Chargement des images -----
            imgs, lbls = [], []
            for _, row in batch.iterrows():
                # Chargement et redimensionnement de l'image
                img = load_img(row['image_path'], target_size=target_size)   # <-- colonne adaptée
                imgs.append(img_to_array(img))
                lbls.append(row['level'])                                    # <-- toujours 'level'

            # Conversion en tableaux NumPy
            X = np.array(imgs, dtype=np.float32)
            y = np.array(lbls, dtype=np.int64)

            # Encodage des labels en one-hot si nécessaire
            if num_classes is not None:
                y = to_categorical(y, num_classes=num_classes)

            # ImageDataGenerator.flow est un générateur infini ; on "déballe" un batch à la fois
            for aug_X, aug_y in datagen.flow(X, y, batch_size=batch_size, shuffle=False):
                yield aug_X, aug_y
                break   # Sortir après un seul batch pour que la boucle externe contrôle l'ordre

# 3 Détection automatique du nombre de classes et création des générateurs
# Identification du nombre de classes à partir du DataFrame d'entraînement
num_classes = balanced_train_df['level'].nunique()

# Création des générateurs d'entraînement et de validation
train_generator = generate_from_dataframe(
    balanced_train_df,
    train_datagen,
    batch_size   = 32,
    target_size  = (224, 224),
    num_classes  = num_classes
)

valid_generator = generate_from_dataframe(
    val_df,
    valid_datagen,
    batch_size   = 32,
    target_size  = (224, 224),
    num_classes  = num_classes,
    shuffle      = False  # Pas de réorganisation des données de validation entre les époques
)

# 4 Vérification rapide : récupérer un batch
X_b, y_b = next(train_generator)
print(f"Shape des images : {X_b.shape}")   # -> (32, 224, 224, 3)
print(f"Shape des labels : {y_b.shape}")   # -> (32, 5) si num_classes == 5



import numpy as np
import matplotlib.pyplot as plt

def visaugmtdimgs(generator, batch_size=8):
    """
    Visualise un batch d'images augmentées avec leurs labels.

    Paramètres :
    - generator : générateur d'images augmentées
    - batch_size : nombre d'images à afficher
    """
    # Obtenir un batch d'images et de labels
    t_x, t_y = next(generator)

    # Vérifier la taille du batch pour s'assurer qu'il est assez grand
    actual_batch_size = min(batch_size, t_x.shape[0])

    # Créer une figure avec des sous-graphiques
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i in range(actual_batch_size):
        # Dé-normaliser les images pour la visualisation
        img = np.clip(t_x[i] * 255, 0, 255).astype(np.uint8)

        # Identification du niveau et de l'œil
        level = np.argmax(t_y[i])  # Récupération du niveau de gravité
        eye_side = "left" if level % 2 == 0 else "right"  # Adaptation simplifiée

        # Affichage de l'image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Niveau : {level}")

    # Supprimer les sous-graphiques inutilisés s'il y en a
    for j in range(actual_batch_size, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation : visualiser les images augmentées du générateur d'entraînement
visaugmtdimgs(train_generator, batch_size=8)



import numpy as np
import matplotlib.pyplot as plt

def visaugmtdimgs(generator, batch_size=10):
    """
    Visualise un nombre précis d'images augmentées.
    
    Paramètres :
    - generator : générateur d'images augmentées
    - batch_size : nombre d'images à afficher (exactement)
    """
    # Initialisation des listes pour accumuler les images et labels
    images, labels = [], []

    # Accumuler les images jusqu'à obtenir `batch_size` images
    while len(images) < batch_size:
        # Obtenir un batch du générateur
        t_x, t_y = next(generator)
        images.extend(t_x)
        labels.extend(t_y)

    # Sélectionner uniquement les `batch_size` premières images
    images = np.array(images[:batch_size], dtype=np.float32)
    labels = np.array(labels[:batch_size])

    # Créer la figure avec des sous-graphiques
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(batch_size):
        # Dé-normaliser les images pour la visualisation
        img = np.clip(images[i] * 255, 0, 255).astype(np.uint8)
        label = np.argmax(labels[i])

        # Affichage de l'image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Niveau : {label}")

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation : visualiser exactement 10 images
visaugmtdimgs(train_generator, batch_size=10)



import numpy as np
import matplotlib.pyplot as plt

def plot_valid_img(validation_gen, batch_size=8):
    """
    Visualise un batch d'images de validation avec leurs informations associées.

    Paramètres :
    - valid_gen : générateur d'images de validation
    - batch_size : nombre d'images à afficher
    """
    # Récupérer un batch complet
    v_x, v_y = next(validation_gen)

    # Ajuster le batch_size si le batch est plus petit
    actual_batch_size = min(batch_size, len(v_x))

    # Configuration de la figure avec des sous-graphiques
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(actual_batch_size):
        # Image
        img = np.clip(v_x[i] * 255, 0, 255).astype(np.uint8)

        # Label (one-hot -> index)
        level = np.argmax(v_y[i])

        # Affichage
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Niveau : {level}")

    # Cacher les axes inutilisés si le batch est incomplet
    for j in range(actual_batch_size, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation : Visualiser un batch de validation
plot_valid_img(valid_generator, batch_size=8)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax') 
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=len(balanced_train_df,) // 32,  
    epochs=2,
    validation_data=valid_generator,
    validation_steps=len(val_df) // 32
)


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import (
    Input, Conv2D, Dropout, multiply, GlobalAveragePooling2D,
    Lambda, Dense, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import TopKCategoricalAccuracy


# 1. Initialisation des hyperparamètres
input_shape = (224, 224, 3)
num_classes = balanced_train_df['level'].nunique()
dropout_rate = 0.5
attention_filters = [64, 16, 8]

# 2. Définition de l'Input Layer
image_input = Input(shape=input_shape, name="Input_Image")

# 3. Modèle pré-entraîné : InceptionV3 (couche de base)
base_model = InceptionV3(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Extraction des caractéristiques de la base pré-entraînée
base_features = base_model(image_input)
base_features = BatchNormalization(name="Base_BatchNorm")(base_features)

# 4. Mécanisme d'Attention
attention_layer = base_features

for filters in attention_filters:
    attention_layer = Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(dropout_rate)(attention_layer))

# Couche finale d'attention avec sigmoid pour la mise à l'échelle des activations
attention_layer = Conv2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid', name="Attention_Mask")(attention_layer)

# Redimensionnement du masque pour qu'il corresponde aux canaux de `base_features`
up_conv = Conv2D(
    filters=base_features.shape[-1],
    kernel_size=(1, 1),
    padding='same',
    activation='linear',
    use_bias=False,
    name="Expand_Mask"
)

# Création du poids constant (1) pour la couche `up_conv`
initial_weights = np.ones((1, 1, 1, base_features.shape[-1]))

# Assigner manuellement les poids à `up_conv`
up_conv.build(input_shape=attention_layer.shape)
up_conv.set_weights([initial_weights])
up_conv.trainable = False

# Appliquer la couche de redimensionnement
attention_layer = up_conv(attention_layer)

# 5. Application du masque d'attention
masked_features = multiply([attention_layer, base_features], name="Masked_Features")

# 6. Global Average Pooling & Rescaling
gap_features = GlobalAveragePooling2D(name="GAP_Features")(masked_features)
gap_mask = GlobalAveragePooling2D(name="GAP_Mask")(attention_layer)

# Recalage pour compenser les pixels masqués
rescaled_gap = Lambda(lambda x: x[0] / (x[1] + 1e-6), name="RescaleGAP")([gap_features, gap_mask])

# 7. Couches Fully Connected
fc_layer = Dropout(0.25, name="Dropout1")(rescaled_gap)
fc_layer = Dense(128, activation='relu', name="FC_Layer")(fc_layer)
fc_layer = Dropout(0.25, name="Dropout2")(fc_layer)

# 8. Couche de sortie
output_layer = Dense(num_classes, activation='softmax', name="Output_Layer")(fc_layer)

# 9. Modèle final
retina_model = Model(inputs=image_input, outputs=output_layer, name="Retina_Attention_Model")

# 10. Compilation du modèle
def top_2_accuracy(y_true, y_pred):
    return TopKCategoricalAccuracy(k=2)(y_true, y_pred)

retina_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', top_2_accuracy]
)

# Résumé du modèle
retina_model.summary()



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Correction du chemin de sauvegarde des poids
weight_path = "retina_weights.best.weights.h5"

checkpoint = ModelCheckpoint(
    filepath=weight_path, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='min'
)

# Réduction du Learning Rate
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.8, 
    patience=3, 
    verbose=1, 
    mode='min', 
    min_delta=0.0001,  # `min_delta` remplace `epsilon`
    cooldown=5, 
    min_lr=1e-5
)

# Arrêt anticipé
early = EarlyStopping(
    monitor="val_loss", 
    mode="min", 
    patience=6,
    verbose=1,
    restore_best_weights=True
)

# Liste des callbacks
callbacks_list = [checkpoint, early, reduceLROnPlat]
print(callbacks_list)



!rm -rf ~/.keras # clean up before starting training



import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Définition des variables globales
# Créer le modèle une seule fois
inputs = Input(shape=(4,))
outputs = Dense(1, activation='linear')(inputs)
simple_model = Model(inputs, outputs)

# Compilation du modèle (une seule fois)
simple_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# Créer une variable TensorFlow unique (si nécessaire)
global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')


# 2. Fonction d'entraînement décorée par @tf.function
@tf.function
def train_step(x, y):
    # Incrémenter global_step en dehors du contexte de GradientTape
    global_step.assign_add(1)

    with tf.GradientTape() as tape:
        predictions = simple_model(x)
        loss = tf.reduce_mean(tf.square(predictions - y))
    
    # Calcul des gradients
    gradients = tape.gradient(loss, simple_model.trainable_variables)
    
    # Application des gradients
    simple_model.optimizer.apply_gradients(zip(gradients, simple_model.trainable_variables))
    
    return loss

# 3. Données d'entraînement
x_train = tf.random.normal((10, 4))
y_train = tf.random.normal((10, 1))

# 4. Entraînement
epochs = 50

for epoch in range(epochs):
    loss_value = train_step(x_train, y_train)
    print(f"Époque {epoch + 1}, Perte : {loss_value.numpy()}")

print(f"Entraînement terminé. Nombre d'étapes globales : {global_step.numpy()}")



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------
# 1. Création du modèle avant toute opération d'entraînement
# ------------------------------------------------------------------
# Vérifier si le modèle existe déjà pour éviter une recréation multiple
if 'retina_model' not in globals():
    # Recréer le modèle une seule fois
    retina_model = Model(inputs=image_input, outputs=output_layer, name="Retina_Attention_Model")

# ------------------------------------------------------------------
# 2. Compilation du modèle
# ------------------------------------------------------------------
retina_model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', top_2_accuracy]
)

# ------------------------------------------------------------------
# 3. Paramètres d'enregistrement des poids
# ------------------------------------------------------------------
weight_path = "retina_attention_model_best.weights.h5"

checkpoint = ModelCheckpoint(
    filepath=weight_path, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='min'
)

# ------------------------------------------------------------------
# 4. Callback de réduction du taux d'apprentissage
# ------------------------------------------------------------------
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,  
    patience=3, 
    verbose=1, 
    mode='min', 
    min_delta=1e-4, 
    cooldown=2, 
    min_lr=1e-6
)

# ------------------------------------------------------------------
# 5. Arrêt anticipé pour éviter le surapprentissage
# ------------------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=10, 
    verbose=1, 
    restore_best_weights=True
)

# ------------------------------------------------------------------
# 6. Logger CSV pour suivre les performances
# ------------------------------------------------------------------
csv_logger = CSVLogger('training_log.csv', append=True)

# Liste des callbacks
callbacks_list = [checkpoint, early_stop, reduce_lr, csv_logger]




# Test des générateurs
x_batch, y_batch = next(train_generator)
print(f"Shape des images d'entraînement : {x_batch.shape}")
print(f"Shape des labels d'entraînement : {y_batch.shape}")

x_val_batch, y_val_batch = next(valid_generator)
print(f"Shape des images de validation : {x_val_batch.shape}")
print(f"Shape des labels de validation : {y_val_batch.shape}")



# Vérifier que le modèle est bien défini
retina_model.summary()

# Vérifier si le modèle est compilé
if retina_model.optimizer is None:
    print("Le modèle n'est pas compilé. Re-compilation...")
    retina_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', top_2_accuracy]
    )
else:
    print("Le modèle est déjà compilé.")




from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Créer le modèle une seule fois
inputs = Input(shape=(4,))
outputs = Dense(1, activation='linear')(inputs)
simple_model = Model(inputs, outputs)

# Compilation du modèle
simple_model.compile(optimizer='adam', loss='mse')

# Fonction de mise à jour du modèle
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = simple_model(x)
        loss = tf.reduce_mean(tf.square(predictions - y))
    
    # Calcul et application des gradients
    gradients = tape.gradient(loss, simple_model.trainable_variables)
    simple_model.optimizer.apply_gradients(zip(gradients, simple_model.trainable_variables))
    
    return loss

# Données d'exemple
x_train = tf.random.normal((10, 4))
y_train = tf.random.normal((10, 1))

# Exécution de la boucle d'entraînement
for epoch in range(50):
    loss_value = train_step(x_train, y_train)
    print(f"Époque {epoch+1}, Perte : {loss_value.numpy()}")



paths_train= glob('/kaggle/input/diabetic-retinopathy-train-unzipped/train/*.jpeg')
image=cv2.imread(paths_train[0])
plt.imshow(image)


paths_test = glob('/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg')
image=cv2.imread(paths_test[0])
plt.imshow(image)


file_sub="/kaggle/input/diabetic-retinopathy-detection/sampleSubmission.csv.zip"
df_submission=pd.read_csv(file_sub,sep=',')
df_submission.loc[0, 'level']=1
df_submission


df_submission.to_csv('submission.csv', index=False)

