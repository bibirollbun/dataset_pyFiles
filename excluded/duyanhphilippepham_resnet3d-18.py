import os
import ast
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Tuple
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
import pydicom
from scipy.ndimage import zoom
import glob
from collections import deque


preprocessing = True
SERIES_ROOT_TRAIN = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
TRAIN_CSV         = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOCALIZER_CSV     = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"

ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
    'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present',
]
TARGET_COL = 'Aneurysm Present'

MODEL_WEIGHT = "/kaggle/working/model/RSNA_Intracranial_Aneurysm_Detection"

DEBUG = False
TRAIN = True

TARGET_SIZE = (32, 384, 384) 


train_df = pd.read_csv(TRAIN_CSV)


class CustomResNet3D:
    def __init__(self, target_size, label_cols):
        self.input_shape = (*target_size, 1)
        self.num_classes = len(label_cols) - 1
        self.model = self._build_model()

    def _conv3d_block(self, x, filters, kernel_size=3, stride=1, dropout_rate=0.3):
        shortcut = x
        x = layers.Conv3D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv3D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv3D(32, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

        x = self._conv3d_block(x, 32)
        x = self._conv3d_block(x, 64, stride=2)
        x = self._conv3d_block(x, 128, stride=2)
        x = self._conv3d_block(x, 256, stride=2)

        x = layers.GlobalAveragePooling3D()(x)

        class_output = layers.Dense(self.num_classes, activation='softmax', name='class_output')(x)
        label_output = layers.Dense(1, activation='sigmoid', name='label_output')(x)

        return models.Model(inputs=inputs, outputs=[class_output, label_output])

    def compile(self, optimizer='adam'):
        self.model.compile(
            optimizer=optimizer,
            loss={
                'class_output': 'categorical_crossentropy',
                'label_output': 'binary_crossentropy',
            },
            metrics={
                'class_output': 'accuracy',
                'label_output': 'accuracy',
            }
        )

    def summary(self):
        return self.model.summary()

    def get_model(self):
        return self.model
        
    # Preprocessing
    def filtering(self, image): #TODO
        image = image.astype(np.float32)
        mean, std = np.mean(image), np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean
        return image

    def resize_and_pad(self, image, target_shape): #TODO
        h, w = image.shape
        th, tw = target_shape
        new_h = min(h, th)
        new_w = min(w, tw)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_h = th - new_h
        pad_w = tw - new_w
        padded = np.pad(resized, ((0, pad_h), (0, pad_w)), 
                        mode='constant', constant_values=0)
        return padded

    def resize_image(self, img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize an image to match the target shape using interpolation.
    
        Parameters:
        - img: Input image as a NumPy array.
        - target_shape: Tuple (height, width) for the desired output size.
    
        Returns:
        - Resized image as a NumPy array.
        """
        target_height, target_width = target_shape
    
        # Choose interpolation method based on scaling direction
        if target_height > img.shape[0] or target_width > img.shape[1]:
            interpolation = cv2.INTER_CUBIC  # better for upscaling
        else:
            interpolation = cv2.INTER_AREA   # better for downscaling
    
        resized = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        return resized
    
    def preprocessing(self, img, target_shape): #TODO
        # print(target_shape)
        image = self.filtering(img)
        image = self.resize_image(image, target_shape)
        # print(target_shape)
        return np.array(image).astype(np.float32)  # float16 possible
    
    # Processing
    ## loading weight 
    def load_latest_weights(self, save_dir=MODEL_WEIGHT):
        ckpts = sorted(glob.glob(os.path.join(save_dir, "weights_*.h5")))
        if not ckpts:
            print("Aucun checkpoint trouvé.")
            return False
        latest = ckpts[-1]
        self.model.load_weights(latest)
        print(f"Poids chargés depuis : {latest}")
        return True
        
    def _save_weights_fifo(self, save_dir=MODEL_WEIGHT, max_keep=5):
        os.makedirs(save_dir, exist_ok=True)
        # Nom du fichier basé sur timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"weights_{timestamp}.h5")
        self.model.save_weights(filepath)
    
        # FIFO : garde seulement les derniers fichiers
        ckpts = sorted(glob.glob(os.path.join(save_dir, "weights_*.h5")))
        if len(ckpts) > max_keep:
            for old_ckpt in ckpts[:-max_keep]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass
    ## loading data
    def load_and_process_dicom_series(self, series_path, target_shape=None):

        """
        Charge un volume DICOM complet, applique le prétraitement slice par slice,
        et retourne un tenseur (1, D, H, W, 1) sans altérer la profondeur.
    
        Args:
            series_path (str): Dossier contenant les fichiers DICOM
            target_shape (tuple): Taille (H, W) des slices après resize/pad
        """
        if target_shape is None:
            target_shape = (self.input_shape[0:3]) # (H, W) à partir de input_shape = (D,H,W,1)
        dicom_files = sorted(os.listdir(series_path))
        
        if len(dicom_files) == 1:
            dcm = pydicom.dcmread(os.path.join(series_path, dicom_files[0]))
            volume = dcm.pixel_array  # (D, H, W)
            processed_slices = [self.preprocessing(slice_, target_shape[1:3]) for slice_ in volume]
        else:
            slices = [pydicom.dcmread(os.path.join(series_path, f)).pixel_array
                      for f in dicom_files]
            processed_slices = [self.preprocessing(slice_, target_shape[1:3]) for slice_ in slices]
        # print(np.shape(processed_slices))
        volume = np.array(processed_slices, dtype=np.float32)
        # print(volume.shape, target_shape, flush = True)
        factors = [t / s for s, t in zip(volume.shape, target_shape)]
        resized_vol = zoom(volume, zoom=factors, order=1)

    
        # --- Ajout channel + batch ---
        volume = np.expand_dims(resized_vol, axis=-1)   # (D, H, W, 1)
        volume = np.expand_dims(volume, axis=0)    # (1, D, H, W, 1)
        return tf.convert_to_tensor(volume, dtype=tf.float32)
    
    def train_on_batch_samples(self, sample_paths, labels_class, labels_binary, 
                               batch_size=4, epochs=10, shuffle=True):
        """
        Entraîne le modèle sur un batch de volumes DICOM.
    
        Args:
            sample_paths (list[str]): Liste des chemins vers les dossiers séries DICOM.
            labels_class (list[int]): Labels pour la classification multi-classes.
            labels_binary (list[int/float]): Labels binaires.
            batch_size (int): Taille de batch.
            epochs (int): Nombre d'époques d'entraînement.
            shuffle (bool): Mélanger les données à chaque époque.
        """
        import numpy as np
        import tensorflow as tf
    
        # --- Préparation des données ---
        X = []
        Y_class = []
        Y_label = []
    
        for path, lc, lb in zip(sample_paths, labels_class, labels_binary):
            volume = self.load_and_process_dicom_series(path)  # (1, D, H, W, 1)
            X.append(volume.numpy()[0])       # on enlève la dim batch ajoutée par load
            Y_class.append(lc)
            Y_label.append(lb)
    
        # Conversion en tenseurs
        X = np.array(X, dtype=np.float32)  # (N, D, H, W, 1)
        Y_class = tf.one_hot(Y_class, depth=self.num_classes)
        Y_label = np.array(Y_label, dtype=np.float32)
    
        # --- Dataset TensorFlow ---
        dataset = tf.data.Dataset.from_tensor_slices((
            X,
            {
                'class_output': Y_class,
                'label_output': Y_label,
            }
        ))
    
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(sample_paths))
    
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(MODEL_WEIGHT, "best_model.h5"),
                        save_best_only=True,
        													 
                        save_weights_only=True
                    )
        ]
        self.model.fit(dataset, epochs=epochs, callbacks=callbacks)

    # Prédict
    def predict_from_series(self, series_path, target_shape=None):
        """
        Charge un volume DICOM complet, le prétraite et renvoie les prédictions du modèle.
        
        Args:
            series_path (str): chemin du dossier contenant la série DICOM.
            target_shape (tuple): taille (H, W) des slices pour le prétraitement.
        
        Returns:
            dict: Prédictions pour 'class_output', 'label_output'
        """
        if target_shape is None:
            target_shape = (self.input_shape[0:3]) # (H, W) à partir de input_shape = (D,H,W,1)
        # Chargement et prétraitement du volume
        volume = self.load_and_process_dicom_series(series_path, target_shape)
        
        # Prédiction
        preds = self.model.predict(volume)
        
        # Renvoyer sous forme lisible
        return {
            'class_output': preds[0][0],  # vecteur de probabilités
            'label_output': float(preds[1][0]),  # probabilité
        }
    def decision_predict(self, series_path, target_shape=None):
        """
        Appelle predict_from_series, puis ajoute une décision binaire
        pour class_output et label_output.
        """
        # Appel de la fonction d'origine
        preds = self.predict_from_series(series_path, target_shape)
    
        # Copie pour ne pas écraser l’original
        result = preds.copy()
    
        # Arrondis pour obtenir 0 ou 1
        result['SeriesInstanceUID'] =  file_name = os.path.basename(series_path)
        result['class_output_abs'] = np.round(result['class_output']).astype(int)
        result['label_output_abs'] = int(np.round(result['label_output']))
        return result


resnet3d = CustomResNet3D(target_size=TARGET_SIZE, label_cols=LABEL_COLS)
resnet3d.compile()
resnet3d.load_latest_weights()

model = resnet3d.get_model()

if DEBUG:
    resnet3d.summary()
    
    # Optionnel : visualisation graphique
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='resnet3d.png', show_shapes=True, expand_nested=True)


if DEBUG:
    import os
    os.makedirs(MODEL_WEIGHT, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resnet3d.get_model().save_weights(os.path.join(MODEL_WEIGHT,f"weights_{timestamp}.weights.h5"))


# Lecture des fichiers
train_df = pd.read_csv(TRAIN_CSV)

# Harmonisation du type de clé si besoin
train_df["SeriesInstanceUID"] = train_df["SeriesInstanceUID"].astype(str)

# Remplacer tous les NaN par 0
train_df = train_df.fillna(0)

# Suppression des colonnes inutiles
cols_to_drop = ["PatientSex", "PatientAge"]
train_df = train_df.drop(columns=cols_to_drop)
if DEBUG:
    print(train_df.columns)


def extract_labels_and_paths(df, label_cols, dicom_dir):
    """
    Extrait labels_class, labels_binary et mes_paths
    à partir d'un DataFrame fusionné.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes labels, 'Aneurysm Present', et 'SeriesInstanceUID'.
        label_cols (list): Liste complète LABEL_COLS (incluant 'Aneurysm Present' en dernier).
        dicom_dir (str): Chemin racine contenant les fichiers DICOM.

    Returns:
        tuple: (labels_class, labels_binary, mes_paths)
    """
    # --- 1) labels_class ---
    labels_class = []
    for _, row in df.iterrows():
        sub_labels = row[label_cols[:-1]]
        if sub_labels.max() == 1:
            idx = sub_labels[sub_labels == 1].index[0]
            class_index = label_cols[:-1].index(idx)
        else:
            class_index = -1  # ou autre valeur sentinelle
        labels_class.append(class_index)

    # --- 2) labels_binary ---
    labels_binary = df['Aneurysm Present'].astype(int).tolist()
    
    # --- 3) mes_paths ---
    mes_paths = [
        os.path.join(dicom_dir, str(uid))
        for uid in df["SeriesInstanceUID"]
    ]

    return labels_class, labels_binary, mes_paths



labels_class, labels_binary, mes_paths = extract_labels_and_paths(
    train_df,
    LABEL_COLS,
    SERIES_ROOT_TRAIN
)


if DEBUG:
    print(labels_class[:5])
    print(labels_binary[:5])
    print(mes_paths[:5])


if DEBUG:
    init = 0
    test_size = 10
    resnet3d.train_on_batch_samples(mes_paths[init:init+test_size], labels_class[init:init+test_size], 
                                    labels_binary[init:init+test_size], 
                                    batch_size=4, epochs=10, shuffle=True)
    


if TRAIN:
    resnet3d.train_on_batch_samples(mes_paths, labels_class, 
                                        labels_binary, 
                                        batch_size=64, epochs=20, shuffle=True)


%%time
import os
import shutil
from collections import defaultdict

import pandas as pd
import polars as pl
import pydicom

import kaggle_evaluation.rsna_inference_server


resnet3d = CustomResNet3D(target_size=TARGET_SIZE, label_cols=LABEL_COLS)
resnet3d.compile()
resnet3d.load_latest_weights()


def prediction_to_dataframe(pred_result, label_cols):
    """
    Transforme un dictionnaire de prédiction en DataFrame d'une seule ligne.

    Parameters
    ----------
    pred_result : dict
        Doit contenir :
        - "SeriesInstanceUID" (str)
        - "class_output_abs" (liste ou array de numériques)
        - "label_output_abs" (numérique)
    label_cols : list[str]
        Liste des noms de colonnes pour class_output_abs.

    Returns
    -------
    pd.DataFrame
        DataFrame avec SeriesInstanceUID + colonnes labels + Aneurysm Present.
    """
    row_data = [pred_result["SeriesInstanceUID"]] \
             + list(map(float, pred_result["class_output_abs"])) \
             + [float(pred_result["label_output_abs"])]
    columns = ["SeriesInstanceUID"] + label_cols
    return pd.DataFrame([row_data], columns=columns)


def predict(series_path):
    """Transforme la sortie de decision_predict en DataFrame formaté."""
    pred_result = resnet3d.decision_predict(series_path)
    return prediction_to_dataframe(pred_result, LABEL_COLS)
    
if DEBUG:
    pred = resnet3d.decision_predict(mes_paths[0])
    print(pred)
    df = prediction_to_dataframe(pred, LABEL_COLS)
    print(predict(mes_paths[0]))


import shutil
import os

# Exemple : adapter vers ton répertoire réel
share_dir = "/kaggle/working"

# Supprime tout le contenu du répertoire (⚠️ irréversible)
for item in os.listdir(share_dir):
    item_path = os.path.join(share_dir, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
    else:
        os.remove(item_path)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

