# Installation des bibliothèques nécessaires
!pip install librosa soundfile tensorflow scikit-learn pandas numpy matplotlib seaborn tqdm optuna


# Importation des bibliothèques
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm.notebook import tqdm
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Définir les styles de visualisation
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Vérifier la disponibilité du GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))


# Définir les chemins des données
DATA_DIR = '../input/birdclef-2025/'
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train_audio')
TEST_AUDIO_DIR = os.path.join(DATA_DIR, 'test_soundscapes')
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'sample_submission.csv')

# Charger le fichier d'exemple de soumission pour comprendre la structure des prédictions attendues
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
print(f"Forme du fichier de soumission: {sample_submission.shape}")
print(f"Colonnes du fichier de soumission: {sample_submission.columns[:10]}...")
sample_submission.head()


# Analyser le nombre d'espèces à prédire
species_columns = sample_submission.columns[1:]
num_species = len(species_columns)
print(f"Nombre d'espèces à prédire: {num_species}")

# Afficher quelques exemples d'espèces
print(f"Exemples d'espèces: {list(species_columns[:10])}...")


# Explorer les fichiers audio d'entraînement
def explore_audio_directory(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ogg') or file.endswith('.mp3') or file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

train_audio_files = explore_audio_directory(TRAIN_AUDIO_DIR)
print(f"Nombre de fichiers audio d'entraînement: {len(train_audio_files)}")
print(f"Exemples de fichiers audio: {[os.path.basename(f) for f in train_audio_files[:5]]}...")


# Analyser la durée des fichiers audio d'entraînement (échantillon)
def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=10)  # Charger seulement les 10 premières secondes pour être rapide
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None

# Analyser un échantillon de fichiers audio
sample_size = min(100, len(train_audio_files))
sample_files = np.random.choice(train_audio_files, sample_size, replace=False)
durations = [get_audio_duration(f) for f in tqdm(sample_files)]
durations = [d for d in durations if d is not None]

# Visualiser la distribution des durées
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20)
plt.xlabel('Durée (secondes)')
plt.ylabel('Nombre de fichiers')
plt.title('Distribution des durées des fichiers audio (échantillon)')
plt.grid(True)
plt.show()

print(f"Durée moyenne: {np.mean(durations):.2f} secondes")
print(f"Durée médiane: {np.median(durations):.2f} secondes")
print(f"Durée minimale: {np.min(durations):.2f} secondes")
print(f"Durée maximale: {np.max(durations):.2f} secondes")


# Visualiser un spectrogramme d'exemple
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(12, 8))
    
    # Forme d'onde
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Forme d\'onde')
    
    # Spectrogramme
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    
    # Spectrogramme Mel
    plt.subplot(3, 1, 3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme Mel')
    
    plt.tight_layout()
    plt.show()
    
    return y, sr

# Sélectionner un fichier audio aléatoire
example_file = np.random.choice(train_audio_files)
print(f"Fichier d'exemple: {os.path.basename(example_file)}")
y, sr = plot_spectrogram(example_file)


class AudioPreprocessor:
    """Classe pour prétraiter les données audio"""
    
    def __init__(self, config=None):
        """Initialise le préprocesseur avec la configuration spécifiée"""
        # Configuration par défaut
        self.config = {
            'sample_rate': 32000,      # Taux d'échantillonnage cible
            'n_mels': 128,             # Nombre de bandes mel
            'n_fft': 1024,             # Taille de la FFT
            'hop_length': 512,         # Longueur du saut pour la STFT
            'segment_duration': 5,     # Durée des segments en secondes
            'overlap': 2.5,            # Chevauchement entre segments en secondes
            'min_duration': 2,         # Durée minimale pour un segment valide
            'normalize': True,         # Normaliser l'audio
            'augment': True            # Appliquer l'augmentation de données
        }
        
        # Mettre à jour la configuration si fournie
        if config:
            self.config.update(config)
    
    def load_audio(self, file_path, start=0, duration=None):
        """Charge un fichier audio avec le taux d'échantillonnage spécifié"""
        try:
            y, sr = librosa.load(file_path, sr=self.config['sample_rate'], offset=start, duration=duration)
            return y, sr
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
            return None, None
    
    def normalize_audio(self, y):
        """Normalise l'audio pour avoir une amplitude maximale de 1"""
        if y is None:
            return None
        
        if self.config['normalize']:
            max_amp = np.max(np.abs(y))
            if max_amp > 0:
                y = y / max_amp
        return y
    
    def segment_audio(self, y, sr):
        """Segmente l'audio en segments de durée fixe avec chevauchement"""
        if y is None or sr is None:
            return []
        
        segment_length = int(self.config['segment_duration'] * sr)
        overlap_length = int(self.config['overlap'] * sr)
        hop_length = segment_length - overlap_length
        
        # Calculer le nombre de segments
        n_segments = 1 + (len(y) - segment_length) // hop_length
        if n_segments <= 0:
            # Si l'audio est trop court, le renvoyer tel quel s'il dépasse la durée minimale
            if len(y) >= int(self.config['min_duration'] * sr):
                return [y]
            else:
                return []
        
        # Créer les segments
        segments = []
        for i in range(n_segments):
            start = i * hop_length
            end = start + segment_length
            segment = y[start:end]
            
            # Vérifier si le segment est assez long
            if len(segment) >= int(self.config['min_duration'] * sr):
                # Padding si nécessaire
                if len(segment) < segment_length:
                    segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
                segments.append(segment)
        
        return segments
    
    def compute_melspectrogram(self, y, sr):
        """Calcule le spectrogramme mel d'un signal audio"""
        if y is None or sr is None:
            return None
        
        # Calculer le spectrogramme mel
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels']
        )
        
        # Convertir en décibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def augment_audio(self, y, sr):
        """Applique des techniques d'augmentation de données à l'audio"""
        if y is None or sr is None or not self.config['augment']:
            return y
        
        # Appliquer des augmentations aléatoires
        augmented = y.copy()
        
        # 1. Pitch shift (changement de hauteur)
        if np.random.rand() > 0.5:
            n_steps = np.random.uniform(-3, 3)
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
        
        # 2. Time stretch (étirement temporel)
        if np.random.rand() > 0.5:
            rate = np.random.uniform(0.8, 1.2)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
            
            # Ajuster la longueur si nécessaire
            if len(augmented) > len(y):
                augmented = augmented[:len(y)]
            elif len(augmented) < len(y):
                augmented = np.pad(augmented, (0, len(y) - len(augmented)), 'constant')
        
        # 3. Ajout de bruit blanc
        if np.random.rand() > 0.5:
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(len(augmented))
            augmented = augmented + noise_level * noise
        
        # 4. Inversion de temps
        if np.random.rand() > 0.8:  # Moins fréquent
            augmented = np.flip(augmented)
        
        return augmented
    
    def process_file(self, file_path, output_dir=None, augment=False):
        """Traite un fichier audio et sauvegarde les spectrogrammes mel"""
        # Charger l'audio
        y, sr = self.load_audio(file_path)
        if y is None:
            return []
        
        # Normaliser l'audio
        y = self.normalize_audio(y)
        
        # Segmenter l'audio
        segments = self.segment_audio(y, sr)
        
        # Traiter chaque segment
        spectrograms = []
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        
        for i, segment in enumerate(segments):
            # Appliquer l'augmentation si demandé
            if augment:
                segment = self.augment_audio(segment, sr)
            
            # Calculer le spectrogramme mel
            mel_spec = self.compute_melspectrogram(segment, sr)
            
            if mel_spec is not None:
                spectrograms.append(mel_spec)
                
                # Sauvegarder le spectrogramme si un répertoire de sortie est spécifié
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"{file_base}_segment_{i}.npy")
                    np.save(output_file, mel_spec)
        
        return spectrograms
    
    def process_directory(self, input_dir, output_dir, pattern="*.ogg", augment=False):
        """Traite tous les fichiers audio d'un répertoire"""
        import glob
        
        # Trouver tous les fichiers audio correspondant au motif
        files = []
        for ext in ['.ogg', '.mp3', '.wav']:
            files.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))
        
        print(f"Traitement de {len(files)} fichiers audio...")
        
        # Traiter chaque fichier
        all_spectrograms = []
        for file in tqdm(files):
            spectrograms = self.process_file(file, output_dir, augment)
            all_spectrograms.extend(spectrograms)
        
        print(f"Traitement terminé. {len(all_spectrograms)} spectrogrammes générés.")
        
        return all_spectrograms


# Tester le préprocesseur sur un fichier d'exemple
preprocessor = AudioPreprocessor()
example_file = np.random.choice(train_audio_files)
print(f"Prétraitement du fichier: {os.path.basename(example_file)}")

# Prétraiter le fichier
spectrograms = preprocessor.process_file(example_file)
print(f"Nombre de spectrogrammes générés: {len(spectrograms)}")

# Visualiser un spectrogramme
if spectrograms:
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrograms[0], sr=preprocessor.config['sample_rate'], 
                            hop_length=preprocessor.config['hop_length'], x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme Mel')
    plt.tight_layout()
    plt.show()


class FeatureExtractor:
    """Classe pour extraire des caractéristiques à partir des spectrogrammes mel"""
    
    def __init__(self, config=None):
        """Initialise l'extracteur de caractéristiques avec la configuration spécifiée"""
        # Configuration par défaut
        self.config = {
            'sample_rate': 32000,  # Taux d'échantillonnage
            'n_mels': 128,         # Nombre de bandes mel
            'features': [
                'mfcc',            # Coefficients cepstraux à l'échelle de Mel
                'spectral_contrast', # Contraste spectral
                'chroma',          # Caractéristiques chromatiques
                'spectral_flatness', # Platitude spectrale
                'spectral_bandwidth', # Largeur de bande spectrale
                'spectral_rolloff', # Rolloff spectral
                'zero_crossing_rate', # Taux de passage par zéro
                'rms'              # Valeur RMS
            ],
            'n_mfcc': 20,          # Nombre de MFCCs à extraire
            'n_chroma': 12,        # Nombre de bandes chromatiques
            'standardize': True    # Standardiser les caractéristiques
        }
        
        # Mettre à jour la configuration si fournie
        if config:
            self.config.update(config)
    
    def extract_features_from_melspec(self, mel_spec, sr=None):
        """Extrait des caractéristiques à partir d'un spectrogramme mel"""
        if mel_spec is None:
            return None
        
        # Convertir de dB à puissance si nécessaire
        if np.min(mel_spec) < 0:
            mel_spec_power = librosa.db_to_power(mel_spec)
        else:
            mel_spec_power = mel_spec
        
        features = {}
        
        # Extraire les caractéristiques demandées
        if 'mfcc' in self.config['features']:
            # Extraire les MFCCs directement à partir du spectrogramme mel
            mfccs = librosa.feature.mfcc(
                S=mel_spec_power,
                n_mfcc=self.config['n_mfcc'],
                sr=self.config['sample_rate'] if sr is None else sr
            )
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            features['mfcc_max'] = np.max(mfccs, axis=1)
            features['mfcc_min'] = np.min(mfccs, axis=1)
        
        # Extraire des statistiques globales du spectrogramme mel
        features['mel_mean'] = np.mean(mel_spec, axis=1)
        features['mel_std'] = np.std(mel_spec, axis=1)
        features['mel_max'] = np.max(mel_spec, axis=1)
        features['mel_min'] = np.min(mel_spec, axis=1)
        
        # Calculer des caractéristiques temporelles
        features['temporal_flatness'] = np.mean(mel_spec, axis=0)
        features['temporal_std'] = np.std(mel_spec, axis=0)
        
        # Calculer des caractéristiques de forme
        features['spectral_centroid'] = np.mean(np.sum(mel_spec * np.arange(mel_spec.shape[0])[:, np.newaxis], axis=0) / np.sum(mel_spec, axis=0))
        features['spectral_bandwidth'] = np.mean(np.sqrt(np.sum(((np.arange(mel_spec.shape[0])[:, np.newaxis] - features['spectral_centroid']) ** 2) * mel_spec, axis=0) / np.sum(mel_spec, axis=0)))
        
        # Calculer des caractéristiques de texture
        features['spectral_contrast'] = np.mean(np.max(mel_spec, axis=0) - np.min(mel_spec, axis=0))
        features['spectral_flatness'] = np.mean(np.exp(np.mean(np.log(mel_spec + 1e-10), axis=0)) / np.mean(mel_spec, axis=0))
        
        # Aplatir les caractéristiques en un vecteur
        feature_vector = self._flatten_features(features)
        
        return feature_vector
    
    def _flatten_features(self, features_dict):
        """Aplatit un dictionnaire de caractéristiques en un vecteur"""
        feature_vector = []
        
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector)


# Tester l'extracteur de caractéristiques sur un spectrogramme d'exemple
if spectrograms:
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_melspec(spectrograms[0])
    print(f"Nombre de caractéristiques extraites: {len(features)}")
    
    # Visualiser quelques caractéristiques
    plt.figure(figsize=(12, 6))
    plt.plot(features[:50])
    plt.title('Premières 50 caractéristiques')
    plt.xlabel('Index de caractéristique')
    plt.ylabel('Valeur')
    plt.grid(True)
    plt.show()


class BirdCLEFModel:
    """Classe pour construire et entraîner un modèle de classification multi-étiquettes pour BirdCLEF+ 2025"""
    
    def __init__(self, config=None):
        """Initialise le modèle avec la configuration spécifiée"""
        # Configuration par défaut
        self.config = {
            'input_shape': (128, None, 1),  # (mel_bins, time_steps, channels)
            'model_type': 'cnn',            # Type de modèle: 'cnn', 'rnn', 'crnn'
            'num_classes': 206,             # Nombre de classes (espèces)
            'learning_rate': 0.001,         # Taux d'apprentissage
            'batch_size': 32,               # Taille du batch
            'epochs': 50,                   # Nombre d'époques
            'patience': 10,                 # Patience pour l'early stopping
            'dropout_rate': 0.5,            # Taux de dropout
            'use_augmentation': True,       # Utiliser l'augmentation de données
            'class_weights': None,          # Poids des classes pour gérer le déséquilibre
            'model_path': 'models/birdclef_model.h5'  # Chemin pour sauvegarder le modèle
        }
        
        # Mettre à jour la configuration si fournie
        if config:
            self.config.update(config)
        
        # Initialiser le modèle
        self.model = None
    
    def build_cnn_model(self):
        """Construit un modèle CNN pour la classification audio"""
        inputs = layers.Input(shape=self.config['input_shape'])
        
        # Premier bloc convolutif
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Deuxième bloc convolutif
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Troisième bloc convolutif
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Quatrième bloc convolutif
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Couches denses
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Couche de sortie (sigmoid pour classification multi-étiquettes)
        outputs = layers.Dense(self.config['num_classes'], activation='sigmoid')(x)
        
        # Créer le modèle
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_rnn_model(self):
        """Construit un modèle RNN pour la classification audio"""
        inputs = layers.Input(shape=self.config['input_shape'])
        
        # Reshape pour RNN (mel_bins, time_steps, channels) -> (time_steps, mel_bins * channels)
        x = layers.Reshape((-1, self.config['input_shape'][0] * self.config['input_shape'][2]))(inputs)
        
        # Couches LSTM bidirectionnelles
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Couches denses
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Couche de sortie (sigmoid pour classification multi-étiquettes)
        outputs = layers.Dense(self.config['num_classes'], activation='sigmoid')(x)
        
        # Créer le modèle
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_crnn_model(self):
        """Construit un modèle CRNN (CNN + RNN) pour la classification audio"""
        inputs = layers.Input(shape=self.config['input_shape'])
        
        # Partie CNN
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Reshape pour RNN
        # (batch, freq, time, channels) -> (batch, time, freq * channels)
        x = layers.Reshape((-1, x.shape[1] * x.shape[3]))(x)
        
        # Partie RNN
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Couches denses
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Couche de sortie (sigmoid pour classification multi-étiquettes)
        outputs = layers.Dense(self.config['num_classes'], activation='sigmoid')(x)
        
        # Créer le modèle
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_model(self):
        """Construit le modèle selon le type spécifié dans la configuration"""
        if self.config['model_type'] == 'cnn':
            self.model = self.build_cnn_model()
        elif self.config['model_type'] == 'rnn':
            self.model = self.build_rnn_model()
        elif self.config['model_type'] == 'crnn':
            self.model = self.build_crnn_model()
        else:
            raise ValueError(f"Type de modèle non reconnu: {self.config['model_type']}")
        
        return self.model


# Créer et afficher un modèle CNN
model_config = {
    'input_shape': (128, 128, 1),  # Forme fixe pour l'exemple
    'model_type': 'cnn',
    'num_classes': len(species_columns)
}

model_builder = BirdCLEFModel(model_config)
model = model_builder.build_model()
model.summary()


def load_data(features_file, labels_file=None, test_size=0.2, random_state=42):
    """Charge les données d'entraînement et de validation"""
    # Charger les caractéristiques
    features_df = pd.read_csv(features_file)
    
    # Séparer les noms de fichiers et les caractéristiques
    file_names = features_df['file_name'].values
    features = features_df.drop('file_name', axis=1).values
    
    # Charger les étiquettes si un fichier est fourni
    if labels_file:
        labels_df = pd.read_csv(labels_file)
        labels = labels_df.drop('file_name', axis=1).values
    else:
        # Créer des étiquettes factices pour les tests
        print("Aucun fichier d'étiquettes fourni, création d'étiquettes factices pour les tests.")
        num_classes = 206  # Nombre d'espèces dans BirdCLEF+ 2025
        labels = np.random.randint(0, 2, size=(len(file_names), num_classes))
    
    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val

def reshape_features_for_model(X, input_shape):
    """Reshape les caractéristiques pour correspondre à l'entrée du modèle"""
    # Déterminer la forme cible
    target_shape = (-1,) + input_shape
    
    # Reshape les caractéristiques
    try:
        X_reshaped = X.reshape(target_shape)
        return X_reshaped
    except ValueError:
        print(f"Erreur lors du reshape des caractéristiques de {X.shape} à {target_shape}")
        # Essayer une approche alternative
        n_samples = X.shape[0]
        X_reshaped = np.zeros((n_samples,) + input_shape)
        
        # Copier autant de données que possible
        for i in range(n_samples):
            # Déterminer les dimensions à copier
            copy_shape = tuple(min(dim, X.shape[j+1]) if j+1 < len(X.shape) else min(dim, 1) 
                              for j, dim in enumerate(input_shape))
            
            # Créer des slices pour la copie
            slices_src = tuple(slice(None, dim) for dim in copy_shape)
            slices_dst = tuple(slice(None, dim) for dim in copy_shape)
            
            # Copier les données
            if len(X.shape) == 2:  # Caractéristiques 1D
                X_reshaped[i, :copy_shape[0], 0, 0] = X[i, :copy_shape[0]]
            else:  # Caractéristiques multidimensionnelles
                X_reshaped[i][slices_dst] = X[i][slices_src]
        
        return X_reshaped

def calculate_class_weights(y_train):
    """Calcule les poids des classes pour gérer le déséquilibre"""
    # Calculer le nombre d'échantillons positifs pour chaque classe
    positive_counts = np.sum(y_train, axis=0)
    
    # Calculer le nombre total d'échantillons
    n_samples = y_train.shape[0]
    
    # Calculer les poids des classes
    class_weights = {}
    for i in range(y_train.shape[1]):
        # Éviter la division par zéro
        if positive_counts[i] > 0:
            # Formule de poids inversement proportionnel à la fréquence
            weight = n_samples / (2 * positive_counts[i])
        else:
            weight = 1.0
        
        class_weights[i] = weight
    
    return class_weights


# Créer des données factices pour tester l'entraînement
n_samples = 1000
n_features = 500
n_classes = len(species_columns)

# Caractéristiques et étiquettes factices
X_train_dummy = np.random.rand(n_samples, n_features)
X_val_dummy = np.random.rand(n_samples // 5, n_features)
y_train_dummy = np.random.randint(0, 2, size=(n_samples, n_classes))
y_val_dummy = np.random.randint(0, 2, size=(n_samples // 5, n_classes))

# Calculer les poids des classes
class_weights = calculate_class_weights(y_train_dummy)

# Reshape les caractéristiques pour le modèle
input_shape = (128, 128, 1)  # Forme fixe pour l'exemple
X_train_reshaped = reshape_features_for_model(X_train_dummy, input_shape)
X_val_reshaped = reshape_features_for_model(X_val_dummy, input_shape)

print(f"Forme des caractéristiques d'entraînement: {X_train_reshaped.shape}")
print(f"Forme des étiquettes d'entraînement: {y_train_dummy.shape}")


# Entraîner le modèle sur les données factices (juste pour démonstration)
# Dans un cas réel, nous utiliserions les vraies données

# Configuration du modèle
model_config = {
    'input_shape': input_shape,
    'model_type': 'cnn',
    'num_classes': n_classes,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 5,  # Réduit pour la démonstration
    'patience': 3,
    'dropout_rate': 0.5,
    'class_weights': class_weights
}

# Créer et construire le modèle
model_builder = BirdCLEFModel(model_config)
model = model_builder.build_model()

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_auc',
        patience=model_config['patience'],
        restore_best_weights=True,
        mode='max'
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

# Entraîner le modèle
history = model.fit(
    X_train_reshaped, y_train_dummy,
    batch_size=model_config['batch_size'],
    epochs=model_config['epochs'],
    validation_data=(X_val_reshaped, y_val_dummy),
    callbacks=callbacks_list,
    class_weight=model_config['class_weights']
)


# Visualiser l'historique d'entraînement
def visualize_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Visualiser la perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    # Visualiser l'AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC')
    plt.xlabel('Époque')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualiser l'historique d'entraînement
visualize_training_history(history)


# Exemple d'optimisation des hyperparamètres avec Optuna
# Note: Cette cellule est commentée car l'exécution complète prendrait trop de temps
# Dans un cas réel, nous exécuterions cette optimisation

'''
import optuna
from optuna.integration import TFKerasPruningCallback

def objective(trial):
    # Hyperparamètres à optimiser
    model_type = trial.suggest_categorical('model_type', ['cnn', 'rnn', 'crnn'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    
    # Configuration du modèle
    model_config = {
        'input_shape': input_shape,
        'model_type': model_type,
        'num_classes': n_classes,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': 20,
        'patience': 5,
        'dropout_rate': dropout_rate,
        'class_weights': class_weights
    }
    
    # Créer et construire le modèle
    model_builder = BirdCLEFModel(model_config)
    model = model_builder.build_model()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=model_config['patience'],
            restore_best_weights=True,
            mode='max'
        ),
        TFKerasPruningCallback(trial, 'val_auc')
    ]
    
    # Entraîner le modèle
    history = model.fit(
        X_train_reshaped, y_train_dummy,
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(X_val_reshaped, y_val_dummy),
        callbacks=callbacks_list,
        class_weight=model_config['class_weights'],
        verbose=0
    )
    
    # Retourner la meilleure valeur d'AUC
    return max(history.history['val_auc'])

# Créer l'étude Optuna
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres:")
print(study.best_params)
print(f"Meilleure valeur d'AUC: {study.best_value:.4f}")
'''


# Exemple d'ensemble de modèles
# Note: Cette cellule est commentée car l'exécution complète prendrait trop de temps
# Dans un cas réel, nous créerions un ensemble de modèles

'''
from sklearn.model_selection import KFold

# Créer un ensemble de modèles avec validation croisée
n_models = 5
kf = KFold(n_splits=n_models, shuffle=True, random_state=42)

ensemble_models = []
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_reshaped)):
    print(f"\nEntraînement du modèle {fold+1}/{n_models}")
    
    # Diviser les données pour cette fold
    X_fold_train, X_fold_val = X_train_reshaped[train_idx], X_train_reshaped[val_idx]
    y_fold_train, y_fold_val = y_train_dummy[train_idx], y_train_dummy[val_idx]
    
    # Calculer les poids des classes pour cette fold
    fold_class_weights = calculate_class_weights(y_fold_train)
    
    # Configuration du modèle
    fold_config = {
        'input_shape': input_shape,
        'model_type': 'cnn',  # Utiliser le meilleur type de modèle trouvé par Optuna
        'num_classes': n_classes,
        'learning_rate': 0.001,  # Utiliser le meilleur taux d'apprentissage trouvé par Optuna
        'batch_size': 32,  # Utiliser la meilleure taille de batch trouvée par Optuna
        'epochs': 20,
        'patience': 5,
        'dropout_rate': 0.5,  # Utiliser le meilleur taux de dropout trouvé par Optuna
        'class_weights': fold_class_weights
    }
    
    # Créer et construire le modèle
    model_builder = BirdCLEFModel(fold_config)
    model = model_builder.build_model()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=fold_config['patience'],
            restore_best_weights=True,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Entraîner le modèle
    history = model.fit(
        X_fold_train, y_fold_train,
        batch_size=fold_config['batch_size'],
        epochs=fold_config['epochs'],
        validation_data=(X_fold_val, y_fold_val),
        callbacks=callbacks_list,
        class_weight=fold_config['class_weights']
    )
    
    # Évaluer le modèle
    metrics = model.evaluate(X_val_reshaped, y_val_dummy)
    fold_metrics.append({
        'loss': metrics[0],
        'accuracy': metrics[1],
        'auc': metrics[2]
    })
    
    # Ajouter le modèle à l'ensemble
    ensemble_models.append(model)

# Prédire avec l'ensemble
def predict_with_ensemble(ensemble_models, X):
    # Prédire avec chaque modèle
    predictions = []
    for model in ensemble_models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Moyenner les prédictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Évaluer l'ensemble
ensemble_pred = predict_with_ensemble(ensemble_models, X_val_reshaped)
ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

# Calculer les métriques
ensemble_metrics = {
    'f1_micro': f1_score(y_val_dummy, ensemble_pred_binary, average='micro'),
    'f1_macro': f1_score(y_val_dummy, ensemble_pred_binary, average='macro'),
    'precision_micro': precision_score(y_val_dummy, ensemble_pred_binary, average='micro'),
    'precision_macro': precision_score(y_val_dummy, ensemble_pred_binary, average='macro'),
    'recall_micro': recall_score(y_val_dummy, ensemble_pred_binary, average='micro'),
    'recall_macro': recall_score(y_val_dummy, ensemble_pred_binary, average='macro'),
    'roc_auc': roc_auc_score(y_val_dummy, ensemble_pred, average='macro')
}

print("\nMétriques de l'ensemble:")
for metric, value in ensemble_metrics.items():
    print(f"{metric}: {value:.4f}")
'''


def preprocess_test_data(test_audio_dir, output_dir=None):
    """Prétraite les données de test"""
    # Créer le préprocesseur
    preprocessor = AudioPreprocessor()
    
    # Prétraiter les fichiers audio
    spectrograms = preprocessor.process_directory(test_audio_dir, output_dir, pattern="*.ogg")
    
    return spectrograms

def extract_features_from_spectrograms(spectrograms):
    """Extrait des caractéristiques à partir des spectrogrammes"""
    # Créer l'extracteur de caractéristiques
    extractor = FeatureExtractor()
    
    # Extraire les caractéristiques
    features_list = []
    file_names = []
    
    for i, spec in enumerate(tqdm(spectrograms)):
        features = extractor.extract_features_from_melspec(spec)
        if features is not None:
            features_list.append(features)
            file_names.append(f"test_audio_{i}.ogg")
    
    # Créer un DataFrame avec les caractéristiques
    features_df = pd.DataFrame(features_list)
    features_df['file_name'] = file_names
    
    return features_df

def generate_submission_file(predictions, row_ids, species_columns, output_file):
    """Génère un fichier de soumission"""
    # Créer un DataFrame de soumission
    submission = pd.DataFrame()
    submission['row_id'] = row_ids
    
    # Ajouter les prédictions pour chaque espèce
    for i, species in enumerate(species_columns):
        submission[species] = predictions[:, i]
    
    # Sauvegarder le fichier de soumission
    submission.to_csv(output_file, index=False)
    
    return submission


# Exemple de génération de prédictions
# Note: Cette cellule est commentée car nous n'avons pas de vraies données de test
# Dans un cas réel, nous exécuterions ce code


# Prétraiter les données de test
test_spectrograms = preprocess_test_data(TEST_AUDIO_DIR, output_dir="test_spectrograms")

# Extraire les caractéristiques
test_features_df = extract_features_from_spectrograms(test_spectrograms)

# Séparer les noms de fichiers et les caractéristiques
test_file_names = test_features_df['file_name'].values
test_features = test_features_df.drop('file_name', axis=1).values

# Reshape les caractéristiques pour le modèle
test_features_reshaped = reshape_features_for_model(test_features, input_shape)

# Prédire avec le modèle ou l'ensemble
#print("Prédiction avec l'ensemble de modèles...")
#test_predictions = predict_with_ensemble(ensemble_models, test_features_reshaped)
print("Prédiction avec le modèle unique...")
test_predictions = model.predict(test_features_reshaped)

# Charger le fichier d'exemple de soumission pour obtenir les row_ids
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
row_ids = sample_submission['row_id'].values
species_cols = sample_submission.columns[1:].tolist()

# Générer le fichier de soumission
submission = generate_submission_file(test_predictions, row_ids, species_cols, "submission.csv")
submission.head()


def submit_to_kaggle(submission_file, competition_name, message=None):
    """Soumet les résultats à la compétition Kaggle"""
    import subprocess
    
    # Vérifier si le fichier de soumission existe
    if not os.path.exists(submission_file):
        print(f"Le fichier de soumission {submission_file} n'existe pas.")
        return False
    
    # Construire la commande de soumission
    cmd = ["kaggle", "competitions", "submit", "-c", competition_name, "-f", submission_file]
    
    # Ajouter le message de soumission si spécifié
    if message:
        cmd.extend(["-m", message])
    
    # Exécuter la commande
    try:
        print(f"Soumission du fichier {submission_file} à la compétition {competition_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Soumission réussie!")
            print(result.stdout)
            return True
        else:
            print(f"Erreur lors de la soumission: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception lors de la soumission: {e}")
        return False


# Exemple de soumission à Kaggle
# Note: Cette cellule est commentée car nous n'avons pas de vrai fichier de soumission
# Dans un cas réel, nous exécuterions ce code


# Soumettre les résultats à Kaggle
submission_success = submit_to_kaggle(
    submission_file="submission.csv",
    competition_name="birdclef-2025",
    message="Soumission avec modèle CNN et ensemble de 5 modèles"
)

if submission_success:
    print("Soumission réussie! Vérifiez votre score sur la page de la compétition.")
else:
    print("La soumission a échoué. Vérifiez les erreurs ci-dessus.")

