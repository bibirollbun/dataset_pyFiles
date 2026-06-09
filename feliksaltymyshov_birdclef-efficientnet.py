import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T # Используем transforms v2 для SpecAugment
import torchaudio.transforms as AT 
import pathlib
import timm # Библиотека для моделей, включая EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm # Для отображения прогресса
import warnings
import random

# Игнорировать предупреждения от librosa (например, об audiocore)
warnings.filterwarnings('ignore')


# Определение основных параметров и путей
class Config:
    SEED = 42 # Для воспроизводимости
    SAMPLE_RATE = 32000 # Целевая частота дискретизации [14]
    DURATION_SECONDS = 5 # Длительность сегмента для обучения и предсказания [20]
    N_MELS = 64 # Количество мел-полос [56] - может потребовать настройки
    FMIN = 50 # Минимальная частота для мел-спектрограммы [56] - может потребовать настройки
    FMAX = 15000 # Максимальная частота для мел-спектрограммы [56] - может потребовать настройки
    N_FFT = 2048 # Размер окна БПФ [56] - может потребовать настройки
    HOP_LENGTH = 512 # Шаг окна БПФ [56] - может потребовать настройки
    N_CLASSES = 206 # Количество видов в соревновании [20]
    MODEL_NAME = 'tf_efficientnet_b0' # Эффективная модель [6, 80] (_ns - noisy student)
    MODEL_PATH = '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth' 
    PRETRAINED = False # Использовать предобученные веса ImageNet [6, 60]
    BATCH_SIZE = 64
    INFERENCE_BATCH_SIZE = 64 # Can be larger than training batch size
    LEARNING_RATE = 1e-3
    EPOCHS = 3 # Количество эпох (может потребоваться больше)
    NUM_WORKERS = 2 # Количество потоков для загрузки данных
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Пути
    DATA_DIR = "/kaggle/input/birdclef-2025"
    SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
    TEST_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "test_soundscapes")
    SUBMISSION_CSV_PATH = "submission.csv"
    Taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
    TRAIN_METADATA_PATH = os.path.join(DATA_DIR, "train.csv")


    # Параметры аугментации
    MIXUP_ALPHA = 0.3 
    USE_MIXUP = True
    USE_SPECAUGMENT = True 

# Функция для установки seed для воспроизводимости
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Можно установить в False для полной детерминированности, но медленнее

seed_everything(Config.SEED)

# --- Загрузка таксономии ---
print("Loading taxonomy data...")
try:
    taxonomy_df = pd.read_csv(Config.Taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    num_classes = len(species_ids)
    print(f"Number of classes: {num_classes}")
except FileNotFoundError:
    print(f"Ошибка: Файл таксономии не найден по пути {cfg.taxonomy_csv}")
    # Можно попробовать загрузить из sample_submission, если таксономии нет
    try:
        sample_sub = pd.read_csv(cfg.submission_csv)
        species_ids = sample_sub.columns[1:].tolist()
        num_classes = len(species_ids)
        print(f"Загружены классы из sample_submission. Количество классов: {num_classes}")
    except FileNotFoundError:
        print(f"Ошибка: Sample submission не найден по пути {cfg.submission_csv}. Невозможно определить классы.")
        exit() # Выход, если классы определить не удалось
except Exception as e:
    print(f"Ошибка при загрузке таксономии: {e}")
    exit()


try:
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    print(f"Загружено {len(train_df)} записей из {Config.TRAIN_METADATA_PATH}")
except FileNotFoundError:
    print(f"Ошибка: Файл {Config.TRAIN_METADATA_PATH} не найден. Укажите правильный путь.")
    # Можно создать dummy DataFrame для продолжения работы над кодом
    train_df = pd.DataFrame({
        'filename': [f'bird_{i}.ogg' for i in range(100)],
        'primary_label': [f'species_{i % 10}' for i in range(100)],
        'secondary_labels': [[] for _ in range(100)] # Пример пустых вторичных меток
    })
    print("Создан примерный DataFrame для демонстрации.")


all_labels = set(train_df['primary_label'].unique())
# Обработка вторичных меток (они могут быть строкой вида '["label1", "label2"]')
def parse_secondary_labels(labels_str):
    try:
        # Используем eval безопасно, т.к. ожидаем формат списка строк
        labels = eval(labels_str)
        return [label for label in labels if isinstance(label, str)]
    except:
        return []

secondary_labels_list = train_df['secondary_labels'].apply(parse_secondary_labels).sum()
all_labels.update(secondary_labels_list)
sorted_labels = sorted(list(all_labels))
sorted_labels.remove('')


# Проверяем, соответствует ли количество найденных классов Config.N_CLASSES
if len(sorted_labels) != Config.N_CLASSES:
    print(f"Предупреждение: Найдено {len(sorted_labels)} уникальных меток, но Config.N_CLASSES={Config.N_CLASSES}.")
    # Можно обновить Config.N_CLASSES или проверить данные/логику
    # Config.N_CLASSES = len(sorted_labels) # Раскомментировать для обновления


label_encoder = LabelEncoder()
label_encoder.fit(sorted_labels)


# Преобразование меток в one-hot encoding
def get_one_hot_vector(primary_label, secondary_labels_str, encoder):
    labels = {primary_label}
    secondary = parse_secondary_labels(secondary_labels_str)
    labels.update(secondary)

    # Создаем вектор нулей
    one_hot = np.zeros(len(encoder.classes_), dtype=np.float32)

    # Устанавливаем 1 для присутствующих меток
    for label in labels:
        if label in encoder.classes_:
            idx = encoder.transform([label])[0]
            one_hot[idx] = 1.0
    return one_hot

train_df['target'] = train_df.apply(
    lambda row: get_one_hot_vector(row['primary_label'], row['secondary_labels'], label_encoder),
    axis=1
)


train_indices, val_indices = train_test_split(
    range(len(train_df)),
    test_size=0.4, 
    random_state=Config.SEED,
    # Стратификация может быть полезна, если есть дисбаланс классов
    stratify=train_df['primary_label'] # Если классов не слишком много
)


def load_and_process_audio(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION_SECONDS):
    try:
        # Загрузка аудио
        # Используем res_type='kaiser_fast' для скорости
        wav, current_sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')

        # Ресемплинг, если необходимо
        if current_sr != sr:
            wav = librosa.resample(wav, orig_sr=current_sr, target_sr=sr, res_type='kaiser_fast')

        # Выбор или паддинг до нужной длины
        target_length = sr * duration
        if len(wav) > target_length:
            # Вырезаем случайный сегмент нужной длины
            max_offset = len(wav) - target_length
            offset = np.random.randint(max_offset)
            wav = wav[offset:(offset + target_length)]
        elif len(wav) < target_length:
            # Дополняем нулями (паддинг)
            padding = target_length - len(wav)
            offset = padding // 2
            wav = np.pad(wav, (offset, padding - offset), 'constant')
        else:
            # Длина совпадает
            pass

        return wav
    except Exception as e:
        print(f"Ошибка при загрузке {file_path}: {e}")
        return np.zeros(sr * duration, dtype=np.float32) # Возвращаем тишину в случае ошибки


# Функция для создания мел-спектрограммы
def get_mel_spectrogram(wav, sr=Config.SAMPLE_RATE, n_fft=Config.N_FFT,
                        hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS,
                        fmin=Config.FMIN, fmax=Config.FMAX):
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # Преобразование в децибелы (логарифмическая шкала)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


 class BirdDataset(Dataset):
    def __init__(self, df, indices, audio_dir, transforms=None, use_mixup=False, mixup_alpha=0.4):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transforms = transforms
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])

        # Загрузка и обработка аудио
        wav = load_and_process_audio(file_path)

        # Получение мел-спектрограммы
        mel_spec = get_mel_spectrogram(wav)

        # Нормализация спектрограммы (приводим к диапазону [0, 1] или [-1, 1])
        # Простая нормализация min-max
        min_val = np.min(mel_spec)
        max_val = np.max(mel_spec)
        if max_val > min_val:
            mel_spec = (mel_spec - min_val) / (max_val - min_val)
        else:
             mel_spec = np.zeros_like(mel_spec) # Если все значения одинаковые

        # Преобразование в тензор и добавление канала (как у изображения)
        image = torch.tensor(mel_spec).unsqueeze(0) # Shape: [1, n_mels, time_steps]

        # Применение трансформаций (аугментаций)
        if self.transforms:
            image = self.transforms(image)

        target = torch.tensor(row['target'], dtype=torch.float32)

        # Реализация Mixup [6, 36]
        if self.use_mixup and random.random() < 0.5: # Применяем Mixup с вероятностью 50%
            mix_idx = random.randint(0, len(self) - 1)
            mix_row = self.df.iloc[mix_idx]
            mix_file_path = os.path.join(self.audio_dir, mix_row['filename'])

            mix_wav = load_and_process_audio(mix_file_path)
            mix_mel_spec = get_mel_spectrogram(mix_wav)
            mix_min_val = np.min(mix_mel_spec)
            mix_max_val = np.max(mix_mel_spec)
            if mix_max_val > mix_min_val:
                 mix_mel_spec = (mix_mel_spec - mix_min_val) / (mix_max_val - mix_min_val)
            else:
                 mix_mel_spec = np.zeros_like(mix_mel_spec)

            mix_image = torch.tensor(mix_mel_spec).unsqueeze(0)
            if self.transforms:
                 mix_image = self.transforms(mix_image)

            mix_target = torch.tensor(mix_row['target'], dtype=torch.float32)

            # Смешивание
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image = lam * image + (1 - lam) * mix_image
            target = lam * target + (1 - lam) * mix_target

        return image, target


# SpecAugment [6, 36] - маскирование временных и частотных полос
# Используем T.SpecAugment из torchvision.transforms v2
train_transforms = T.Compose([
    T.RandomApply([
        AT.SpecAugment(
            # --- Добавленные аргументы ---
            n_freq_masks=1,  
            n_time_masks=1,  
            # --- Существующие аргументы (размер масок) ---
            freq_mask_param=Config.N_MELS // 8,
            time_mask_param=int(Config.DURATION_SECONDS * Config.SAMPLE_RATE / Config.HOP_LENGTH / 8)
        )
    ], p=0.5) if Config.USE_SPECAUGMENT else nn.Identity(),
    # ... другие возможные трансформации ...
])

val_transforms = T.Compose([
    nn.Identity() # Передаем тождественную(пустую) трансформацию
])


train_dataset = BirdDataset(
    train_df,
    train_indices,
    Config.TRAIN_AUDIO_DIR,
    transforms=train_transforms,
    use_mixup=Config.USE_MIXUP,
    mixup_alpha=Config.MIXUP_ALPHA
)
val_dataset = BirdDataset(
    train_df,
    val_indices,
    Config.TRAIN_AUDIO_DIR,
    transforms=val_transforms, # Без Mixup и SpecAugment для валидации
    use_mixup=False
)

# Создание загрузчиков данных
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True # Ускоряет передачу данных на GPU
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE * 2, # Можно увеличить batch_size для валидации
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)

print(f"Созданы DataLoader: {len(train_loader)} батчей для обучения, {len(val_loader)} для валидации.")



# Загрузка модели EfficientNet с помощью timm [6, 35, 60]
model = timm.create_model(
    Config.MODEL_NAME,
    pretrained=Config.PRETRAINED,
    in_chans=1, # 1 канал для спектрограммы
    num_classes=Config.N_CLASSES # Количество выходных нейронов
)

try:
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    print(f"Pre-trained weights loaded from {pretrained_weights_path}")
except FileNotFoundError:
    print(f"Error: Pre-trained weights file not found at {pretrained_weights_path}")
except Exception as e:
    print(f"Error loading pre-trained weights: {e}")

# Перемещение модели на выбранное устройство (GPU или CPU)
model.to(Config.DEVICE)
print(f"Модель {Config.MODEL_NAME} загружена и перемещена на {Config.DEVICE}.")




# Функция потерь и оптимизатор
# BCEWithLogitsLoss подходит для multi-label классификации (несколько птиц в одном клипе)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
# Можно добавить планировщик скорости обучения (scheduler)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)


# Цикл обучения
for epoch in range(Config.EPOCHS):
    print(f"\n--- Эпоха {epoch+1}/{Config.EPOCHS} ---")

    # Фаза обучения
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc="Обучение", leave=False)
    for images, targets in train_loop:
        images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)

        # Прямой проход
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    print(f"Средняя потеря на обучении: {avg_train_loss:.4f}")

    # Фаза валидации
    # model.eval()
    # val_loss = 0.0
    # all_preds = []
    # all_targets = []
    # val_loop = tqdm(val_loader, desc="Валидация", leave=False)
    # with torch.no_grad(): # Отключаем вычисление градиентов
    #     for images, targets in val_loop:
    #         images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)

    #         # Прямой проход
    #         outputs = model(images)
    #         loss = criterion(outputs, targets)
    #         val_loss += loss.item()

    #         # Сохраняем предсказания (вероятности после сигмоиды) и цели
    #         preds = torch.sigmoid(outputs)
    #         all_preds.append(preds.cpu().numpy())
    #         all_targets.append(targets.cpu().numpy())

    # avg_val_loss = val_loss / len(val_loader)
    # print(f"Средняя потеря на валидации: {avg_val_loss:.4f}")

    # # Расчет метрики ROC-AUC (требует sklearn)
    # try:
    #     from sklearn.metrics import roc_auc_score
    #     all_preds = np.concatenate(all_preds)
    #     all_targets = np.concatenate(all_targets)

    #     # Расчет макро-усредненной ROC-AUC [27]
    #     # Обработка случая, когда в батче нет примеров какого-то класса
    #     valid_targets = all_targets.sum(axis=0) > 0
    #     macro_roc_auc = roc_auc_score(
    #         all_targets[:, valid_targets], # Берем только классы, которые были в валидации
    #         all_preds[:, valid_targets],
    #         average='macro'
    #     )
    #     print(f"Макро ROC-AUC на валидации: {macro_roc_auc:.4f}")
    # except ImportError:
    #     print("Библиотека scikit-learn не найдена. Пропустите расчет ROC-AUC.")
    # except Exception as e:
    #     print(f"Ошибка при расчете ROC-AUC: {e}")


    # Обновление планировщика (если используется)
    # scheduler.step()

    # TODO: Сохранение лучшей модели на основе валидационной метрики (например, ROC-AUC)


def load_audio_segment(file_path, start_time, duration, sr=Config.SAMPLE_RATE):
    """Loads a specific segment from an audio file."""
    try:
        # Load the audio, potentially just the required segment
        # Using offset and duration in librosa load
        # Ensure duration doesn't exceed file length from offset
        info = torchaudio.info(str(file_path))
        file_duration = info.num_frames / info.sample_rate
        actual_duration_to_load = min(duration, file_duration - start_time)
        if actual_duration_to_load <= 0:
             return np.zeros(int(duration * sr), dtype=np.float32) # Segment is beyond file end

        wav, current_sr = librosa.load(
            file_path,
            sr=None, # Load at original sample rate
            offset=start_time, # Start loading from this time
            duration=actual_duration_to_load, # Load for this duration
            res_type='kaiser_fast'
        )

        # Resample if necessary
        if current_sr != sr:
            wav = librosa.resample(wav, orig_sr=current_sr, target_sr=sr, res_type='kaiser_fast')

        # Pad if the loaded segment is shorter than expected (e.g., at the very end of the file)
        target_length = int(duration * sr)
        if len(wav) < target_length:
            padding = target_length - len(wav)
            wav = np.pad(wav, (0, padding), 'constant')
        elif len(wav) > target_length:
             # Should not happen with offset+duration unless calculation is off
             wav = wav[:target_length]

        return wav

    except Exception as e:
        print(f"Error loading segment from {file_path} at {start_time}s: {e}")
        return np.zeros(int(duration * sr), dtype=np.float32) # Return silence on error


def normalize_spectrogram(mel_spec):
    """Normalizes mel spectrogram similar to training."""
    min_val = np.min(mel_spec)
    max_val = np.max(mel_spec)
    if max_val > min_val:
        # Simple min-max scaling to [0, 1]
        mel_spec = (mel_spec - min_val) / (max_val - min_val)
    else:
         # Handle case with constant values (e.g., silence)
         mel_spec = np.zeros_like(mel_spec)
    return mel_spec


# --- Test Dataset ---
class TestSoundscapesDataset(Dataset):
    def __init__(self, segment_list, audio_dir, config):
        """
        Args:
            segment_list (list): List of tuples (filename, start_time_seconds, soundscape_id).
                                 soundscape_id is extracted from filename (e.g., '12345' from 'soundscape_12345.ogg')
            audio_dir (str): Directory containing the soundscape audio files.
            config: Configuration object.
        """
        self.segment_list = segment_list
        self.audio_dir = audio_dir
        self.config = config

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        filename, start_time, soundscape_id = self.segment_list[idx]
        file_path = os.path.join(self.audio_dir, filename)

        # Load the specific segment
        wav = load_audio_segment(file_path, start_time, self.config.DURATION_SECONDS, self.config.SAMPLE_RATE)

        # Get mel spectrogram
        mel_spec = get_mel_spectrogram(
            wav,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX
        )

        # Normalize
        mel_spec = normalize_spectrogram(mel_spec)

        # Convert to tensor and add channel dimension
        # Expected shape by EfficientNet is [Batch, Channels, Height, Width] -> [B, 1, N_Mels, TimeSteps]
        image = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)

        # Create row_id: soundscape_[soundscape_id]_[end_time]
        end_time = int(start_time + self.config.DURATION_SECONDS) # End time is start + duration
        row_id = f"soundscape_{soundscape_id}_{end_time}"

        return image, row_id

# --- Main Submission Logic ---
print("Starting submission script...")


# --- Load Sample Submission to get required row_ids and species columns ---
sample_submission_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
required_row_ids = sample_submission_df['row_id'].tolist()
submission_species_cols = sample_submission_df.columns[1:].tolist()
print(f"Loaded {len(required_row_ids)} row IDs and {len(submission_species_cols)} species columns from sample submission.")

if len(submission_species_cols) != Config.N_CLASSES:
     print(f"FATAL Error: Number of species columns in sample submission ({len(submission_species_cols)}) does not match Config.N_CLASSES ({Config.N_CLASSES}). Check configuration or data.")


# --- Load Label Encoder ---
# We need the label encoder that maps species names to indices (0 to N_CLASSES-1)
# based on how your model was trained. It's crucial this mapping is consistent.
# The safest way is to rebuild it using the same sorted list of ALL species
# found in the training data primary/secondary labels.
print("Creating/Loading label encoder...")
try:
    train_df_for_labels = pd.read_csv(Config.TRAIN_METADATA_PATH)
    all_labels_set = set(train_df_for_labels['primary_label'].unique())
    def parse_secondary_labels_safe(labels_str):
        try:
            # Use ast.literal_eval for safer evaluation if the string is a list literal
            import ast
            labels = ast.literal_eval(labels_str)
            return [label for label in labels if isinstance(label, str)]
        except:
            return [] # Return empty list on error

    secondary_labels_list_for_labels = train_df_for_labels['secondary_labels'].apply(parse_secondary_labels_safe).sum()
    all_labels_set.update(secondary_labels_list_for_labels)
    sorted_labels_for_encoder = sorted(list(all_labels_set))
    if '' in sorted_labels_for_encoder: # Remove potential empty string if present
        sorted_labels_for_encoder.remove('')

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted_labels_for_encoder)

    print(f"Label encoder fitted with {len(label_encoder.classes_)} classes based on training data.")
    if len(label_encoder.classes_) != Config.N_CLASSES:
         print(f"Warning: Number of classes in encoder ({len(label_encoder.classes_)}) does not match Config.N_CLASSES ({Config.N_CLASSES}). Check your training data / Config.N_CLASSES.")
         # You might need to adjust N_CLASSES in Config if the actual number of species in train.csv is different
         # Or check why the species list derived from training data doesn't match 206.

except FileNotFoundError as e:
    print(f"Error loading training data for label encoder: {e}. Cannot build robust encoder.")
    # If you cannot load train.csv, you might have issues mapping model outputs correctly.
    # This is a critical point - the mapping MUST be correct.
    # A potential fallback is to use the species names from sample_submission and hope they cover everything,
    # but this is risky if your model was trained on more/different species.
    print("FATAL Error: Could not build label encoder from training data. Please check paths.")
except Exception as e:
    print(f"An unexpected error occurred while building label encoder: {e}")

# --- Create a mapping from species ID string (from submission columns) to encoder index ---
# This maps the required output column to the index in the model's output tensor
species_id_to_encoder_idx = {
    species_id: label_encoder.transform([species_id])[0]
    for species_id in submission_species_cols
    if species_id in label_encoder.classes_ # Ensure species from submission are known to encoder
}
# Check if all submission species are known
if len(species_id_to_encoder_idx) != len(submission_species_cols):
     missing_species = [sp for sp in submission_species_cols if sp not in label_encoder.classes_]
     print(f"Warning: The following species from sample_submission are NOT in the training data's label encoder: {missing_species}")
     # Predictions for these species will likely be zeros or based on random chance, depending on model's final layer initialization

# --- Load Model ---
# print(f"Loading model from {Config.SAVED_MODEL_PATH}...")
# model = timm.create_model(
#     Config.MODEL_NAME,
#     pretrained=False, # We are loading custom weights
#     in_chans=1,
#     num_classes=Config.N_CLASSES # Model output size must match the number of classes the encoder knows
# )

# try:
#     model.load_state_dict(torch.load(Config.SAVED_MODEL_PATH, map_location=Config.DEVICE))
#     print("Model weights loaded successfully.")
# except FileNotFoundError:
#     print(f"FATAL Error: Saved model not found at {Config.SAVED_MODEL_PATH}. Please update Config.SAVED_MODEL_PATH.")
#     exit() # Exit if model weights are not found
# except Exception as e:
#     print(f"Error loading model state dict: {e}")
#     exit() # Exit on other loading errors

# model.to(Config.DEVICE)
# model.eval() # Set model to evaluation mode

# --- Prepare Test Data Segments ---
print("Preparing test data segments...")
test_audio_files = list(pathlib.Path(Config.TEST_SOUNDSCAPES_DIR).glob("*.ogg"))
all_test_segments = []

for file_path in tqdm(test_audio_files, desc="Calculating test segments"):
    filename = file_path.name
    soundscape_id = pathlib.Path(filename).stem.split('_')[-1] # Extract ID from soundscape_xxxxxx.ogg
    try:
        info = torchaudio.info(str(file_path))
        file_duration = info.num_frames / info.sample_rate

        # Calculate segment start times (0, 5, 10, ..., 55 for a 60s file)
        # The last segment might be shorter if the file duration is not a multiple of DURATION_SECONDS
        # Kaggle usually evaluates 5s segments starting at 0, 5, 10...
        segment_starts = np.arange(0, file_duration, Config.DURATION_SECONDS)

        for start_time in segment_starts:
            # Ensure we don't create segments starting exactly at the end of the file
            if start_time < file_duration:
                all_test_segments.append((filename, start_time, soundscape_id))

    except Exception as e:
        print(f"Could not process {filename} for segmentation: {e}")
        # Optionally skip this file or add dummy segments if needed

print(f"Generated {len(all_test_segments)} segments for inference.")

# Create Test Dataset and DataLoader
test_dataset = TestSoundscapesDataset(all_test_segments, Config.TEST_SOUNDSCAPES_DIR, Config)
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.INFERENCE_BATCH_SIZE,
    shuffle=False, # Maintain order
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)

print(f"Created Test DataLoader with {len(test_loader)} batches.")

# --- Run Inference ---
print("Running inference...")
# Lists to store results before creating DataFrame
all_row_ids = []
all_probability_vectors = [] # Store the full probability vector for each segment

with torch.no_grad(): # Disable gradient calculations for inference
    test_loop = tqdm(test_loader, desc="Inference", leave=True) # Set leave=True to show final bar
    for images, batch_row_ids in test_loop:
        images = images.to(Config.DEVICE)

        # Get predictions (logits)
        outputs = model(images)

        # Apply sigmoid to get probabilities (0-1)
        probabilities = torch.sigmoid(outputs) # Shape: [batch_size, N_CLASSES]

        # Store row IDs and probability vectors
        all_row_ids.extend(batch_row_ids)
        all_probability_vectors.append(probabilities.cpu().numpy())

print("Inference complete.")

# Concatenate all probability vectors from batches
if all_probability_vectors:
    all_probability_vectors = np.concatenate(all_probability_vectors, axis=0)
    print(f"Combined probability vectors shape: {all_probability_vectors.shape}")
else:
    print("No segments were processed. Probability matrix is empty.")
    all_probability_vectors = np.empty((0, Config.N_CLASSES)) # Handle empty case

# --- Create Submission DataFrame ---
print(f"Creating submission file: {Config.SUBMISSION_CSV_PATH}")

# Initialize the submission DataFrame with required columns
submission_df = pd.DataFrame(index=range(len(all_row_ids)), columns=['row_id'] + submission_species_cols)

# Populate the row_id column
submission_df['row_id'] = all_row_ids

# Populate the species probability columns
for species_id in submission_species_cols:
    if species_id in species_id_to_encoder_idx:
        # Get the index this species corresponds to in the model's output tensor
        encoder_idx = species_id_to_encoder_idx[species_id]
        # Assign the probabilities from the collected vectors
        submission_df[species_id] = all_probability_vectors[:, encoder_idx]
    else:
        # If a species from sample submission wasn't in the training encoder, predict 0 probability
        print(f"Warning: Filling column '{species_id}' with zeros as it was not found in the training label encoder.")
        submission_df[species_id] = 0.0 # Or 0.5, or some other default

# --- Ensure all required row_ids from sample_submission are present ---
# This is a critical step to avoid submission errors.
# Create a DataFrame from our results
our_results_df = submission_df.copy()

# Merge our results with the sample submission row_ids
# This keeps all rows from sample_submission and fills in probabilities where we have them
final_submission_df = sample_submission_df[['row_id']].merge(our_results_df, on='row_id', how='left')

# Fill any row_ids that were in sample_submission but not in our generated segments
# This might happen if there was an error processing a file, or if the segmentation logic slightly differs.
# Fill probability columns with 0.0 for missing rows/species
# The merge will add columns from our_results_df. We only need to fill the species columns.
for col in submission_species_cols:
     if col in final_submission_df.columns:
        final_submission_df[col] = final_submission_df[col].fillna(0.0) # Fill with 0.0 probability
     else:
        # This should not happen if our_results_df was created correctly based on sample_submission_cols
        final_submission_df[col] = 0.0
        print(f"Error: Column '{col}' missing in merged DataFrame. Adding with zeros.")

# The 'row_id' column from the merge should not have NaNs if sample_submission was loaded correctly.

print(f"Final submission DataFrame shape: {final_submission_df.shape}")
print(f"Final submission columns: {final_submission_df.columns.tolist()}")

# Save the submission file
final_submission_df.to_csv(Config.SUBMISSION_CSV_PATH, index=False)

print("Submission file created successfully!")




