!pip install efficientnet_pytorch


import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio import transforms as T
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import random



INPUT_COMPETITION_DATA_PATH = '../input/itmo-acoustic-event-detectin-2025'
INPUT_MY_DATASET_PATH = '../input/itogi-raboty'

train_meta_fname = 'train.csv'
test_meta_fname = 'sample_submission.csv' 
train_data_folder_relative = 'audio_train/train'
test_data_folder_relative = 'audio_test/test'  

PRETRAINED_MODEL_PATH = os.path.join(INPUT_MY_DATASET_PATH, 'baseline_model.pt')

FINETUNE_LEARNING_RATE = 2e-5
FINETUNE_EPOCHS = 10 
FINETUNE_BATCH_SIZE = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000 
WAV_LEN_SAMPLES = TARGET_SR * 2 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


df_train_full = pd.read_csv(os.path.join(INPUT_COMPETITION_DATA_PATH, train_meta_fname))
df_test_for_submission = pd.read_csv(os.path.join(INPUT_COMPETITION_DATA_PATH, test_meta_fname))

n_classes = df_train_full.label.nunique()
classes_dict = {cl:i for i,cl in enumerate(df_train_full.label.unique())}
idx_to_class_dict = {i:cl for cl, i in classes_dict.items()}
df_train_full['label_encoded'] = df_train_full.label.map(classes_dict)
print(f"Количество классов: {n_classes}")

def add_gaussian_noise_to_waveform(waveform, min_amplitude=0.001, max_amplitude=0.005, p=0.5):
    if torch.rand(1).item() < p:
        noise_amplitude = random.uniform(min_amplitude, max_amplitude)
        noise = torch.randn_like(waveform) * noise_amplitude
        return waveform + noise
    return waveform

class BaseLineModel(nn.Module):
    def __init__(self, sample_rate=TARGET_SR, n_classes=n_classes):
        super().__init__()
        self.ms = T.MelSpectrogram(sample_rate=sample_rate)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3, padding=1)
        
        self.features = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.lin1 = nn.Linear(self.features._fc.in_features, 333)
        self.lin2 = nn.Linear(333, 111)
        self.lin3 = nn.Linear(111, n_classes)
        
    def forward(self, x):
        x = self.ms(x)
        x = self.amplitude_to_db(x)
        
        x_mean = x.mean(dim=(2,3), keepdim=True)
        x_std = x.std(dim=(2,3), keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn3(x))
        
        x = self.features.extract_features(x)
        x = self.features._avg_pooling(x)
        x = x.flatten(start_dim=1)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x, dim=1)
        return x


def sample_or_pad(waveform, wav_len=WAV_LEN_SAMPLES, is_train=True):
    m, n = waveform.shape
    if n < wav_len:
        padded_wav = torch.zeros(1, wav_len, device=waveform.device)
        padded_wav[:, :n] = waveform
        return padded_wav
    elif n > wav_len:
        if is_train: 
            offset = np.random.randint(0, n - wav_len + 1) 
        else: 
            offset = (n - wav_len) // 2 
        sampled_wav = waveform[:, offset:offset+wav_len]
        return sampled_wav
    else:
        return waveform
        
class EventDetectionDataset(Dataset):
    def __init__(self, data_root_path, relative_folder, fnames_list, labels_list=None, 
                 sr=TARGET_SR, wav_len=WAV_LEN_SAMPLES, apply_noise_augmentation=False, is_train_set=False):
        self.data_root_path = data_root_path
        self.relative_folder = relative_folder
        self.fnames = fnames_list
        self.labels = labels_list
        self.sr = sr
        self.wav_len = wav_len
        self.apply_noise_augmentation = apply_noise_augmentation
        self.is_train_set = is_train_set
    
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path2wav = os.path.join(self.data_root_path, self.relative_folder, fname)
        
        try:
            waveform, sample_rate = torchaudio.load(path2wav)
        except Exception as e:
            waveform = torch.zeros((1, self.sr)) 
            sample_rate = self.sr

        if sample_rate != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sr)
        
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.apply_noise_augmentation and self.is_train_set:
            waveform = add_gaussian_noise_to_waveform(waveform, p=0.4) # Вероятность применения шума

        waveform = sample_or_pad(waveform, wav_len=self.wav_len, is_train=self.is_train_set)
        
        if self.labels is not None:
            label = self.labels[idx]
            return waveform, torch.tensor(label, dtype=torch.long)
        return waveform


X_train, X_val, y_train, y_val = train_test_split(
    df_train_full.fname.values, 
    df_train_full.label_encoded.values, 
    test_size=0.15, 
    random_state=42, 
    stratify=df_train_full.label_encoded.values
)

train_dataset = EventDetectionDataset(
    INPUT_COMPETITION_DATA_PATH, train_data_folder_relative, X_train, y_train,
    apply_noise_augmentation=True, is_train_set=True
)
val_dataset = EventDetectionDataset(
    INPUT_COMPETITION_DATA_PATH, train_data_folder_relative, X_val, y_val, 
    is_train_set=False
)

train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


model = BaseLineModel(sample_rate=TARGET_SR, n_classes=n_classes).to(DEVICE)

print(f"Загрузка весов из: {PRETRAINED_MODEL_PATH}")
if os.path.exists(PRETRAINED_MODEL_PATH):
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        print("Веса базовой модели успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки весов базовой модели: {e}. Обучение начнется с весов EfficientNet (ImageNet).")
else:
    print(f"Файл весов базовой модели не найден: {PRETRAINED_MODEL_PATH}. Обучение начнется с весов EfficientNet (ImageNet).")


if hasattr(model, 'features') and isinstance(model.features, EfficientNet):
    for param in model.features.parameters():
        param.requires_grad = False
    print("Слои EfficientNet (model.features) заморожены.")
else:
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось найти 'model.features' для заморозки EfficientNet.")

for name, child_module in model.named_children():
    if name not in ['features', 'ms', 'amplitude_to_db']:
        for param in child_module.parameters():
            param.requires_grad = True
print("Кастомные слои (cnn, linear) установлены как обучаемые.")
            
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=FINETUNE_LEARNING_RATE, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True)


def eval_model(model_to_eval, eval_loader):
    model_to_eval.eval()
    all_forecasts, all_true_labs = [], []
    with torch.no_grad():
        for wavs, labs in tqdm(eval_loader, desc="Evaluating", leave=False):
            wavs, labs = wavs.to(DEVICE), labs.to(DEVICE)
            outputs = model_to_eval(wavs) 
            _, predicted_classes = torch.max(outputs, 1)
            all_forecasts.extend(predicted_classes.cpu().numpy())
            all_true_labs.extend(labs.cpu().numpy())
    return f1_score(all_true_labs, all_forecasts, average='macro', zero_division=0)


best_f1_finetune = 0.0 
print("\n--- Начало дообучения (Fine-tuning) ---")
for epoch in range(FINETUNE_EPOCHS):
    model.train() 
    epoch_train_loss = 0
    
    if hasattr(model, 'features') and isinstance(model.features, EfficientNet):
        for param in model.features.parameters():
            param.requires_grad = False
            
    progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS} [Fine-tune Train]", leave=False)
    for wavs, labs in progress_bar_train:
        wavs, labs = wavs.to(DEVICE), labs.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(wavs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        progress_bar_train.set_postfix(loss=loss.item())

    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    current_f1_val = eval_model(model, val_loader)
    
    print(f"Epoch: {epoch+1}/{FINETUNE_EPOCHS}, Train Loss: {avg_epoch_train_loss:.4f}, Val F1: {current_f1_val:.4f}")
    
    scheduler.step(current_f1_val)

    if current_f1_val > best_f1_finetune:
        best_f1_finetune = current_f1_val
        save_path = f'finetuned_noise_model_best_ep{epoch+1}_f1_{best_f1_finetune:.4f}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Сохранена лучшая дообученная модель: {save_path} с Val F1: {best_f1_finetune:.4f}")

print("--- Дообучение завершено ---")
print(f"Лучший Val F1 после дообучения: {best_f1_finetune:.4f}")


print("\n--- Создание сабмишна с лучшей ДООБУЧЕННОЙ моделью ---")
best_saved_model_path = None
highest_f1_in_filename = -1.0
# Ищем лучший файл в текущей директории (/kaggle/working/)
for f_name in os.listdir('.'): 
    if f_name.startswith('finetuned_noise_model_best_ep') and f_name.endswith('.pt'):
        try:
            f1_from_name = float(f_name.split('_f1_')[-1].replace('.pt',''))
            if f1_from_name > highest_f1_in_filename:
                highest_f1_in_filename = f1_from_name
                best_saved_model_path = f_name
        except:
            continue

if best_saved_model_path and os.path.exists(best_saved_model_path):
    print(f"Загрузка лучшей дообученной модели: {best_saved_model_path}")
    model.load_state_dict(torch.load(best_saved_model_path, map_location=DEVICE))
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ: Файл лучшей дообученной модели не найден. Будут использованы веса последней эпохи дообучения (если обучение было).")

model.eval()
final_forecasts = []
test_dataset_for_submission = EventDetectionDataset(
    INPUT_COMPETITION_DATA_PATH, test_data_folder_relative, df_test_for_submission.fname.values, None, 
    is_train_set=False
)
test_loader_for_submission = DataLoader(
    test_dataset_for_submission, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

with torch.no_grad():
    for wavs in tqdm(test_loader_for_submission, desc="Generating Submission", leave=False):
        wavs = wavs.to(DEVICE)
        outputs = model.inference(wavs) 
        predicted_classes = outputs.argmax(dim=1).cpu().numpy()
        final_forecasts.extend(predicted_classes)

decoded_forecasts = [idx_to_class_dict[idx] for idx in final_forecasts]
df_submission_finetuned = pd.DataFrame({'fname': df_test_for_submission.fname.values, 'label': decoded_forecasts})

# Исправляем формирование имени файла
f1_suffix = f"{best_f1_finetune:.4f}" if best_f1_finetune > 0 else "initial"
submission_filename = f'submission_finetuned_f1_{f1_suffix}.csv'

df_submission_finetuned.to_csv(submission_filename, index=None)
print(f"Файл сабмишна сохранен как: {submission}")




