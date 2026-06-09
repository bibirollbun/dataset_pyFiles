import shutil
import os
#Копирование и установка ResNeSt
shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True)
os.system('pip install "./resnet" --no-deps')


import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt


#Глобальные константы
AUDIO_SAMPLE_RATE = 32000
SEGMENT_DURATION = 5  #секунд
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "../input/birdclef-2021"
TEST_AUDIO_PATH = os.path.join(DATA_ROOT, "test_soundscapes")
MODEL_WEIGHTS_FILE = "../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"

print(f"Device: {DEVICE}")


#Загрузка метаданных и кодирование классов
train_meta = pd.read_csv(os.path.join(DATA_ROOT, "train_metadata.csv"))
species_list = sorted(train_meta["primary_label"].unique())
label_encoder = LabelEncoder().fit(species_list)
NUM_SPECIES = len(species_list)
print(f"Total species: {NUM_SPECIES}")


species_list[0:20:2]


#Преобразование аудио в нормализованную мел-спектрограмму
def transform_audio_to_spec(signal, sr=AUDIO_SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=128, fmin=0, fmax=sr//2,
        n_fft=sr//10, hop_length=sr//40
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    normalized = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    min_val, max_val = normalized.min(), normalized.max()
    if max_val - min_val > 1e-6:
        scaled = 255 * (normalized - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(normalized)
    return scaled.astype(np.uint8)

#Преобразование спектрограммы в 3-канальный тензор
def make_rgb_tensor(spec_img):
    return np.stack([spec_img] * 3, axis=0).astype(np.float32) / 255.0


#датасет для сегментов аудио
class AudioSegmentDataset(Dataset):
    def __init__(self, annotations, audio_folder, sr=AUDIO_SAMPLE_RATE, seg_len_sec=5):
        self.annotations = annotations.reset_index(drop=True)
        self.folder = audio_folder
        self.sr = sr
        self.seg_len = seg_len_sec
        self._audio_cache = {}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        row_id = row["row_id"]
        file_prefix, _, end_time = row_id.rsplit("_", 2)
        full_prefix = "_".join(row_id.split("_")[:2])

        try:
            if full_prefix not in self._audio_cache:
                audio_filename = next(f for f in os.listdir(self.folder) if f.startswith(full_prefix))
                audio_full, orig_sr = librosa.load(os.path.join(self.folder, audio_filename), sr=None, res_type='kaiser_fast')
                if orig_sr != self.sr:
                    audio_full = librosa.resample(audio_full, orig_sr=orig_sr, target_sr=self.sr)
                self._audio_cache[full_prefix] = audio_full
            else:
                audio_full = self._audio_cache[full_prefix]

            start_sample = max(0, (int(end_time) - self.seg_len) * self.sr)
            end_sample = min(len(audio_full), int(end_time) * self.sr)
            segment = audio_full[start_sample:end_sample]

            if len(segment) < self.seg_len * self.sr:
                segment = np.pad(segment, (0, self.seg_len * self.sr - len(segment)))

            spec = transform_audio_to_spec(segment, self.sr)
            tensor = make_rgb_tensor(spec)
            return tensor

        except Exception:
            return np.zeros((3, 128, 313), dtype=np.float32)


#Загрузка модели ResNeSt с весами
def build_inference_model(weight_path, num_classes):
    net = resnest50(pretrained=False)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)

    checkpoint = torch.load(weight_path, map_location="cpu")
    clean_state = {k.replace("model.", ""): v for k, v in checkpoint.items()}
    net.load_state_dict(clean_state)
    net.to(DEVICE)
    net.eval()
    return net


#Функция для инференса модели
def run_inference_on_batch(batch_data, model, thr=0.1):
    with torch.no_grad():
        inputs = torch.from_numpy(batch_data).to(DEVICE)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for prob_vec in probs:
        active_labels = np.where(prob_vec > thr)[0]
        if len(active_labels) == 0:
            results.append("nocall")
        else:
            names = label_encoder.inverse_transform(active_labels)
            results.append(" ".join(sorted(names)))
    return results


#Подготовка тестовых данных
test_meta = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
use_train_as_test = len(test_meta) < 10
audio_source = os.path.join(DATA_ROOT, "train_soundscapes") if use_train_as_test else TEST_AUDIO_PATH
if use_train_as_test:
    test_meta = pd.read_csv(os.path.join(DATA_ROOT, "train_soundscape_labels.csv"))

print(f"Total segments to process: {len(test_meta)}")


#Загрузка данных и инференс модели
audio_dataset = AudioSegmentDataset(test_meta, audio_source)
loader = DataLoader(audio_dataset, batch_size=64, shuffle=False, num_workers=0)

final_predictions = []
for batch in tqdm(loader, desc="Running inference"):
    stacked_batch = np.stack([b.numpy() for b in batch])
    batch_preds = run_inference_on_batch(stacked_batch, build_inference_model(MODEL_WEIGHTS_FILE, NUM_SPECIES), thr=0.1)
    final_predictions.extend(batch_preds)


from IPython.display import Audio, display

#Несколько примеров для визуализации
detected_examples = [i for i, p in enumerate(final_predictions) if p != "nocall"][-10:]

if detected_examples:
    for idx, i in enumerate(detected_examples, 1):
        row_id = test_meta.iloc[i]["row_id"]
        pred_label = final_predictions[i]
        print(f"\n{'='*60}")
        print(f"Example {idx} | Row ID: {row_id}")
        print(f"Predicted birds: '{pred_label}'")
        print(f"{'='*60}")

        #Загрузка аудио
        prefix_key = "_".join(row_id.split("_")[:2])
        end_sec = int(row_id.split("_")[-1])

        audio_file = next(f for f in os.listdir(audio_source) if f.startswith(prefix_key))
        full_audio, sr_orig = librosa.load(os.path.join(audio_source, audio_file), sr=None)
        if sr_orig != AUDIO_SAMPLE_RATE:
            full_audio = librosa.resample(full_audio, orig_sr=sr_orig, target_sr=AUDIO_SAMPLE_RATE)

        start_samp = max(0, (end_sec - SEGMENT_DURATION) * AUDIO_SAMPLE_RATE)
        end_samp = min(len(full_audio), end_sec * AUDIO_SAMPLE_RATE)
        audio_clip = full_audio[start_samp:end_samp]
        if len(audio_clip) < SEGMENT_DURATION * AUDIO_SAMPLE_RATE:
            audio_clip = np.pad(audio_clip, (0, SEGMENT_DURATION * AUDIO_SAMPLE_RATE - len(audio_clip)))

        display(Audio(audio_clip, rate=AUDIO_SAMPLE_RATE))

        #waveform и спектрограмма
        fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 2]})

        #waveform
        time_wave = np.linspace(0, SEGMENT_DURATION, len(audio_clip))
        axs[0].plot(time_wave, audio_clip, color='steelblue', linewidth=0.8)
        axs[0].set_title("Waveform", fontsize=11)
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlim(0, SEGMENT_DURATION)
        axs[0].grid(True, linestyle='--', alpha=0.5)

        #спектрограмма
        spec = audio_dataset[i][0] 
        im = axs[1].imshow(
            spec,
            aspect='auto',
            origin='lower',
            cmap='viridis', 
            extent=[0, SEGMENT_DURATION, 0, AUDIO_SAMPLE_RATE // 2 // 1000]  # ось частот в кГц
        )
        axs[1].set_title("Mel-spectrogram", fontsize=11)
        axs[1].set_xlabel("Время (секунды)")
        axs[1].set_ylabel("Частота (кГц)")
        plt.colorbar(im, ax=axs[1], shrink=0.6)

        plt.tight_layout()
        plt.show()

else:
    print("Классы не определены.")


#Сохранение результата
submission_df = pd.DataFrame({
    "row_id": test_meta["row_id"],
    "birds": final_predictions
})
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head(10))

