
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import librosa
import librosa.display
import seaborn as sns
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Игнорируем предупреждения
# from warnings import filterwarnings
# filterwarnings("ignore")


print(librosa.__version__)


train_csv_file_path = '/kaggle/input/freesound-audio-tagging/train.csv'
train_audio_dir_path = '/kaggle/input/freesound-audio-tagging/audio_train'
test_csv_file_path = '/kaggle/input/freesound-audio-tagging/test_post_competition.csv'
test_audio_dir_path = '/kaggle/input/freesound-audio-tagging/audio_test'


def get_spectrogram(y, sr):
    X = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.amplitude_to_db(np.abs(X))

def plot_spectrogram(spec_array, title=''):
    plt.figure(figsize=(14, 3))
    librosa.display.specshow(spec_array, x_axis='time', y_axis='linear')
    plt.colorbar(format="%+2.f dB")
    plt.title(title + ' Спектрограмма. Линейный масштаб')
    plt.ylabel("Частота (Гц)")
    plt.xlabel("Время (сек.)")
    plt.show()


# n_frames_list = []
# for filename in tqdm(os.listdir(train_audio_dir_path), desc='Подсчет n_frames'):
#     file_path = os.path.join(train_audio_dir_path, filename)
#     if not os.path.isfile(file_path):
#         continue
#
#     data, sr = librosa.load(file_path)
#     sp = get_spectrogram(data, sr)
#     n_frames_list.append(sp.shape[1])
#
# plt.hist(n_frames_list, bins=300, color='blue', alpha=0.7)
# plt.title('Гистограмма для n_frames')
# plt.xlabel('Число фреймов')
# plt.ylabel('Количество файлов')
# plt.grid(True)
# plt.show()



# import statistics
#
# statistics.median([i[1]for i in n_frames_list])


spec_height, spec_width = 128, 175


def calc_spectrograms(dir_path: str, save: bool = False, save_filename: str = 'spectrograms.npy'):
    spectrograms = np.zeros((len(os.listdir(dir_path)), spec_height, spec_width))

    for i, filename in tqdm(enumerate(os.listdir(dir_path)), desc='Построение спектрограмм'):
        file_path = os.path.join(dir_path, filename)
        if not os.path.isfile(file_path):
            continue
    
        data, sr = librosa.load(file_path)
        spec = get_spectrogram(data, sr)
        min_height = min(spectrograms.shape[1], spec.shape[0])
        min_width = min(spectrograms.shape[2], spec.shape[1])
        spectrograms[i][:min_height, :min_width] = spec[:min_height, :min_width]

    if save:
        np.save(save_filename, spectrograms)

    return spectrograms


spectrograms = None
spec_dataset_filename = '/kaggle/input/spectrograms/spectrograms.npy'
spec_filename = 'spectrograms.npy'

if os.path.isfile(spec_dataset_filename):
    spectrograms = np.load(spec_dataset_filename)
elif os.path.isfile(spec_filename):
    spectrograms = np.load(spec_filename)
else:
    spectrograms = calc_spectrograms(
        dir_path=train_audio_dir_path,
        save=True,
        save_filename=spec_filename
    )


train_df = pd.read_csv(train_csv_file_path)
label_dict = {k: v for k, v in zip(train_df['fname'], train_df['label'])}
label_set = sorted(set(train_df['label']))
label_to_idx = {label: idx for idx, label in enumerate(label_set)}


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SpectrogramDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs.astype('float32')
        self.labels = labels

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        x = self.specs[idx]
        x = torch.from_numpy(x).unsqueeze(0)
        y = self.labels[idx]
        return x, y


import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 32 * 43, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (batch, 16, 128, 175)
        x = self.pool(x)              # (batch, 16, 64, 87)
        x = self.relu(self.conv2(x))  # (batch, 32, 64, 87)
        x = self.pool(x)              # (batch, 32, 32, 43)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


labels = [
    label_to_idx[label_dict[filename]]
    for filename in os.listdir(train_audio_dir_path)
]
labels = torch.tensor(labels)

dataset = SpectrogramDataset(spectrograms, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

print("Обучение завершено.")


model_path = 'simple_cnn_model.pth'
torch.save(model.state_dict(), model_path)


test_spectrograms = None
test_spec_dataset_filename = '/kaggle/input/spectrograms/test_spectrograms.npy'
test_spec_filename = 'test_spectrograms.npy'

if os.path.isfile(test_spec_dataset_filename):
    test_spectrograms = np.load(test_spec_dataset_filename)
elif os.path.isfile(test_spec_filename):
    test_spectrograms = np.load(test_spec_filename)
else:
    test_spectrograms = calc_spectrograms(
        dir_path=test_audio_dir_path,
        save=True,
        save_filename=test_spec_filename
    )


test_df = pd.read_csv(test_csv_file_path)
test_label_dict = {k: v for k, v in zip(test_df['fname'], test_df['label'])}
test_label_set = sorted(set(train_df['label']))
test_label_to_idx = {label: idx for idx, label in enumerate(test_label_set)}
idx_to_label = {idx: label for label, idx in test_label_to_idx.items()}
labels_array = [idx_to_label[i] for i in range(len(test_label_to_idx))]


model_path = '/kaggle/input/audio-cnn/pytorch/default/1/simple_cnn_model.pth'
model_loaded = SimpleCNN(num_classes=len(label_set))
model_loaded.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_loaded.to('cpu')
model_loaded.eval()



model_loaded.eval()
correct = 0
total = 0
out_labels = [0] * len(test_spectrograms)
with torch.no_grad():
    for i, spec in enumerate(test_spectrograms):
        spec_tensor = torch.from_numpy(spec.astype('float32')).unsqueeze(0).unsqueeze(0)
        out = model_loaded(spec_tensor)
        _, max_prob_idx = torch.max(out, 1)
        out_labels[i] = labels_array[max_prob_idx.item()]


out_filenames = [filename for filename in os.listdir(test_audio_dir_path)]
print(len(out_filenames), len(out_labels))
out_df = pd.DataFrame({
    'fname': out_filenames,
    'label': out_labels
})


out_df.to_csv('submission.csv', index=False)

