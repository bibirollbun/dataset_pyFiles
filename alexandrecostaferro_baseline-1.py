import os
import numpy as np # linear algebra
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import glob
import torchaudio
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
        
train_path = "/kaggle/input/deepfake-detection-challenge-pav-2025/train/"
dev_path = "/kaggle/input/deepfake-detection-challenge-pav-2025/dev"
test_path = "/kaggle/input/deepfake-detection-challenge-pav-2025/test/"



class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = ["real", "fake"]
        self.audio_files = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            print(os.path.join(class_dir, "**", "*.flac"))
            for file in glob.glob(os.path.join(class_dir, "**", "*.flac"), recursive=True):
                if file.endswith(".flac"):
                    self.audio_files.append(os.path.join(class_dir, file))
                    self.labels.append(class_idx)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio
        audio, sr = torchaudio.load(audio_file)
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        if sr != TARGET_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(audio)

        # Pad or truncate the audio to a fixed length
        fixed_length = (
            TARGET_SAMPLE_RATE * 3
        )  # Adjust this value based on your requirements
        
        if audio.shape[1] < fixed_length:
            audio = torch.nn.functional.pad(audio, (0, fixed_length - audio.shape[1]))
        else:
            audio = audio[:, :fixed_length]

        audio = self.mel_spectrogram(audio)

        return audio, label


class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_files = []

        for file in os.listdir(data_dir):
            if file.endswith(".flac"):
                self.audio_files.append(os.path.join(data_dir, file)) 

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        audio, sr = torchaudio.load(audio_file)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        if sr != TARGET_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(audio)

        fixed_length = TARGET_SAMPLE_RATE * 3
        if audio.shape[1] < fixed_length:
            audio = torch.nn.functional.pad(audio, (0, fixed_length - audio.shape[1]))
        else:
            audio = audio[:, :fixed_length]

        audio = self.mel_spectrogram(audio)

        return audio, audio_file


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 7, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        x = input_data
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        output = self.log_softmax(x)

        return output


TARGET_SAMPLE_RATE = 16000

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train():
    
    # Usage
    dataset = AudioDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    from torch.optim import Adam

    # Create model, loss function, and optimizer
    model = CNNNetwork()
    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for audio, labels in dataloader:
            audio = audio.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / len(dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )
    torch.save(model.state_dict(), "model.pth")


train()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNNetwork()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
model.to(device)


dev_dataset = AudioDataset(dev_path)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)


results_dev = []

with torch.no_grad():
    for audio, labels in dev_dataloader:

        audio = audio.to(device)
        outputs = model(audio)

        _, preds = torch.max(outputs, 1)

        for i in range(len(outputs)):
            results_dev.append((labels[i].item(), torch.exp(outputs[i][1]).item()))


def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

def ComputeEER(fnrs, fprs, thresholds):
    min_difference = float('inf')
    eer = None
    eer_threshold = None

    # Iterate through all FNR and FPR values to find the minimum difference
    for i in range(len(fnrs)):
        difference = abs(fnrs[i] - fprs[i])
        if difference < min_difference:
            min_difference = difference
            eer = (fnrs[i] + fprs[i]) / 2  # Approximate EER as the average at the point of minimum difference
            eer_threshold = thresholds[i]

    return eer, eer_threshold

labels = [infer[0] for infer in results_dev]
preds = [infer[1] for infer in results_dev]

fnrs, fprs, thresholds = ComputeErrorRates(preds, labels)
eer, eer_threshold = ComputeEER(fnrs, fprs, thresholds)

print(f"EER: {eer}\nEER Treshold: {eer_threshold}")


def plotar_matriz_confusao(y_true, y_pred, figsize=(10, 8)):

    cm = confusion_matrix(y_true, y_pred)
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        
        plt.title('Matriz de Confusão - Detecção de DeepFake', fontsize=16)
        plt.ylabel('Rótulo Verdadeiro', fontsize=12)
        plt.xlabel('Predição do Modelo', fontsize=12)
        
        metricas_texto = (
            f'Acurácia: {accuracy:.4f}\n'
            f'Precisão: {precision:.4f}\n'
            f'Recall: {recall:.4f}\n'
            f'F1-Score: {f1:.4f}'
        )
        
        plt.figtext(0.36, -0.1, metricas_texto, fontsize=12, 
                   bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.show()
        
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    else:
        print("Erro: A matriz de confusão não é 2x2. Verifique os dados.")


discrete_preds = [1 if pred > eer_threshold else 0 for pred in preds]
plotar_matriz_confusao(labels, discrete_preds)


test_dataset = TestDataset(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

results = []

with torch.no_grad():
    for audio, audio_file in test_dataloader:
        audio = audio.to(device)
        outputs = model(audio)

        _, preds = torch.max(outputs, 1)

        for i in range(len(outputs)):
            results.append((audio_file[i], torch.exp(outputs[i][1]).item()))

with open("submission.csv", "w") as f:
    f.write("id,fake_prob\n")
    for result in results:
        f.write("/".join(result[0].split("/")[-2:]) + "," + str(result[1]) + "\n")

