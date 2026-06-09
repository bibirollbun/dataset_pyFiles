# Comment modules import
import os
import numpy as np
import pandas as pd
from pathlib import Path
import io
import sys

# Display import
# from IPython.display import Image, display
from PIL import Image
import random


# Mel-Spectrogram processing import
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm


# Model import
import torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader


# Label encoding
from sklearn.preprocessing import LabelEncoder


# Gabage collector for optimizing memories
import gc


import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# input_dir = '/kaggle/input/birdclef-2025/train_audio'

# print("Folders in train_audio:", os.listdir(input_dir)[:5])


# def audio_to_melspectrogram(file_path, save_path):
#     try:
#         y, sr = librosa.load(file_path, sr=None)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         S_DB = librosa.power_to_db(S, ref=np.max)

#         plt.figure(figsize=(2.56, 2.56), dpi=100)
#         librosa.display.specshow(S_DB, sr=sr, cmap='magma')
#         plt.axis('off')
#         plt.tight_layout(pad=0)
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         plt.close()
#     except Exception as e:
#         print(f"âš ï¸� Error on {file_path}: {e}")

# # Limit number of files to avoid long runtime (e.g., for Starter demonstration)
# file_list = []
# for root, _, files in os.walk(input_dir):
#     for file in files:
#         if file.endswith('.ogg'):
#             full_path = os.path.join(root, file)
#             file_list.append(full_path)

# # Only use first 50 files for demo purposes
# file_list = file_list[21000:22000]
# print(f"Number of files used for conversion: {len(file_list)}")

# output_dir = '/kaggle/working/train_images'
# os.makedirs(output_dir, exist_ok=True)

# for input_path in tqdm(file_list):
#     base_name = os.path.basename(input_path).replace('.ogg', '.png')
#     output_path = os.path.join(output_dir, base_name)
#     if not os.path.exists(output_path):
#         audio_to_melspectrogram(input_path, output_path)


# image_folder = '/kaggle/input/processed-audio-file/train_images'
# image_files = os.listdir(image_folder)
# sample_image_path = os.path.join(image_folder, image_files[0])  
# display(Image(filename=sample_image_path))


taxonomy_path = '/kaggle/input/birdclef-2025/taxonomy.csv'
taxonomy_df = pd.read_csv(taxonomy_path)
species_ids = taxonomy_df['primary_label'].tolist()
# labels = taxonomy_df['primary_label'].tolist()


# encoder = LabelEncoder()
# encoded_labels = encoder.fit_transform(labels)

# for idx in encoded_labels:
#     encoded_labels
# print("Encoded Labels:", encoded_labels)


# # Set up a basic transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    transforms.Lambda(lambda x: torch.clamp(x, min=0.0)),
    transforms.Lambda(lambda x: torch.clamp(x, max=1.0))
])


# # Define a dummy dataset using 10 random images
# class SpectrogramDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.image_files = os.listdir(image_dir)
#         # self.image_files = random.sample(self.image_files,1000)  # select 10 files
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         image = Image.open(img_path).convert("RGB")

#         index = 0
#         if self.transform:
#             image = self.transform(image)

#         one_hot_tensor = torch.zeros(206)
#         finder = self.image_files[idx].replace('.png', '.ogg')
#         audio_folders_path = '/kaggle/input/birdclef-2025/train_audio'
#         audio_folders = [f for f in os.listdir(audio_folders_path) if os.path.isdir(os.path.join(audio_folders_path, f))]
#         for bird_id in audio_folders:
#             bird_id_audio_path = audio_folders_path + "/" + bird_id
#             file_list = [f for f in os.listdir(bird_id_audio_path) if os.path.isfile(os.path.join(bird_id_audio_path, f))]
#             if finder in file_list:
#                 index = list(encoder.classes_).index(bird_id)
#                 break
        
#         one_hot_tensor[index] = 1.0
#         label = one_hot_tensor 
#         return image, label

# dataset = SpectrogramDataset('/kaggle/input/processed-audio-file/train_images', transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Load pretrained ResNet18 and modify the output layer
# model = models.resnet18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, 206)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define optimizer and loss
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



# # Training loop (100 epochs)
# model.train()
# for epoch in range(50):
#     total_loss = 0.0
#     for images, labels in dataloader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# torch.save(model.state_dict(), 'model1.pth')


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 206)
model.load_state_dict(torch.load('/kaggle/input/resnet18_001/pytorch/debug_model/1/model1.pth'))
model.eval()


def predict_on_spectrogram(audio_path):
    """
    Process a single audio file and predict species presence for each 5-second segment.
    
    :param audio_path: Path to the audio file.
    :return: Tuple of (row_id, predictions) for each segment.
    """
    FS = 32000  # Sampling frequency (32kHz)
    WINDOW_SIZE = 5  # Window size (in seconds)
    
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}...")
        audio_data, sr = librosa.load(audio_path, sr=FS)

        if len(audio_data) < FS * WINDOW_SIZE:
            audio_data = np.pad(
                audio_data,
                (0, FS * WINDOW_SIZE - len(audio_data)),
                mode='constant'
            )
        
        segment_samples = int(WINDOW_SIZE * sr)
        total_segments = int(len(audio_data) / (FS * WINDOW_SIZE))

        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * FS * WINDOW_SIZE
            end_sample = start_sample + FS * WINDOW_SIZE
            end_time_sec = (segment_idx + 1) * WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
            
            segment_audio = audio_data[start_sample:end_sample]
            S = librosa.feature.melspectrogram(y=segment_audio, sr=sr, n_mels=128, fmax=FS)
            S_DB = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=(2.56, 2.56), dpi=20)
            librosa.display.specshow(S_DB, sr=sr, cmap='magma')
            plt.axis('off')
            plt.tight_layout(pad=0)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to('cpu')
            img = None
            buf.close()

            S = None
            S_DB = None
            gc.collect()
            
            final_preds = []
            # if len(models) == 1:
            #     with torch.no_grad():
            #         outputs = models[0](img_tensor)
            #         final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
            # else:
            #     segment_preds = []
            #     for model in models:
            #         with torch.no_grad():
            #             outputs = model(img_tensor)
            #             probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            #             segment_preds.append(probs)
            #     final_preds = np.mean(segment_preds, axis=0) 
            with torch.no_grad():
                outputs = model(img_tensor)
                final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
            predictions.append(final_preds)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions


def run_inference():
        """
        Perform inference on all test soundscape audio files.
        
        :return: Tuple of (row_ids, predictions) aggregated from all files.
        """
        test_soundscapes_path = '/kaggle/input/birdclef-2025/test_soundscapes'
        debug = '/kaggle/input/birdclef-2025/train_audio/65373'
        test_files = list(Path(test_soundscapes_path).glob('*.ogg'))  # Get the test soundscape file
        print(f"Found {len(test_files)} test soundscape files.")

        all_row_ids = []
        all_predictions = []

        for audio_path in tqdm(test_files, disable = True):
            row_ids, predictions = predict_on_spectrogram(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions



def create_submission(row_ids, predictions):
        """
        Create a submission DataFrame based on prediction results.
        
        :param row_ids: List of row identifiers for each segment.
        :param predictions: List of prediction arrays for each segment.
        :return: Submission formatted pandas DataFrame.
        """
        print("Creating submission DataFrame...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        submission_csv_path = '/kaggle/input/birdclef-2025/sample_submission.csv'
        sample_sub = pd.read_csv(submission_csv_path, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} species are missing in the submission.")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns]  # Align columns with the sample submission
        submission_df = submission_df.reset_index()
        
        return submission_df


def smooth_submission(submission_path):
    """
    Smooth the predictions in the submission file to maintain temporal consistency.
    
    :param submission_path: Path to the submission CSV file.
    """
    print("Smoothing predictions in submission file...")
    sub = pd.read_csv(submission_path)
    cols = sub.columns[1:]
    # Extract group based on 'row_id'
    groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        sub_group = sub.iloc[idx].copy()
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        
        if predictions.shape[0] > 1:
            # Smooth by averaging predictions with adjacent segments
            new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
            new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
            for i in range(1, predictions.shape[0]-1):
                new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
        sub.iloc[idx, 1:] = new_predictions
    
    sub.to_csv(submission_path, index=False)
    sub = None
    gc.collect()
    print(f"Smoothed submission saved at {submission_path}")


row_ids, predictions = run_inference()


# print(row_ids)
# print(predictions)


submission_df = create_submission(row_ids, predictions)

submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
submission_df = None
gc.collect()
smooth_submission(submission_path)


# import shutil

# # NÃ©n toÃ n bá»™ thÆ° má»¥c working thÃ nh file ZIP
# shutil.make_archive('/kaggle/working/working_folder_backup', 'zip', '/kaggle/working')

# print("Folder has been zipped. You can now download it.")


# import os
# import psutil
# import numpy as np

# def cpu_stats():
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     memory_use = py.memory_info()[0] / 2. ** 30  # Chuyá»ƒn Ä‘á»•i sang GB
#     return 'memory GB:' + str(np.round(memory_use, 2))

# print(cpu_stats())

