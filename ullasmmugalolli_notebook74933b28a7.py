# ==============================================================================
# CELL 1: SETUP FOR KAGGLE - Run this cell first!
# ==============================================================================

# --- Necessary Imports ---
import torch, torch.nn as nn, torch.nn.functional as F, torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
import os, pandas as pd, shutil
from PIL import Image
from tqdm.notebook import tqdm

print("All necessary libraries imported.")

# ------------------------------------------------------------------------------
# --- Path Definitions for Kaggle Environment ---
# ------------------------------------------------------------------------------
FULL_DATA_PATH = '/kaggle/input/bms-molecular-translation'
FULL_IMAGES_FOLDER = os.path.join(FULL_DATA_PATH, 'train')
FULL_LABELS_FILE = os.path.join(FULL_DATA_PATH, 'train_labels.csv')
SUBSET_PATH = '/kaggle/working/dataset_subset'
SUBSET_IMAGES_FOLDER = os.path.join(SUBSET_PATH, 'images')
SUBSET_LABELS_FILE = os.path.join(SUBSET_PATH, 'labels.csv')
SAMPLE_SIZE = 30000

# ------------------------------------------------------------------------------
# --- Create the Dataset Subset ---
# ------------------------------------------------------------------------------
print("Creating dataset subset...")
os.makedirs(SUBSET_IMAGES_FOLDER, exist_ok=True)
full_labels_df = pd.read_csv(FULL_LABELS_FILE)
subset_df = full_labels_df.sample(n=SAMPLE_SIZE, random_state=42)
subset_df.to_csv(SUBSET_LABELS_FILE, index=False)

print(f"Copying {SAMPLE_SIZE} images... (This may take a few minutes)")
for i, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
    image_id = row['image_id']
    filename = f"{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png"
    source_path = os.path.join(FULL_IMAGES_FOLDER, filename)
    destination_path = os.path.join(SUBSET_IMAGES_FOLDER, f"{image_id}.png")
    shutil.copy(source_path, destination_path)
    
print("✅ Dataset subset is ready.")

# ------------------------------------------------------------------------------
# --- Class Definitions (Corrected Versions) ---
# ------------------------------------------------------------------------------
class Vocabulary:
    def __init__(self):
        self.char2idx, self.idx2char = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}, {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
    def __len__(self): return len(self.char2idx)
    def build_vocab(self, text_series):
        chars = set(''.join(text_series))
        for char in sorted(list(chars)):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char], self.idx2char[idx] = idx, char

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        for param in resnet.parameters(): param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
    def forward(self, images):
        features = self.resnet(images)
        batch_size, num_channels = features.size(0), features.size(1)
        return features.view(batch_size, num_channels, -1).permute(0, 2, 1)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att, self.decoder_att = nn.Linear(encoder_dim, attention_dim), nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu, self.softmax = nn.ReLU(), nn.Softmax(dim=1)
    def forward(self, encoder_out, decoder_hidden):
        att1, att2 = self.encoder_att(encoder_out), self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        return (encoder_out * alpha.unsqueeze(2)).sum(dim=1), alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, encoder_dim=512, attention_dim=512):
        super(DecoderRNN, self).__init__()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.init_h, self.init_c = nn.Linear(encoder_dim, hidden_size), nn.Linear(encoder_dim, hidden_size)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)
    def forward(self, features, captions, lengths):
        batch_size, vocab_size = features.size(0), self.vocab_size
        predictions = torch.zeros(batch_size, max(lengths), vocab_size).to(features.device)
        embeddings, (h, c) = self.embedding(captions), self.init_hidden_state(features)
        for t in range(max(lengths)):
            batch_size_t = sum(l > t for l in lengths)
            attention_weighted_encoding, _ = self.attention(features[:batch_size_t], h[:batch_size_t])
            lstm_input = torch.cat((embeddings[:batch_size_t, t, :], attention_weighted_encoding), dim=1)
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            predictions[:batch_size_t, t, :] = self.fcn(h)
        return predictions

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, encoder_dim=512):
        super(EncoderDecoder, self).__init__()
        self.encoder, self.decoder = EncoderCNN(), DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, encoder_dim)
    def forward(self, images, captions, lengths):
        return self.decoder(self.encoder(images), captions, lengths)

print("✅ All classes are defined and ready to use!")

