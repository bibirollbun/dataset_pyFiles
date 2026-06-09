# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import joblib
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
df_test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')


df_train.describe()


pd.set_option('display.max_rows', 400)


# missing_data = df_train.isnull().sum()
# missing_data


df_train.columns


scaler = StandardScaler()
le = LabelEncoder()


df_train['gesture_encoded'] = le.fit_transform(df_train['gesture'])



df_updated_train = df_train.drop(['row_id', 'sequence_type', 'sequence_counter', 'subject',
       'orientation', 'behavior', 'phase', 'gesture'], axis=1)

#df_updated_test = df_test.drop(['row_id', 'sequence_type', 'subject'], axis=1)


feature_cols = df_updated_train.drop(['sequence_id', 'gesture_encoded'], axis=1).columns
df_training = df_updated_train[feature_cols].ffill().bfill().fillna(0)
#df_testing = df_updated_test[feature_cols].ffill().bfill().fillna(0)

df_training.head()



lst_cols = list(feature_cols)


X_scaled_train = scaler.fit_transform(df_training)
#X_scaled_test = scaler.transform(df_testing)
X_scaled_train[0:10]


df_scaled_train = pd.DataFrame(X_scaled_train, columns=feature_cols)
df_scaled_train[['sequence_id', 'gesture_encoded']] = df_updated_train[['sequence_id', 'gesture_encoded']]
#df_scaled_test = pd.DataFrame(X_scaled_test, columns=feature_cols)
#df_scaled_test[['sequence_id', 'gesture_encoded']] = df_updated_test[['sequence_id', 'gesture_encoded']]


df_scaled_train.head()


sequence_lengths = df_scaled_train.groupby('sequence_id').size()
print(sequence_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


plt.figure(figsize=(10, 6))
sns.histplot(sequence_lengths, bins=100, kde=True)
plt.title("Distribution of Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE = 1e-3
SEQUENCE_LENGTH = 150
INPUT_DIM = len(feature_cols)
OUTPUT_DIM = df_scaled_train['gesture_encoded'].nunique()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


gesture_map = {label: idx for idx, label in enumerate(le.classes_)}
gesture_map


class BFRDDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.df = df
        self.sequence_ids = df['sequence_id'].unique()
        self.sequence_len = sequence_length
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.sequence_ids)

    def pad_or_truncate(self, X):
        padded = np.zeros((self.sequence_len, X.shape[1]))
        length = min(self.sequence_len, len(X))
        padded[:length] = X[:length]
        return padded

    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        sequence_df = self.df[self.df['sequence_id']==sequence_id]
        X = sequence_df[self.feature_cols].to_numpy()
        X = self.pad_or_truncate(X)

        gesture = sequence_df['gesture_encoded'].iloc[0]
#        target = self.gesture_map[gesture]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(gesture, dtype=torch.long)
        
        


class_weights = compute_class_weight('balanced', classes=np.unique(df_scaled_train['gesture_encoded']), y=df_scaled_train['gesture_encoded'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


training_data = BFRDDataset(df_scaled_train, SEQUENCE_LENGTH)
#testing_data = BFRDDataset(df_scaled_test, SEQUENCE_LENGTH)


train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(testing_data, batch_size=1)


feature_cols.shape


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights*lstm_out, dim=1)
        return context, weights


class Model_Architecture(nn.Module):
    def __init__(self, input_dim=332, hidden_dim=128, output_dim=18, num_layers=3, sequence_len=200):
        super(Model_Architecture, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.sequence_len = sequence_len
        self.num_layers=num_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2, batch_first=True, device=device)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.attention = Attention(hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.ln(out)
        out, weights = self.attention(out)
        out = self.fc(out)
        #out = self.fc(out[:, -1, :])
        return out
        


target_gesture_names = ['Cheek - pinch skin', 'Forehead - pull hairline', 'Neck - scratch',
       'Neck - pinch skin', 'Eyelash - pull hair', 'Eyebrow - pull hair',
       'Forehead - scratch', 'Above ear - pull hair']


def evaluate_bfrb_f1(y_true_encoded, y_pred_encoded, label_encoder, target_gestures):
    """
    Evaluates Binary F1 and Macro F1 as per the BFRB competition rules.
    
    Parameters:
    - y_true_encoded: array-like of encoded true gesture labels
    - y_pred_encoded: array-like of encoded predicted gesture labels
    - label_encoder: fitted sklearn.preprocessing.LabelEncoder
    - target_gestures: list of gesture names considered as target gestures (BFRB)
    
    Returns:
    - A dictionary with binary_f1, macro_f1, and final_score
    """
    # Decode labels back to gesture names
    y_true = label_encoder.inverse_transform(y_true_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Convert to binary: 'target' vs 'non_target'
    y_true_binary = ['target' if gesture in target_gestures else 'non_target' for gesture in y_true]
    y_pred_binary = ['target' if gesture in target_gestures else 'non_target' for gesture in y_pred]

    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label='target')

    # Collapse all non-target gestures into 'non_target' for macro F1
    y_true_collapsed = [gesture if gesture in target_gestures else 'non_target' for gesture in y_true]
    y_pred_collapsed = [gesture if gesture in target_gestures else 'non_target' for gesture in y_pred]

    macro_f1 = f1_score(y_true_collapsed, y_pred_collapsed, average='macro')

    final_score = (binary_f1 + macro_f1) / 2

    return {
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'final_score': final_score
    }    


def model_training(model, optimizer, criterion, epochs=50):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average='macro')
        # model.eval()
        # test_loss = 0
        # test_preds = []
        # test_labels = []
        # with torch.no_grad():
        #     for X_batch, y_batch in test_loader:
        #         X_batch, y_batch = features.to(device), labels.to(device)
        #         outputs = model(X_batch)
        #         loss = criterion(outputs, y_batch)
        #         test_loss += loss.item()
        #         predicted = torch.argmax(outputs, dim=1)
        #         test_preds.extend(predicted.cpu().numpy())
        #         test_labels.extend(y_batch.cpu().numpy())
            
        #     test_f1 = f1_score(all_targets, all_preds, average='macro')
        
        custom_f1 = evaluate_bfrb_f1(all_targets, all_preds, le, target_gesture_names)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Macro F1: {f1:.4f}")
        print(f"Binary F1: {custom_f1['binary_f1']:.4f}")
        print(f"Macro F1: {custom_f1['macro_f1']:.4f}")
        print(f"Final Score: {custom_f1['final_score']:.4f}")





df_train['gesture_encoded'].nunique()


model = Model_Architecture(input_dim=332, hidden_dim=256, output_dim=18, num_layers=3, sequence_len=SEQUENCE_LENGTH).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



model_training(model, optimizer, criterion, epochs=EPOCHS)


torch.save(model.state_dict(), "model_weights.pth")
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')


