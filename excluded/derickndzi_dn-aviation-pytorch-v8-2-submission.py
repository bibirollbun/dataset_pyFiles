# Misc.
import gc
import importlib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Scikit-Learn
from sklearn.preprocessing import StandardScaler


BATCH_SIZE = 64
MODEL_FEATURES = [
    'eeg_c3', 'eeg_c4', 'eeg_cz', 'eeg_fp1', 'eeg_fp2', 'eeg_f3', 'eeg_f4', 'eeg_f7', 'eeg_f8', 
    'eeg_fz', 'eeg_o1', 'eeg_o2', 'eeg_p3', 'eeg_p4', 'eeg_poz', 'eeg_pz', 'eeg_t3', 'eeg_t4', 
    'eeg_t5', 'eeg_t6', 'ecg', 'r', 'gsr'
]


from IPython.display import HTML

css = """
<style>
    .dataframe td, .dataframe th, .dataframe tfoot td, p {
        color: #cfcfcf !important;  /* Light gray text color */
        background-color: #1e1e1e !important; /* Dark background */
    }
</style>
"""

HTML(css)


print('Loading test dataset...')
test = pd.read_csv('/kaggle/input/aviation-test-augmented-dataset-v7-0/test_augmented_v70.csv')
test


scaler = joblib.load(
    '/kaggle/input/aviation-pytorch-v8.2/pytorch/default/1/aviation_pytorch_scaler_v82.joblib')
scaler


print('Scaling test data.')
test_features = scaler.transform(test[MODEL_FEATURES])
print('Test features shape:', test_features.shape)


scaled_test = pd.concat([
        test[['sequence']].reset_index(drop=True), 
        pd.DataFrame(test_features)
    ],
    axis=1)
scaled_test


del test
gc.collect()


def make_tensors(df):
    sequence_groups = df.groupby('sequence')

    # Convert sequences to a list of tensors
    sequences = []
    
    for _, group in sequence_groups:
        # Extract features
        features = group.drop(columns=['sequence']).values
    
        # Convert to tensors
        sequences.append(torch.tensor(features, dtype=torch.float32))
        
    print(f"Found {len(sequences)} sequences.")

    return sequences


test_sequences = make_tensors(scaled_test)


del scaled_test
gc.collect()


test_padded_inputs = pad_sequence(test_sequences, batch_first=True, padding_value=0.0)
print(f"Padded inputs shape: {test_padded_inputs.shape}")


del test_sequences
gc.collect()


class SequenceDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

test_dataset = SequenceDataset(test_padded_inputs)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


%run '/kaggle/input/aviation-pytorch-v8.2/pytorch/default/1/aviation_pytorch_v82.py'
RNNSequenceClassifier


# Instantiate the model
input_size = len(MODEL_FEATURES)
hidden_size = 256
num_classes = 4
dropout_prob = 0.4
model = RNNSequenceClassifier(input_size, hidden_size, num_classes, dropout_prob)
model.load_state_dict(torch.load(
    '/kaggle/input/aviation-pytorch-v8.2/pytorch/default/1/aviation_pytorch_state_best_model_v82.pth',
    weights_only=True
))
model.eval()


print('Inferencing...')
all_preditions = list()

with torch.no_grad():
    for test_batch_inputs in test_dataloader:
        logits = model(test_batch_inputs)
        logits = logits.view(-1, num_classes)
        probabilities = F.softmax(logits, dim=1)

        # remove the predictions made on padding values
        padding_mask = test_batch_inputs.bool().any(dim=-1).view(-1)
        probabilities = probabilities[padding_mask]
        
        all_preditions.append(probabilities)


preds = pd.DataFrame(np.concatenate(all_preditions), columns=['A', 'B', 'C', 'D'])
preds.index.name = "id"
preds


summary_stats = preds.describe()
with pd.option_context('display.float_format', '{:.6f}'.format):
    display(summary_stats)


print('Saving submission file.')
preds.to_csv('submission.csv', header=True, index=True)

