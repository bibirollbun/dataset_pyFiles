!pip uninstall -y cupy -q
!pip uninstall -y cupy-cuda12x -q
!pip install cupy-cuda11x -q

# https://claude.ai/chat/ca65c790-2155-4ce0-8f7b-d47741ec4f62

# Make sure torch and torchvision versions match
!pip install torch==2.0.1 torchvision==0.15.2 -q
!pip install helical -q


# Try installing helical normally instead of to a custom location
!pip install helical -q --upgrade
#!pip install helical -q # --target=/kaggle/working/ --upgrade #--use-deprecated=legacy-resolver
from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel



import subprocess
import torch
import pandas as pd


# Load data

train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")


from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel


device = "cuda" if torch.cuda.is_available() else "cpu"


# We set the max length to the maximum length of the sequences in the training data + 10 to include space for special tokens
helix_mrna_config = HelixmRNAConfig(device=device, batch_size=1, max_length=max(len(s) for s in train_sequences["sequence"])+10)
helix_mrna = HelixmRNA(helix_mrna_config)


def replace_chars(text, char_list, replacement):
    for char in char_list:
        text = text.replace(char, replacement)
    return text


replacement = ""
chars_to_replace = ["X", "-"]

#processed_train_data = helix_mrna.process_data([s.replace("X", "") for s in train_sequences['sequence']])
processed_train_data = helix_mrna.process_data([replace_chars(s, chars_to_replace, replacement) for s in train_sequences['sequence']])

embeddings = helix_mrna.get_embeddings(processed_train_data)
embeddings = embeddings[:, -2, :]
print(embeddings.shape)
print(embeddings[:1][0:10])


df = pd.DataFrame(embeddings)
df.to_csv("/kaggle/working/train_helix_embeddings.csv", index=False)

