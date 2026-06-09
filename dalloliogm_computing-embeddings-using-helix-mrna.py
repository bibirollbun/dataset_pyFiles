import subprocess
import torch
import pandas as pd


!pip install helical -q --target=/kaggle/working/ --upgrade #--use-deprecated=legacy-resolver
from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel


!pip uninstall -y cupy -q
!pip uninstall -y cupy-cuda12x -q
!pip install cupy-cuda11x -q
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

#pip install helical -q --target=/kaggle/working/ --upgrade
!pip install -q --upgrade git+https://github.com/helicalAI/helical.git
from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel




!pip install helical -q # --target=/kaggle/working/ --upgrade #--use-deprecated=legacy-resolver
from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel



# Load data

train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")


from helical.models.helix_mrna import HelixmRNAConfig, HelixmRNA, HelixmRNAFineTuningModel


device = "cuda" if torch.cuda.is_available() else "cpu"


# We set the max length to the maximum length of the sequences in the training data + 10 to include space for special tokens
helix_mrna_config = HelixmRNAConfig(device=device, batch_size=1, max_length=max(len(s) for s in train_sequences["sequence"])+10)
helix_mrna = HelixmRNA(helix_mrna_config)




processed_train_data = helix_mrna.process_data([s.replace("X", "") for s in train_sequences['sequence']])
embeddings = helix_mrna.get_embeddings(processed_train_data)
embeddings = embeddings[:, -2, :]
print(embeddings.shape)
print(embeddings[:1][0:10])


embeddings.to_csv("/kaggle/output/train_helix_embeddings.csv")

