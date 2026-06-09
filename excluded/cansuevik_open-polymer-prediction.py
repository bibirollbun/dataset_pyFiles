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


# GitHub projenizi klonlayın
!git clone https://github.com/Cansu126/open-polymer-prediction.git

# Proje klasörüne geçiş yapın
%cd open-polymer-prediction



# Gereksinim dosyasındaki kütüphaneleri yükleyin
!pip install -r requirements.txt



# Kaggle dataset'i indir
!kaggle datasets download -d <dataset-id>

# Veriyi çıkartın
!unzip <dataset-name>.zip -d ./data



!python train_and_save_model.py


# PyTorch ve torch_geometric kütüphanesini yükleyin
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install torch-scatter -f https://data.pyg.org/whl/cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/cpu.html
!pip install torch-cluster -f https://data.pyg.org/whl/cpu.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/cpu.html
!pip install torch-geometric



!python train_and_save_model.py



!pip install rdkit



!python train_and_save_model.py



# Uygun PyTorch ve TorchVision sürümlerini yükleyin
!pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cpu



!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



!python train_and_save_model.py



!pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu



!pip install --upgrade transformers



!python train_and_save_model.py



import os
print(os.listdir('/kaggle/input'))



import pandas as pd

train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(train_df.head())



!mkdir -p data
!cp /kaggle/input/neurips-open-polymer-prediction-2025/train.csv data/



!python train_and_save_model.py


from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CCO"  # Örnek SMILES
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)  # Hidrojenleri ekle

# 3D konformer oluştur
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

# Artık mol.getNumConformers() >= 1 olmalı
print(mol.GetNumConformers())  # 1 veya daha fazla çıktı alırsınız

# Sonra 3D özellik hesaplayabilirsiniz



!python train_and_save_model.py

