import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train.tail(3)


train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_labels.tail(3)


val = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
val.tail(3)


train['description'].value_counts()


train['all_sequences'].value_counts()


!pip install arnie


# Install Eternafold
!conda config --set auto_update_conda false
!conda install -c bioconda eternafold --yes


%env ETERNAFOLD_PATH=/opt/conda/bin/eternafold-bin
%env ETERNAFOLD_PARAMETERS=/opt/conda/lib/eternafold-lib/parameters/EternaFoldParams.v1


fireball = train[(train['description']=='Crystal structure of a substrate-bound full H/ACA RNP from Pyrococcus furiosus')].reset_index(drop=True)
fireball.head()


sequence = fireball[fireball["target_id"] == "3HAY_E"]["sequence"][0]#.compute()
sequence


from arnie.mfe import mfe
fireball_sequence = "GGCUGCCUGGGUCCGCCUUGAGUGCCCGGGUGAGAAGCAUGAUCCCGGGUAAUUAUGGCGGACCCACAGAU"
structure = mfe(fireball_sequence,package="eternafold")
print(structure)


!pip install draw_rna

from draw_rna.ipynb_draw import draw_struct


draw_struct(fireball_sequence, structure)


from arnie.bpps import bpps
bpps(sequence,package="eternafold")

